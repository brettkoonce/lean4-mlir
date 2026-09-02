module @m {
  func.func @resnet34_momwd00_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %v0 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<32x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<32x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<32x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<32x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<32x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<32x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<32x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<32x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<32x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<32x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<32x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<32x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<32x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<32x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<32x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<32x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<32x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<32x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<32x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<32x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<32x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<32x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<32x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<32x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<32x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<32x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<32x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<32x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<32x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<32x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<32x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<32x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v88 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<32x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<32x64x56x56xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v96 = stablehlo.convolution(%v95, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<32x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v103 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v104 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v105 = stablehlo.broadcast_in_dim %v104, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v106 = stablehlo.divide %v105, %v102 : tensor<32x64x56x56xf32>
    %v107 = stablehlo.subtract %v100, %v106 : tensor<32x64x56x56xf32>
    %v108 = stablehlo.multiply %v107, %v107 : tensor<32x64x56x56xf32>
    %v109 = stablehlo.reduce(%v108 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v111 = stablehlo.divide %v110, %v102 : tensor<32x64x56x56xf32>
    %v112 = stablehlo.add %v111, %v103 : tensor<32x64x56x56xf32>
    %v113 = stablehlo.rsqrt %v112 : tensor<32x64x56x56xf32>
    %v114 = stablehlo.multiply %v107, %v113 : tensor<32x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v116 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v117 = stablehlo.multiply %v114, %v115 : tensor<32x64x56x56xf32>
    %v118 = stablehlo.add %v117, %v116 : tensor<32x64x56x56xf32>
    %v119 = stablehlo.reshape %v118 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v121 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v122 = stablehlo.maximum %v120, %v121 : tensor<32x64x56x56xf32>
    %v123 = stablehlo.reshape %v122 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v125 = stablehlo.convolution(%v124, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v126 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v127 = stablehlo.add %v125, %v126 : tensor<32x64x56x56xf32>
    %v128 = stablehlo.reshape %v127 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v131 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v132 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v133 = stablehlo.reduce(%v129 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v134 = stablehlo.broadcast_in_dim %v133, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v135 = stablehlo.divide %v134, %v131 : tensor<32x64x56x56xf32>
    %v136 = stablehlo.subtract %v129, %v135 : tensor<32x64x56x56xf32>
    %v137 = stablehlo.multiply %v136, %v136 : tensor<32x64x56x56xf32>
    %v138 = stablehlo.reduce(%v137 init: %v130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v139 = stablehlo.broadcast_in_dim %v138, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v140 = stablehlo.divide %v139, %v131 : tensor<32x64x56x56xf32>
    %v141 = stablehlo.add %v140, %v132 : tensor<32x64x56x56xf32>
    %v142 = stablehlo.rsqrt %v141 : tensor<32x64x56x56xf32>
    %v143 = stablehlo.multiply %v136, %v142 : tensor<32x64x56x56xf32>
    %v144 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v146 = stablehlo.multiply %v143, %v144 : tensor<32x64x56x56xf32>
    %v147 = stablehlo.add %v146, %v145 : tensor<32x64x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v150 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v151 = stablehlo.add %v149, %v150 : tensor<32x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<32x64x56x56xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v157 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v158 = stablehlo.convolution(%v157, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v160 = stablehlo.add %v158, %v159 : tensor<32x64x56x56xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v164 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v165 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v166 = stablehlo.reduce(%v162 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v167 = stablehlo.broadcast_in_dim %v166, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v168 = stablehlo.divide %v167, %v164 : tensor<32x64x56x56xf32>
    %v169 = stablehlo.subtract %v162, %v168 : tensor<32x64x56x56xf32>
    %v170 = stablehlo.multiply %v169, %v169 : tensor<32x64x56x56xf32>
    %v171 = stablehlo.reduce(%v170 init: %v163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v172 = stablehlo.broadcast_in_dim %v171, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v173 = stablehlo.divide %v172, %v164 : tensor<32x64x56x56xf32>
    %v174 = stablehlo.add %v173, %v165 : tensor<32x64x56x56xf32>
    %v175 = stablehlo.rsqrt %v174 : tensor<32x64x56x56xf32>
    %v176 = stablehlo.multiply %v169, %v175 : tensor<32x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v178 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v179 = stablehlo.multiply %v176, %v177 : tensor<32x64x56x56xf32>
    %v180 = stablehlo.add %v179, %v178 : tensor<32x64x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v183 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v184 = stablehlo.maximum %v182, %v183 : tensor<32x64x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v187 = stablehlo.convolution(%v186, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<32x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<32x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<32x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<32x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<32x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<32x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<32x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<32x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<32x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<32x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v212 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v213 = stablehlo.add %v211, %v212 : tensor<32x64x56x56xf32>
    %v214 = stablehlo.reshape %v213 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v216 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v217 = stablehlo.maximum %v215, %v216 : tensor<32x64x56x56xf32>
    %v218 = stablehlo.reshape %v217 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v220 = stablehlo.convolution(%v219, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v221 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v222 = stablehlo.add %v220, %v221 : tensor<32x128x28x28xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v226 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v227 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v228 = stablehlo.reduce(%v224 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v229 = stablehlo.broadcast_in_dim %v228, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v230 = stablehlo.divide %v229, %v226 : tensor<32x128x28x28xf32>
    %v231 = stablehlo.subtract %v224, %v230 : tensor<32x128x28x28xf32>
    %v232 = stablehlo.multiply %v231, %v231 : tensor<32x128x28x28xf32>
    %v233 = stablehlo.reduce(%v232 init: %v225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v234 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v235 = stablehlo.divide %v234, %v226 : tensor<32x128x28x28xf32>
    %v236 = stablehlo.add %v235, %v227 : tensor<32x128x28x28xf32>
    %v237 = stablehlo.rsqrt %v236 : tensor<32x128x28x28xf32>
    %v238 = stablehlo.multiply %v231, %v237 : tensor<32x128x28x28xf32>
    %v239 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v241 = stablehlo.multiply %v238, %v239 : tensor<32x128x28x28xf32>
    %v242 = stablehlo.add %v241, %v240 : tensor<32x128x28x28xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<32x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v249 = stablehlo.convolution(%v248, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<32x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<32x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<32x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<32x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<32x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<32x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<32x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<32x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<32x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<32x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v273 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v274 = stablehlo.convolution(%v273, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v275 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v276 = stablehlo.add %v274, %v275 : tensor<32x128x28x28xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v280 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v281 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v282 = stablehlo.reduce(%v278 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v283 = stablehlo.broadcast_in_dim %v282, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v284 = stablehlo.divide %v283, %v280 : tensor<32x128x28x28xf32>
    %v285 = stablehlo.subtract %v278, %v284 : tensor<32x128x28x28xf32>
    %v286 = stablehlo.multiply %v285, %v285 : tensor<32x128x28x28xf32>
    %v287 = stablehlo.reduce(%v286 init: %v279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v280 : tensor<32x128x28x28xf32>
    %v290 = stablehlo.add %v289, %v281 : tensor<32x128x28x28xf32>
    %v291 = stablehlo.rsqrt %v290 : tensor<32x128x28x28xf32>
    %v292 = stablehlo.multiply %v285, %v291 : tensor<32x128x28x28xf32>
    %v293 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v294 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v295 = stablehlo.multiply %v292, %v293 : tensor<32x128x28x28xf32>
    %v296 = stablehlo.add %v295, %v294 : tensor<32x128x28x28xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v298 = stablehlo.reshape %v272 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v299 = stablehlo.reshape %v297 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<32x128x28x28xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v303 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v304 = stablehlo.maximum %v302, %v303 : tensor<32x128x28x28xf32>
    %v305 = stablehlo.reshape %v304 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v307 = stablehlo.convolution(%v306, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v308 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v309 = stablehlo.add %v307, %v308 : tensor<32x128x28x28xf32>
    %v310 = stablehlo.reshape %v309 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v313 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v314 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v315 = stablehlo.reduce(%v311 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v316 = stablehlo.broadcast_in_dim %v315, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v317 = stablehlo.divide %v316, %v313 : tensor<32x128x28x28xf32>
    %v318 = stablehlo.subtract %v311, %v317 : tensor<32x128x28x28xf32>
    %v319 = stablehlo.multiply %v318, %v318 : tensor<32x128x28x28xf32>
    %v320 = stablehlo.reduce(%v319 init: %v312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v321 = stablehlo.broadcast_in_dim %v320, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v322 = stablehlo.divide %v321, %v313 : tensor<32x128x28x28xf32>
    %v323 = stablehlo.add %v322, %v314 : tensor<32x128x28x28xf32>
    %v324 = stablehlo.rsqrt %v323 : tensor<32x128x28x28xf32>
    %v325 = stablehlo.multiply %v318, %v324 : tensor<32x128x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v327 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v328 = stablehlo.multiply %v325, %v326 : tensor<32x128x28x28xf32>
    %v329 = stablehlo.add %v328, %v327 : tensor<32x128x28x28xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v332 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v333 = stablehlo.maximum %v331, %v332 : tensor<32x128x28x28xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v336 = stablehlo.convolution(%v335, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v337 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v338 = stablehlo.add %v336, %v337 : tensor<32x128x28x28xf32>
    %v339 = stablehlo.reshape %v338 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v340 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v342 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v344 = stablehlo.reduce(%v340 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v345 = stablehlo.broadcast_in_dim %v344, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v346 = stablehlo.divide %v345, %v342 : tensor<32x128x28x28xf32>
    %v347 = stablehlo.subtract %v340, %v346 : tensor<32x128x28x28xf32>
    %v348 = stablehlo.multiply %v347, %v347 : tensor<32x128x28x28xf32>
    %v349 = stablehlo.reduce(%v348 init: %v341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v350 = stablehlo.broadcast_in_dim %v349, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v351 = stablehlo.divide %v350, %v342 : tensor<32x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v343 : tensor<32x128x28x28xf32>
    %v353 = stablehlo.rsqrt %v352 : tensor<32x128x28x28xf32>
    %v354 = stablehlo.multiply %v347, %v353 : tensor<32x128x28x28xf32>
    %v355 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v357 = stablehlo.multiply %v354, %v355 : tensor<32x128x28x28xf32>
    %v358 = stablehlo.add %v357, %v356 : tensor<32x128x28x28xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v361 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v362 = stablehlo.add %v360, %v361 : tensor<32x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<32x128x28x28xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v369 = stablehlo.convolution(%v368, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<32x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<32x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<32x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<32x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<32x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<32x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<32x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<32x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<32x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<32x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v394 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v395 = stablehlo.maximum %v393, %v394 : tensor<32x128x28x28xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v398 = stablehlo.convolution(%v397, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v399 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v400 = stablehlo.add %v398, %v399 : tensor<32x128x28x28xf32>
    %v401 = stablehlo.reshape %v400 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v404 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v405 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v406 = stablehlo.reduce(%v402 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v407 = stablehlo.broadcast_in_dim %v406, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v408 = stablehlo.divide %v407, %v404 : tensor<32x128x28x28xf32>
    %v409 = stablehlo.subtract %v402, %v408 : tensor<32x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v409 : tensor<32x128x28x28xf32>
    %v411 = stablehlo.reduce(%v410 init: %v403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v412 = stablehlo.broadcast_in_dim %v411, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v413 = stablehlo.divide %v412, %v404 : tensor<32x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v405 : tensor<32x128x28x28xf32>
    %v415 = stablehlo.rsqrt %v414 : tensor<32x128x28x28xf32>
    %v416 = stablehlo.multiply %v409, %v415 : tensor<32x128x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v419 = stablehlo.multiply %v416, %v417 : tensor<32x128x28x28xf32>
    %v420 = stablehlo.add %v419, %v418 : tensor<32x128x28x28xf32>
    %v421 = stablehlo.reshape %v420 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v423 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v424 = stablehlo.add %v422, %v423 : tensor<32x128x28x28xf32>
    %v425 = stablehlo.reshape %v424 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v426 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v427 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v428 = stablehlo.maximum %v426, %v427 : tensor<32x128x28x28xf32>
    %v429 = stablehlo.reshape %v428 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v431 = stablehlo.convolution(%v430, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v432 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v433 = stablehlo.add %v431, %v432 : tensor<32x128x28x28xf32>
    %v434 = stablehlo.reshape %v433 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v436 = stablehlo.constant dense<0.0> : tensor<f32>
    %v437 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v438 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v439 = stablehlo.reduce(%v435 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v440 = stablehlo.broadcast_in_dim %v439, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v441 = stablehlo.divide %v440, %v437 : tensor<32x128x28x28xf32>
    %v442 = stablehlo.subtract %v435, %v441 : tensor<32x128x28x28xf32>
    %v443 = stablehlo.multiply %v442, %v442 : tensor<32x128x28x28xf32>
    %v444 = stablehlo.reduce(%v443 init: %v436) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v445 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v446 = stablehlo.divide %v445, %v437 : tensor<32x128x28x28xf32>
    %v447 = stablehlo.add %v446, %v438 : tensor<32x128x28x28xf32>
    %v448 = stablehlo.rsqrt %v447 : tensor<32x128x28x28xf32>
    %v449 = stablehlo.multiply %v442, %v448 : tensor<32x128x28x28xf32>
    %v450 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v452 = stablehlo.multiply %v449, %v450 : tensor<32x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v451 : tensor<32x128x28x28xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<32x128x28x28xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v460 = stablehlo.convolution(%v459, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<32x128x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<32x128x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<32x128x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<32x128x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<32x128x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<32x128x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<32x128x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<32x128x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<32x128x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<32x128x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v485 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<32x128x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<32x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v493 = stablehlo.convolution(%v492, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<32x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<32x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<32x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<32x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<32x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<32x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<32x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<32x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<32x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<32x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v519 = stablehlo.maximum %v517, %v518 : tensor<32x256x14x14xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<32x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<32x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<32x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<32x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<32x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<32x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<32x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<32x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<32x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<32x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v546 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v547 = stablehlo.convolution(%v546, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v548 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<32x256x14x14xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v554 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v555 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v556 = stablehlo.broadcast_in_dim %v555, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v557 = stablehlo.divide %v556, %v553 : tensor<32x256x14x14xf32>
    %v558 = stablehlo.subtract %v551, %v557 : tensor<32x256x14x14xf32>
    %v559 = stablehlo.multiply %v558, %v558 : tensor<32x256x14x14xf32>
    %v560 = stablehlo.reduce(%v559 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v562 = stablehlo.divide %v561, %v553 : tensor<32x256x14x14xf32>
    %v563 = stablehlo.add %v562, %v554 : tensor<32x256x14x14xf32>
    %v564 = stablehlo.rsqrt %v563 : tensor<32x256x14x14xf32>
    %v565 = stablehlo.multiply %v558, %v564 : tensor<32x256x14x14xf32>
    %v566 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v567 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v568 = stablehlo.multiply %v565, %v566 : tensor<32x256x14x14xf32>
    %v569 = stablehlo.add %v568, %v567 : tensor<32x256x14x14xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v571 = stablehlo.reshape %v545 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v572 = stablehlo.reshape %v570 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<32x256x14x14xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v579 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v580 = stablehlo.convolution(%v579, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v581 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v582 = stablehlo.add %v580, %v581 : tensor<32x256x14x14xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v586 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v588 = stablehlo.reduce(%v584 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v589 = stablehlo.broadcast_in_dim %v588, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v590 = stablehlo.divide %v589, %v586 : tensor<32x256x14x14xf32>
    %v591 = stablehlo.subtract %v584, %v590 : tensor<32x256x14x14xf32>
    %v592 = stablehlo.multiply %v591, %v591 : tensor<32x256x14x14xf32>
    %v593 = stablehlo.reduce(%v592 init: %v585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v594 = stablehlo.broadcast_in_dim %v593, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v595 = stablehlo.divide %v594, %v586 : tensor<32x256x14x14xf32>
    %v596 = stablehlo.add %v595, %v587 : tensor<32x256x14x14xf32>
    %v597 = stablehlo.rsqrt %v596 : tensor<32x256x14x14xf32>
    %v598 = stablehlo.multiply %v591, %v597 : tensor<32x256x14x14xf32>
    %v599 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v600 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v601 = stablehlo.multiply %v598, %v599 : tensor<32x256x14x14xf32>
    %v602 = stablehlo.add %v601, %v600 : tensor<32x256x14x14xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v605 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v606 = stablehlo.maximum %v604, %v605 : tensor<32x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v609 = stablehlo.convolution(%v608, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v610 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<32x256x14x14xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<32x256x14x14xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<32x256x14x14xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<32x256x14x14xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<32x256x14x14xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<32x256x14x14xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<32x256x14x14xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<32x256x14x14xf32>
    %v628 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v629 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<32x256x14x14xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<32x256x14x14xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v634 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<32x256x14x14xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v639 = stablehlo.maximum %v637, %v638 : tensor<32x256x14x14xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v642 = stablehlo.convolution(%v641, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v643 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v644 = stablehlo.add %v642, %v643 : tensor<32x256x14x14xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v647 = stablehlo.constant dense<0.0> : tensor<f32>
    %v648 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v649 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v650 = stablehlo.reduce(%v646 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v651 = stablehlo.broadcast_in_dim %v650, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v652 = stablehlo.divide %v651, %v648 : tensor<32x256x14x14xf32>
    %v653 = stablehlo.subtract %v646, %v652 : tensor<32x256x14x14xf32>
    %v654 = stablehlo.multiply %v653, %v653 : tensor<32x256x14x14xf32>
    %v655 = stablehlo.reduce(%v654 init: %v647) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v656 = stablehlo.broadcast_in_dim %v655, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v657 = stablehlo.divide %v656, %v648 : tensor<32x256x14x14xf32>
    %v658 = stablehlo.add %v657, %v649 : tensor<32x256x14x14xf32>
    %v659 = stablehlo.rsqrt %v658 : tensor<32x256x14x14xf32>
    %v660 = stablehlo.multiply %v653, %v659 : tensor<32x256x14x14xf32>
    %v661 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v663 = stablehlo.multiply %v660, %v661 : tensor<32x256x14x14xf32>
    %v664 = stablehlo.add %v663, %v662 : tensor<32x256x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<32x256x14x14xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v671 = stablehlo.convolution(%v670, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v672 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<32x256x14x14xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v678 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v679 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v680 = stablehlo.broadcast_in_dim %v679, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v681 = stablehlo.divide %v680, %v677 : tensor<32x256x14x14xf32>
    %v682 = stablehlo.subtract %v675, %v681 : tensor<32x256x14x14xf32>
    %v683 = stablehlo.multiply %v682, %v682 : tensor<32x256x14x14xf32>
    %v684 = stablehlo.reduce(%v683 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v686 = stablehlo.divide %v685, %v677 : tensor<32x256x14x14xf32>
    %v687 = stablehlo.add %v686, %v678 : tensor<32x256x14x14xf32>
    %v688 = stablehlo.rsqrt %v687 : tensor<32x256x14x14xf32>
    %v689 = stablehlo.multiply %v682, %v688 : tensor<32x256x14x14xf32>
    %v690 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v691 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v692 = stablehlo.multiply %v689, %v690 : tensor<32x256x14x14xf32>
    %v693 = stablehlo.add %v692, %v691 : tensor<32x256x14x14xf32>
    %v694 = stablehlo.reshape %v693 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v695 = stablehlo.reshape %v694 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v696 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v697 = stablehlo.add %v695, %v696 : tensor<32x256x14x14xf32>
    %v698 = stablehlo.reshape %v697 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v700 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v701 = stablehlo.maximum %v699, %v700 : tensor<32x256x14x14xf32>
    %v702 = stablehlo.reshape %v701 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v704 = stablehlo.convolution(%v703, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<32x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<32x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<32x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<32x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<32x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<32x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<32x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<32x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<32x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<32x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v728 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v729 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v730 = stablehlo.maximum %v728, %v729 : tensor<32x256x14x14xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v733 = stablehlo.convolution(%v732, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v735 = stablehlo.add %v733, %v734 : tensor<32x256x14x14xf32>
    %v736 = stablehlo.reshape %v735 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v738 = stablehlo.constant dense<0.0> : tensor<f32>
    %v739 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v741 = stablehlo.reduce(%v737 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v743 = stablehlo.divide %v742, %v739 : tensor<32x256x14x14xf32>
    %v744 = stablehlo.subtract %v737, %v743 : tensor<32x256x14x14xf32>
    %v745 = stablehlo.multiply %v744, %v744 : tensor<32x256x14x14xf32>
    %v746 = stablehlo.reduce(%v745 init: %v738) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v747 = stablehlo.broadcast_in_dim %v746, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v748 = stablehlo.divide %v747, %v739 : tensor<32x256x14x14xf32>
    %v749 = stablehlo.add %v748, %v740 : tensor<32x256x14x14xf32>
    %v750 = stablehlo.rsqrt %v749 : tensor<32x256x14x14xf32>
    %v751 = stablehlo.multiply %v744, %v750 : tensor<32x256x14x14xf32>
    %v752 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v754 = stablehlo.multiply %v751, %v752 : tensor<32x256x14x14xf32>
    %v755 = stablehlo.add %v754, %v753 : tensor<32x256x14x14xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v758 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v759 = stablehlo.add %v757, %v758 : tensor<32x256x14x14xf32>
    %v760 = stablehlo.reshape %v759 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v762 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v763 = stablehlo.maximum %v761, %v762 : tensor<32x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v766 = stablehlo.convolution(%v765, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v767 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<32x256x14x14xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v773 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v774 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v772 : tensor<32x256x14x14xf32>
    %v777 = stablehlo.subtract %v770, %v776 : tensor<32x256x14x14xf32>
    %v778 = stablehlo.multiply %v777, %v777 : tensor<32x256x14x14xf32>
    %v779 = stablehlo.reduce(%v778 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v781 = stablehlo.divide %v780, %v772 : tensor<32x256x14x14xf32>
    %v782 = stablehlo.add %v781, %v773 : tensor<32x256x14x14xf32>
    %v783 = stablehlo.rsqrt %v782 : tensor<32x256x14x14xf32>
    %v784 = stablehlo.multiply %v777, %v783 : tensor<32x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v786 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v787 = stablehlo.multiply %v784, %v785 : tensor<32x256x14x14xf32>
    %v788 = stablehlo.add %v787, %v786 : tensor<32x256x14x14xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v790 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v791 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v792 = stablehlo.maximum %v790, %v791 : tensor<32x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v795 = stablehlo.convolution(%v794, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<32x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<32x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<32x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<32x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<32x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<32x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<32x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<32x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<32x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<32x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v820 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<32x256x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v825 = stablehlo.maximum %v823, %v824 : tensor<32x256x14x14xf32>
    %v826 = stablehlo.reshape %v825 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v828 = stablehlo.convolution(%v827, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<32x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<32x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<32x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<32x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<32x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<32x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<32x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<32x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<32x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<32x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<32x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v857 = stablehlo.convolution(%v856, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v858 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v859 = stablehlo.add %v857, %v858 : tensor<32x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v863 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v864 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v865 = stablehlo.reduce(%v861 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v866 = stablehlo.broadcast_in_dim %v865, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v867 = stablehlo.divide %v866, %v863 : tensor<32x256x14x14xf32>
    %v868 = stablehlo.subtract %v861, %v867 : tensor<32x256x14x14xf32>
    %v869 = stablehlo.multiply %v868, %v868 : tensor<32x256x14x14xf32>
    %v870 = stablehlo.reduce(%v869 init: %v862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v871 = stablehlo.broadcast_in_dim %v870, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v872 = stablehlo.divide %v871, %v863 : tensor<32x256x14x14xf32>
    %v873 = stablehlo.add %v872, %v864 : tensor<32x256x14x14xf32>
    %v874 = stablehlo.rsqrt %v873 : tensor<32x256x14x14xf32>
    %v875 = stablehlo.multiply %v868, %v874 : tensor<32x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v877 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v878 = stablehlo.multiply %v875, %v876 : tensor<32x256x14x14xf32>
    %v879 = stablehlo.add %v878, %v877 : tensor<32x256x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v882 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v883 = stablehlo.add %v881, %v882 : tensor<32x256x14x14xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v885 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v886 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v887 = stablehlo.maximum %v885, %v886 : tensor<32x256x14x14xf32>
    %v888 = stablehlo.reshape %v887 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v890 = stablehlo.convolution(%v889, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v891 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v892 = stablehlo.add %v890, %v891 : tensor<32x512x7x7xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v896 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v897 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v898 = stablehlo.reduce(%v894 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v899 = stablehlo.broadcast_in_dim %v898, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v900 = stablehlo.divide %v899, %v896 : tensor<32x512x7x7xf32>
    %v901 = stablehlo.subtract %v894, %v900 : tensor<32x512x7x7xf32>
    %v902 = stablehlo.multiply %v901, %v901 : tensor<32x512x7x7xf32>
    %v903 = stablehlo.reduce(%v902 init: %v895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v905 = stablehlo.divide %v904, %v896 : tensor<32x512x7x7xf32>
    %v906 = stablehlo.add %v905, %v897 : tensor<32x512x7x7xf32>
    %v907 = stablehlo.rsqrt %v906 : tensor<32x512x7x7xf32>
    %v908 = stablehlo.multiply %v901, %v907 : tensor<32x512x7x7xf32>
    %v909 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v910 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v911 = stablehlo.multiply %v908, %v909 : tensor<32x512x7x7xf32>
    %v912 = stablehlo.add %v911, %v910 : tensor<32x512x7x7xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v916 = stablehlo.maximum %v914, %v915 : tensor<32x512x7x7xf32>
    %v917 = stablehlo.reshape %v916 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v919 = stablehlo.convolution(%v918, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v920 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<32x512x7x7xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v925 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v926 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v927 = stablehlo.reduce(%v923 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v928 = stablehlo.broadcast_in_dim %v927, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v929 = stablehlo.divide %v928, %v925 : tensor<32x512x7x7xf32>
    %v930 = stablehlo.subtract %v923, %v929 : tensor<32x512x7x7xf32>
    %v931 = stablehlo.multiply %v930, %v930 : tensor<32x512x7x7xf32>
    %v932 = stablehlo.reduce(%v931 init: %v924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v934 = stablehlo.divide %v933, %v925 : tensor<32x512x7x7xf32>
    %v935 = stablehlo.add %v934, %v926 : tensor<32x512x7x7xf32>
    %v936 = stablehlo.rsqrt %v935 : tensor<32x512x7x7xf32>
    %v937 = stablehlo.multiply %v930, %v936 : tensor<32x512x7x7xf32>
    %v938 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v939 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v940 = stablehlo.multiply %v937, %v938 : tensor<32x512x7x7xf32>
    %v941 = stablehlo.add %v940, %v939 : tensor<32x512x7x7xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v943 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v944 = stablehlo.convolution(%v943, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x1x1xf32>) -> tensor<32x512x7x7xf32>
    %v945 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v946 = stablehlo.add %v944, %v945 : tensor<32x512x7x7xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v950 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v951 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v952 = stablehlo.reduce(%v948 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v953 = stablehlo.broadcast_in_dim %v952, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v954 = stablehlo.divide %v953, %v950 : tensor<32x512x7x7xf32>
    %v955 = stablehlo.subtract %v948, %v954 : tensor<32x512x7x7xf32>
    %v956 = stablehlo.multiply %v955, %v955 : tensor<32x512x7x7xf32>
    %v957 = stablehlo.reduce(%v956 init: %v949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v959 = stablehlo.divide %v958, %v950 : tensor<32x512x7x7xf32>
    %v960 = stablehlo.add %v959, %v951 : tensor<32x512x7x7xf32>
    %v961 = stablehlo.rsqrt %v960 : tensor<32x512x7x7xf32>
    %v962 = stablehlo.multiply %v955, %v961 : tensor<32x512x7x7xf32>
    %v963 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v965 = stablehlo.multiply %v962, %v963 : tensor<32x512x7x7xf32>
    %v966 = stablehlo.add %v965, %v964 : tensor<32x512x7x7xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v968 = stablehlo.reshape %v942 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v969 = stablehlo.reshape %v967 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v970 = stablehlo.add %v968, %v969 : tensor<32x512x7x7xf32>
    %v971 = stablehlo.reshape %v970 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v973 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v974 = stablehlo.maximum %v972, %v973 : tensor<32x512x7x7xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v977 = stablehlo.convolution(%v976, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x512x7x7xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v984 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v985 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v987 = stablehlo.divide %v986, %v983 : tensor<32x512x7x7xf32>
    %v988 = stablehlo.subtract %v981, %v987 : tensor<32x512x7x7xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x512x7x7xf32>
    %v990 = stablehlo.reduce(%v989 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v991 = stablehlo.broadcast_in_dim %v990, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v992 = stablehlo.divide %v991, %v983 : tensor<32x512x7x7xf32>
    %v993 = stablehlo.add %v992, %v984 : tensor<32x512x7x7xf32>
    %v994 = stablehlo.rsqrt %v993 : tensor<32x512x7x7xf32>
    %v995 = stablehlo.multiply %v988, %v994 : tensor<32x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v997 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v998 = stablehlo.multiply %v995, %v996 : tensor<32x512x7x7xf32>
    %v999 = stablehlo.add %v998, %v997 : tensor<32x512x7x7xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1002 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1003 = stablehlo.maximum %v1001, %v1002 : tensor<32x512x7x7xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1006 = stablehlo.convolution(%v1005, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<32x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<32x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<32x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<32x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<32x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<32x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<32x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<32x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<32x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<32x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1030 = stablehlo.reshape %v1029 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1031 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1032 = stablehlo.add %v1030, %v1031 : tensor<32x512x7x7xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1034 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1035 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1036 = stablehlo.maximum %v1034, %v1035 : tensor<32x512x7x7xf32>
    %v1037 = stablehlo.reshape %v1036 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1039 = stablehlo.convolution(%v1038, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1040 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1041 = stablehlo.add %v1039, %v1040 : tensor<32x512x7x7xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1045 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1046 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1047 = stablehlo.reduce(%v1043 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1048 = stablehlo.broadcast_in_dim %v1047, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1049 = stablehlo.divide %v1048, %v1045 : tensor<32x512x7x7xf32>
    %v1050 = stablehlo.subtract %v1043, %v1049 : tensor<32x512x7x7xf32>
    %v1051 = stablehlo.multiply %v1050, %v1050 : tensor<32x512x7x7xf32>
    %v1052 = stablehlo.reduce(%v1051 init: %v1044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1053 = stablehlo.broadcast_in_dim %v1052, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1054 = stablehlo.divide %v1053, %v1045 : tensor<32x512x7x7xf32>
    %v1055 = stablehlo.add %v1054, %v1046 : tensor<32x512x7x7xf32>
    %v1056 = stablehlo.rsqrt %v1055 : tensor<32x512x7x7xf32>
    %v1057 = stablehlo.multiply %v1050, %v1056 : tensor<32x512x7x7xf32>
    %v1058 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1059 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1060 = stablehlo.multiply %v1057, %v1058 : tensor<32x512x7x7xf32>
    %v1061 = stablehlo.add %v1060, %v1059 : tensor<32x512x7x7xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1064 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1065 = stablehlo.maximum %v1063, %v1064 : tensor<32x512x7x7xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1068 = stablehlo.convolution(%v1067, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1069 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1070 = stablehlo.add %v1068, %v1069 : tensor<32x512x7x7xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1074 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1075 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1076 = stablehlo.reduce(%v1072 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1078 = stablehlo.divide %v1077, %v1074 : tensor<32x512x7x7xf32>
    %v1079 = stablehlo.subtract %v1072, %v1078 : tensor<32x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1079, %v1079 : tensor<32x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1083 = stablehlo.divide %v1082, %v1074 : tensor<32x512x7x7xf32>
    %v1084 = stablehlo.add %v1083, %v1075 : tensor<32x512x7x7xf32>
    %v1085 = stablehlo.rsqrt %v1084 : tensor<32x512x7x7xf32>
    %v1086 = stablehlo.multiply %v1079, %v1085 : tensor<32x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1088 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1089 = stablehlo.multiply %v1086, %v1087 : tensor<32x512x7x7xf32>
    %v1090 = stablehlo.add %v1089, %v1088 : tensor<32x512x7x7xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1093 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<32x512x7x7xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1098 = stablehlo.maximum %v1096, %v1097 : tensor<32x512x7x7xf32>
    %v1099 = stablehlo.reshape %v1098 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1102 = stablehlo.reduce(%v1100 init: %v1101) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %v1103 = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %v1104 = stablehlo.divide %v1102, %v1103 : tensor<32x512xf32>
    %v1105 = stablehlo.dot_general %v1104, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
    %v1106 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1107 = stablehlo.add %v1105, %v1106 : tensor<32x10xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.exponential %v1108 : tensor<32x1x10xf32>
    %v1111 = stablehlo.reduce(%v1110 init: %v1109) applies stablehlo.add across dimensions = [2] : (tensor<32x1x10xf32>, tensor<f32>) -> tensor<32x1xf32>
    %v1112 = stablehlo.broadcast_in_dim %v1111, dims = [0, 1] : (tensor<32x1xf32>) -> tensor<32x1x10xf32>
    %v1113 = stablehlo.divide %v1110, %v1112 : tensor<32x1x10xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x1x10xf32>) -> tensor<32x10xf32>
    %v1115 = stablehlo.subtract %v1114, %onehot : tensor<32x10xf32>
    %v1116 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1117 = stablehlo.multiply %onehot, %v1116 : tensor<32x10xf32>
    %v1118 = stablehlo.add %v1115, %v1117 : tensor<32x10xf32>
    %v1119 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<32x10xf32>
    %v1121 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1122 = stablehlo.divide %v1120, %v1121 : tensor<32x10xf32>
    %v1123 = stablehlo.reshape %v1122 : (tensor<32x10xf32>) -> tensor<32x1x10xf32>
    %v1124 = stablehlo.dot_general %v1123, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1x10xf32>, tensor<512x10xf32>) -> tensor<32x1x512xf32>
    %v1125 = stablehlo.reshape %v1124 : (tensor<32x1x512xf32>) -> tensor<32x512xf32>
    %v1126 = stablehlo.dot_general %v1104, %v1122, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1128 = stablehlo.reduce(%v1122 init: %v1127) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %v1129 = stablehlo.broadcast_in_dim %v1125, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %v1130 = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %v1131 = stablehlo.divide %v1129, %v1130 : tensor<32x512x7x7xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1134 = stablehlo.reshape %v1095 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1135 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1136 = stablehlo.compare GT, %v1134, %v1135 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1137 = stablehlo.select %v1136, %v1133, %v1135 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1139 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1141 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1142 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1143 = stablehlo.reduce(%v1139 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1145 = stablehlo.divide %v1144, %v1141 : tensor<32x512x7x7xf32>
    %v1146 = stablehlo.subtract %v1139, %v1145 : tensor<32x512x7x7xf32>
    %v1147 = stablehlo.multiply %v1146, %v1146 : tensor<32x512x7x7xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1150 = stablehlo.divide %v1149, %v1141 : tensor<32x512x7x7xf32>
    %v1151 = stablehlo.add %v1150, %v1142 : tensor<32x512x7x7xf32>
    %v1152 = stablehlo.rsqrt %v1151 : tensor<32x512x7x7xf32>
    %v1153 = stablehlo.multiply %v1146, %v1152 : tensor<32x512x7x7xf32>
    %v1154 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1155 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1156 = stablehlo.multiply %v1154, %v1155 : tensor<32x512x7x7xf32>
    %v1157 = stablehlo.reduce(%v1156 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1158 = stablehlo.broadcast_in_dim %v1157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1159 = stablehlo.multiply %v1153, %v1156 : tensor<32x512x7x7xf32>
    %v1160 = stablehlo.reduce(%v1159 init: %v1140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1161 = stablehlo.broadcast_in_dim %v1160, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1162 = stablehlo.multiply %v1156, %v1141 : tensor<32x512x7x7xf32>
    %v1163 = stablehlo.subtract %v1162, %v1158 : tensor<32x512x7x7xf32>
    %v1164 = stablehlo.multiply %v1153, %v1161 : tensor<32x512x7x7xf32>
    %v1165 = stablehlo.subtract %v1163, %v1164 : tensor<32x512x7x7xf32>
    %v1166 = stablehlo.divide %v1152, %v1141 : tensor<32x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1166, %v1165 : tensor<32x512x7x7xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1169 = stablehlo.reshape %v1168 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1170 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1171 = stablehlo.transpose %v1170, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1172 = stablehlo.convolution(%v1169, %v1171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1175 = stablehlo.reshape %v1062 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1176 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1177 = stablehlo.compare GT, %v1175, %v1176 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1178 = stablehlo.select %v1177, %v1174, %v1176 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1180 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1182 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1183 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1184 = stablehlo.reduce(%v1180 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1185 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1186 = stablehlo.divide %v1185, %v1182 : tensor<32x512x7x7xf32>
    %v1187 = stablehlo.subtract %v1180, %v1186 : tensor<32x512x7x7xf32>
    %v1188 = stablehlo.multiply %v1187, %v1187 : tensor<32x512x7x7xf32>
    %v1189 = stablehlo.reduce(%v1188 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1190 = stablehlo.broadcast_in_dim %v1189, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1191 = stablehlo.divide %v1190, %v1182 : tensor<32x512x7x7xf32>
    %v1192 = stablehlo.add %v1191, %v1183 : tensor<32x512x7x7xf32>
    %v1193 = stablehlo.rsqrt %v1192 : tensor<32x512x7x7xf32>
    %v1194 = stablehlo.multiply %v1187, %v1193 : tensor<32x512x7x7xf32>
    %v1195 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1196 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1197 = stablehlo.multiply %v1195, %v1196 : tensor<32x512x7x7xf32>
    %v1198 = stablehlo.reduce(%v1197 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1199 = stablehlo.broadcast_in_dim %v1198, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1200 = stablehlo.multiply %v1194, %v1197 : tensor<32x512x7x7xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1203 = stablehlo.multiply %v1197, %v1182 : tensor<32x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1203, %v1199 : tensor<32x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1194, %v1202 : tensor<32x512x7x7xf32>
    %v1206 = stablehlo.subtract %v1204, %v1205 : tensor<32x512x7x7xf32>
    %v1207 = stablehlo.divide %v1193, %v1182 : tensor<32x512x7x7xf32>
    %v1208 = stablehlo.multiply %v1207, %v1206 : tensor<32x512x7x7xf32>
    %v1209 = stablehlo.reshape %v1208 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1211 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1212 = stablehlo.transpose %v1211, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1213 = stablehlo.convolution(%v1210, %v1212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1214 = stablehlo.reshape %v1213 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1216 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1217 = stablehlo.add %v1215, %v1216 : tensor<32x512x7x7xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1219 = stablehlo.reshape %v1037 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1220 = stablehlo.reshape %v1209 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1221 = stablehlo.transpose %v1219, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1222 = stablehlo.transpose %v1220, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1223 = stablehlo.convolution(%v1221, %v1222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1224 = stablehlo.transpose %v1223, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1225 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1227 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1228 = stablehlo.reduce(%v1225 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1229 = stablehlo.broadcast_in_dim %v1228, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1230 = stablehlo.divide %v1229, %v1227 : tensor<32x512x7x7xf32>
    %v1231 = stablehlo.subtract %v1225, %v1230 : tensor<32x512x7x7xf32>
    %v1232 = stablehlo.multiply %v1231, %v1231 : tensor<32x512x7x7xf32>
    %v1233 = stablehlo.reduce(%v1232 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1234 = stablehlo.broadcast_in_dim %v1233, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1235 = stablehlo.divide %v1234, %v1227 : tensor<32x512x7x7xf32>
    %v1236 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1237 = stablehlo.add %v1235, %v1236 : tensor<32x512x7x7xf32>
    %v1238 = stablehlo.rsqrt %v1237 : tensor<32x512x7x7xf32>
    %v1239 = stablehlo.multiply %v1231, %v1238 : tensor<32x512x7x7xf32>
    %v1240 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<32x512x7x7xf32>
    %v1242 = stablehlo.reduce(%v1241 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1243 = stablehlo.reshape %v1179 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1245 = stablehlo.reduce(%v1243 init: %v1244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1246 = stablehlo.reshape %v1066 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1247 = stablehlo.reshape %v1168 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1248 = stablehlo.transpose %v1246, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1249 = stablehlo.transpose %v1247, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1250 = stablehlo.convolution(%v1248, %v1249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1251 = stablehlo.transpose %v1250, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1252 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1254 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1255 = stablehlo.reduce(%v1252 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1256 = stablehlo.broadcast_in_dim %v1255, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1257 = stablehlo.divide %v1256, %v1254 : tensor<32x512x7x7xf32>
    %v1258 = stablehlo.subtract %v1252, %v1257 : tensor<32x512x7x7xf32>
    %v1259 = stablehlo.multiply %v1258, %v1258 : tensor<32x512x7x7xf32>
    %v1260 = stablehlo.reduce(%v1259 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1262 = stablehlo.divide %v1261, %v1254 : tensor<32x512x7x7xf32>
    %v1263 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1264 = stablehlo.add %v1262, %v1263 : tensor<32x512x7x7xf32>
    %v1265 = stablehlo.rsqrt %v1264 : tensor<32x512x7x7xf32>
    %v1266 = stablehlo.multiply %v1258, %v1265 : tensor<32x512x7x7xf32>
    %v1267 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1268 = stablehlo.multiply %v1267, %v1266 : tensor<32x512x7x7xf32>
    %v1269 = stablehlo.reduce(%v1268 init: %v1253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1270 = stablehlo.reshape %v1138 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1272 = stablehlo.reduce(%v1270 init: %v1271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1273 = stablehlo.reshape %v1218 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1274 = stablehlo.reshape %v1033 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1275 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1276 = stablehlo.compare GT, %v1274, %v1275 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1277 = stablehlo.select %v1276, %v1273, %v1275 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1279 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1281 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1282 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1283 = stablehlo.reduce(%v1279 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1284 = stablehlo.broadcast_in_dim %v1283, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1285 = stablehlo.divide %v1284, %v1281 : tensor<32x512x7x7xf32>
    %v1286 = stablehlo.subtract %v1279, %v1285 : tensor<32x512x7x7xf32>
    %v1287 = stablehlo.multiply %v1286, %v1286 : tensor<32x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1287 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1281 : tensor<32x512x7x7xf32>
    %v1291 = stablehlo.add %v1290, %v1282 : tensor<32x512x7x7xf32>
    %v1292 = stablehlo.rsqrt %v1291 : tensor<32x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1286, %v1292 : tensor<32x512x7x7xf32>
    %v1294 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1295 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1296 = stablehlo.multiply %v1294, %v1295 : tensor<32x512x7x7xf32>
    %v1297 = stablehlo.reduce(%v1296 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1298 = stablehlo.broadcast_in_dim %v1297, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1299 = stablehlo.multiply %v1293, %v1296 : tensor<32x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1299 init: %v1280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1302 = stablehlo.multiply %v1296, %v1281 : tensor<32x512x7x7xf32>
    %v1303 = stablehlo.subtract %v1302, %v1298 : tensor<32x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1293, %v1301 : tensor<32x512x7x7xf32>
    %v1305 = stablehlo.subtract %v1303, %v1304 : tensor<32x512x7x7xf32>
    %v1306 = stablehlo.divide %v1292, %v1281 : tensor<32x512x7x7xf32>
    %v1307 = stablehlo.multiply %v1306, %v1305 : tensor<32x512x7x7xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1310 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1311 = stablehlo.transpose %v1310, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1312 = stablehlo.convolution(%v1309, %v1311)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1315 = stablehlo.reshape %v1000 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1317 = stablehlo.compare GT, %v1315, %v1316 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1318 = stablehlo.select %v1317, %v1314, %v1316 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1320 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1322 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1323 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1324 = stablehlo.reduce(%v1320 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1326 = stablehlo.divide %v1325, %v1322 : tensor<32x512x7x7xf32>
    %v1327 = stablehlo.subtract %v1320, %v1326 : tensor<32x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1327, %v1327 : tensor<32x512x7x7xf32>
    %v1329 = stablehlo.reduce(%v1328 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1331 = stablehlo.divide %v1330, %v1322 : tensor<32x512x7x7xf32>
    %v1332 = stablehlo.add %v1331, %v1323 : tensor<32x512x7x7xf32>
    %v1333 = stablehlo.rsqrt %v1332 : tensor<32x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1327, %v1333 : tensor<32x512x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1336 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1337 = stablehlo.multiply %v1335, %v1336 : tensor<32x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1337 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1340 = stablehlo.multiply %v1334, %v1337 : tensor<32x512x7x7xf32>
    %v1341 = stablehlo.reduce(%v1340 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1343 = stablehlo.multiply %v1337, %v1322 : tensor<32x512x7x7xf32>
    %v1344 = stablehlo.subtract %v1343, %v1339 : tensor<32x512x7x7xf32>
    %v1345 = stablehlo.multiply %v1334, %v1342 : tensor<32x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1344, %v1345 : tensor<32x512x7x7xf32>
    %v1347 = stablehlo.divide %v1333, %v1322 : tensor<32x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1347, %v1346 : tensor<32x512x7x7xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1350 = stablehlo.reshape %v1349 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1351 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1352 = stablehlo.transpose %v1351, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1353 = stablehlo.convolution(%v1350, %v1352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1355 = stablehlo.reshape %v1354 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1356 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<32x512x7x7xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1359 = stablehlo.reshape %v975 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1360 = stablehlo.reshape %v1349 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1361 = stablehlo.transpose %v1359, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1362 = stablehlo.transpose %v1360, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1363 = stablehlo.convolution(%v1361, %v1362)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1364 = stablehlo.transpose %v1363, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1365 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1367 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1368 = stablehlo.reduce(%v1365 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1370 = stablehlo.divide %v1369, %v1367 : tensor<32x512x7x7xf32>
    %v1371 = stablehlo.subtract %v1365, %v1370 : tensor<32x512x7x7xf32>
    %v1372 = stablehlo.multiply %v1371, %v1371 : tensor<32x512x7x7xf32>
    %v1373 = stablehlo.reduce(%v1372 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1375 = stablehlo.divide %v1374, %v1367 : tensor<32x512x7x7xf32>
    %v1376 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<32x512x7x7xf32>
    %v1378 = stablehlo.rsqrt %v1377 : tensor<32x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1371, %v1378 : tensor<32x512x7x7xf32>
    %v1380 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1381 = stablehlo.multiply %v1380, %v1379 : tensor<32x512x7x7xf32>
    %v1382 = stablehlo.reduce(%v1381 init: %v1366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1383 = stablehlo.reshape %v1319 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1385 = stablehlo.reduce(%v1383 init: %v1384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1386 = stablehlo.reshape %v1004 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1308 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1388 = stablehlo.transpose %v1386, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1389 = stablehlo.transpose %v1387, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1390 = stablehlo.convolution(%v1388, %v1389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1391 = stablehlo.transpose %v1390, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1392 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1392 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1394 : tensor<32x512x7x7xf32>
    %v1398 = stablehlo.subtract %v1392, %v1397 : tensor<32x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<32x512x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1394 : tensor<32x512x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<32x512x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<32x512x7x7xf32>
    %v1406 = stablehlo.multiply %v1398, %v1405 : tensor<32x512x7x7xf32>
    %v1407 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1408 = stablehlo.multiply %v1407, %v1406 : tensor<32x512x7x7xf32>
    %v1409 = stablehlo.reduce(%v1408 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1410 = stablehlo.reshape %v1278 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.reduce(%v1410 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1413 = stablehlo.reshape %v1358 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1414 = stablehlo.reshape %v971 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1416 = stablehlo.compare GT, %v1414, %v1415 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1417 = stablehlo.select %v1416, %v1413, %v1415 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1418 = stablehlo.reshape %v1417 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1419 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1421 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1422 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1423 = stablehlo.reduce(%v1419 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1421 : tensor<32x512x7x7xf32>
    %v1426 = stablehlo.subtract %v1419, %v1425 : tensor<32x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1426 : tensor<32x512x7x7xf32>
    %v1428 = stablehlo.reduce(%v1427 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1430 = stablehlo.divide %v1429, %v1421 : tensor<32x512x7x7xf32>
    %v1431 = stablehlo.add %v1430, %v1422 : tensor<32x512x7x7xf32>
    %v1432 = stablehlo.rsqrt %v1431 : tensor<32x512x7x7xf32>
    %v1433 = stablehlo.multiply %v1426, %v1432 : tensor<32x512x7x7xf32>
    %v1434 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1434, %v1435 : tensor<32x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1433, %v1436 : tensor<32x512x7x7xf32>
    %v1440 = stablehlo.reduce(%v1439 init: %v1420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.broadcast_in_dim %v1440, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1442 = stablehlo.multiply %v1436, %v1421 : tensor<32x512x7x7xf32>
    %v1443 = stablehlo.subtract %v1442, %v1438 : tensor<32x512x7x7xf32>
    %v1444 = stablehlo.multiply %v1433, %v1441 : tensor<32x512x7x7xf32>
    %v1445 = stablehlo.subtract %v1443, %v1444 : tensor<32x512x7x7xf32>
    %v1446 = stablehlo.divide %v1432, %v1421 : tensor<32x512x7x7xf32>
    %v1447 = stablehlo.multiply %v1446, %v1445 : tensor<32x512x7x7xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1449 = stablehlo.reshape %v1448 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1450 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1451 = stablehlo.transpose %v1450, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1452 = stablehlo.convolution(%v1449, %v1451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1454 = stablehlo.reshape %v1453 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1455 = stablehlo.reshape %v913 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1456 = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %v1457 = stablehlo.compare GT, %v1455, %v1456 : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %v1458 = stablehlo.select %v1457, %v1454, %v1456 : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1460 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1463 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1462 : tensor<32x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1460, %v1466 : tensor<32x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<32x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1462 : tensor<32x512x7x7xf32>
    %v1472 = stablehlo.add %v1471, %v1463 : tensor<32x512x7x7xf32>
    %v1473 = stablehlo.rsqrt %v1472 : tensor<32x512x7x7xf32>
    %v1474 = stablehlo.multiply %v1467, %v1473 : tensor<32x512x7x7xf32>
    %v1475 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1476 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1475, %v1476 : tensor<32x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1480 = stablehlo.multiply %v1474, %v1477 : tensor<32x512x7x7xf32>
    %v1481 = stablehlo.reduce(%v1480 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.broadcast_in_dim %v1481, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1483 = stablehlo.multiply %v1477, %v1462 : tensor<32x512x7x7xf32>
    %v1484 = stablehlo.subtract %v1483, %v1479 : tensor<32x512x7x7xf32>
    %v1485 = stablehlo.multiply %v1474, %v1482 : tensor<32x512x7x7xf32>
    %v1486 = stablehlo.subtract %v1484, %v1485 : tensor<32x512x7x7xf32>
    %v1487 = stablehlo.divide %v1473, %v1462 : tensor<32x512x7x7xf32>
    %v1488 = stablehlo.multiply %v1487, %v1486 : tensor<32x512x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1492 = stablehlo.pad %v1490, %v1491, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1493 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1494 = stablehlo.transpose %v1493, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1495 = stablehlo.convolution(%v1492, %v1494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1497 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1500 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<32x512x7x7xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<32x512x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<32x512x7x7xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<32x512x7x7xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<32x512x7x7xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<32x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<32x512x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1513 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1514 = stablehlo.multiply %v1512, %v1513 : tensor<32x512x7x7xf32>
    %v1515 = stablehlo.reduce(%v1514 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1517 = stablehlo.multiply %v1511, %v1514 : tensor<32x512x7x7xf32>
    %v1518 = stablehlo.reduce(%v1517 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1514, %v1499 : tensor<32x512x7x7xf32>
    %v1521 = stablehlo.subtract %v1520, %v1516 : tensor<32x512x7x7xf32>
    %v1522 = stablehlo.multiply %v1511, %v1519 : tensor<32x512x7x7xf32>
    %v1523 = stablehlo.subtract %v1521, %v1522 : tensor<32x512x7x7xf32>
    %v1524 = stablehlo.divide %v1510, %v1499 : tensor<32x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1523 : tensor<32x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<32x512x7x7xf32>) -> tensor<32x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.pad %v1527, %v1528, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1530 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1531 = stablehlo.transpose %v1530, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1532 = stablehlo.convolution(%v1529, %v1531)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x1x1xf32>) -> tensor<32x256x14x14xf32>
    %v1533 = stablehlo.reshape %v1532 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1534 = stablehlo.reshape %v1496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1533 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1536 = stablehlo.add %v1534, %v1535 : tensor<32x256x14x14xf32>
    %v1537 = stablehlo.reshape %v1536 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1538 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1489 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.pad %v1539, %v1540, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1542 = stablehlo.transpose %v1538, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1543 = stablehlo.transpose %v1541, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1544 = stablehlo.convolution(%v1542, %v1543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1545 = stablehlo.transpose %v1544, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1546 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1548 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1549 = stablehlo.reduce(%v1546 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1551 = stablehlo.divide %v1550, %v1548 : tensor<32x512x7x7xf32>
    %v1552 = stablehlo.subtract %v1546, %v1551 : tensor<32x512x7x7xf32>
    %v1553 = stablehlo.multiply %v1552, %v1552 : tensor<32x512x7x7xf32>
    %v1554 = stablehlo.reduce(%v1553 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1555 = stablehlo.broadcast_in_dim %v1554, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1556 = stablehlo.divide %v1555, %v1548 : tensor<32x512x7x7xf32>
    %v1557 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1558 = stablehlo.add %v1556, %v1557 : tensor<32x512x7x7xf32>
    %v1559 = stablehlo.rsqrt %v1558 : tensor<32x512x7x7xf32>
    %v1560 = stablehlo.multiply %v1552, %v1559 : tensor<32x512x7x7xf32>
    %v1561 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1562 = stablehlo.multiply %v1561, %v1560 : tensor<32x512x7x7xf32>
    %v1563 = stablehlo.reduce(%v1562 init: %v1547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1564 = stablehlo.reshape %v1459 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1566 = stablehlo.reduce(%v1564 init: %v1565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1567 = stablehlo.reshape %v917 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1568 = stablehlo.reshape %v1448 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1569 = stablehlo.transpose %v1567, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1570 = stablehlo.transpose %v1568, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %v1571 = stablehlo.convolution(%v1569, %v1570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1572 = stablehlo.transpose %v1571, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1573 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1575 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1576 = stablehlo.reduce(%v1573 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1578 = stablehlo.divide %v1577, %v1575 : tensor<32x512x7x7xf32>
    %v1579 = stablehlo.subtract %v1573, %v1578 : tensor<32x512x7x7xf32>
    %v1580 = stablehlo.multiply %v1579, %v1579 : tensor<32x512x7x7xf32>
    %v1581 = stablehlo.reduce(%v1580 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1582 = stablehlo.broadcast_in_dim %v1581, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1583 = stablehlo.divide %v1582, %v1575 : tensor<32x512x7x7xf32>
    %v1584 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1585 = stablehlo.add %v1583, %v1584 : tensor<32x512x7x7xf32>
    %v1586 = stablehlo.rsqrt %v1585 : tensor<32x512x7x7xf32>
    %v1587 = stablehlo.multiply %v1579, %v1586 : tensor<32x512x7x7xf32>
    %v1588 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1589 = stablehlo.multiply %v1588, %v1587 : tensor<32x512x7x7xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1591 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1593 = stablehlo.reduce(%v1591 init: %v1592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1594 = stablehlo.reshape %v888 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1595 = stablehlo.reshape %v1526 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1597 = stablehlo.pad %v1595, %v1596, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %v1598 = stablehlo.transpose %v1594, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1599 = stablehlo.transpose %v1597, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %v1600 = stablehlo.convolution(%v1598, %v1599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x1x1xf32>
    %v1601 = stablehlo.transpose %v1600, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1602 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1604 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v1605 = stablehlo.reduce(%v1602 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1606 = stablehlo.broadcast_in_dim %v1605, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1607 = stablehlo.divide %v1606, %v1604 : tensor<32x512x7x7xf32>
    %v1608 = stablehlo.subtract %v1602, %v1607 : tensor<32x512x7x7xf32>
    %v1609 = stablehlo.multiply %v1608, %v1608 : tensor<32x512x7x7xf32>
    %v1610 = stablehlo.reduce(%v1609 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1611 = stablehlo.broadcast_in_dim %v1610, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v1612 = stablehlo.divide %v1611, %v1604 : tensor<32x512x7x7xf32>
    %v1613 = stablehlo.constant dense<1.0e-05> : tensor<32x512x7x7xf32>
    %v1614 = stablehlo.add %v1612, %v1613 : tensor<32x512x7x7xf32>
    %v1615 = stablehlo.rsqrt %v1614 : tensor<32x512x7x7xf32>
    %v1616 = stablehlo.multiply %v1608, %v1615 : tensor<32x512x7x7xf32>
    %v1617 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1618 = stablehlo.multiply %v1617, %v1616 : tensor<32x512x7x7xf32>
    %v1619 = stablehlo.reduce(%v1618 init: %v1603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1620 = stablehlo.reshape %v1418 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v1621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1622 = stablehlo.reduce(%v1620 init: %v1621) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1623 = stablehlo.reshape %v1537 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1624 = stablehlo.reshape %v884 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1625 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1626 = stablehlo.compare GT, %v1624, %v1625 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1627 = stablehlo.select %v1626, %v1623, %v1625 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1628 = stablehlo.reshape %v1627 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1629 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1631 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1632 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1633 = stablehlo.reduce(%v1629 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1634 = stablehlo.broadcast_in_dim %v1633, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1635 = stablehlo.divide %v1634, %v1631 : tensor<32x256x14x14xf32>
    %v1636 = stablehlo.subtract %v1629, %v1635 : tensor<32x256x14x14xf32>
    %v1637 = stablehlo.multiply %v1636, %v1636 : tensor<32x256x14x14xf32>
    %v1638 = stablehlo.reduce(%v1637 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1639 = stablehlo.broadcast_in_dim %v1638, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1640 = stablehlo.divide %v1639, %v1631 : tensor<32x256x14x14xf32>
    %v1641 = stablehlo.add %v1640, %v1632 : tensor<32x256x14x14xf32>
    %v1642 = stablehlo.rsqrt %v1641 : tensor<32x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1636, %v1642 : tensor<32x256x14x14xf32>
    %v1644 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1645 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1646 = stablehlo.multiply %v1644, %v1645 : tensor<32x256x14x14xf32>
    %v1647 = stablehlo.reduce(%v1646 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1649 = stablehlo.multiply %v1643, %v1646 : tensor<32x256x14x14xf32>
    %v1650 = stablehlo.reduce(%v1649 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1651 = stablehlo.broadcast_in_dim %v1650, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1652 = stablehlo.multiply %v1646, %v1631 : tensor<32x256x14x14xf32>
    %v1653 = stablehlo.subtract %v1652, %v1648 : tensor<32x256x14x14xf32>
    %v1654 = stablehlo.multiply %v1643, %v1651 : tensor<32x256x14x14xf32>
    %v1655 = stablehlo.subtract %v1653, %v1654 : tensor<32x256x14x14xf32>
    %v1656 = stablehlo.divide %v1642, %v1631 : tensor<32x256x14x14xf32>
    %v1657 = stablehlo.multiply %v1656, %v1655 : tensor<32x256x14x14xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1660 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1661 = stablehlo.transpose %v1660, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1662 = stablehlo.convolution(%v1659, %v1661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1665 = stablehlo.reshape %v851 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1666 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1667 = stablehlo.compare GT, %v1665, %v1666 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1668 = stablehlo.select %v1667, %v1664, %v1666 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1670 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1672 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1673 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1674 = stablehlo.reduce(%v1670 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1676 = stablehlo.divide %v1675, %v1672 : tensor<32x256x14x14xf32>
    %v1677 = stablehlo.subtract %v1670, %v1676 : tensor<32x256x14x14xf32>
    %v1678 = stablehlo.multiply %v1677, %v1677 : tensor<32x256x14x14xf32>
    %v1679 = stablehlo.reduce(%v1678 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1680 = stablehlo.broadcast_in_dim %v1679, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1681 = stablehlo.divide %v1680, %v1672 : tensor<32x256x14x14xf32>
    %v1682 = stablehlo.add %v1681, %v1673 : tensor<32x256x14x14xf32>
    %v1683 = stablehlo.rsqrt %v1682 : tensor<32x256x14x14xf32>
    %v1684 = stablehlo.multiply %v1677, %v1683 : tensor<32x256x14x14xf32>
    %v1685 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1686 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1687 = stablehlo.multiply %v1685, %v1686 : tensor<32x256x14x14xf32>
    %v1688 = stablehlo.reduce(%v1687 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1689 = stablehlo.broadcast_in_dim %v1688, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1690 = stablehlo.multiply %v1684, %v1687 : tensor<32x256x14x14xf32>
    %v1691 = stablehlo.reduce(%v1690 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1692 = stablehlo.broadcast_in_dim %v1691, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1693 = stablehlo.multiply %v1687, %v1672 : tensor<32x256x14x14xf32>
    %v1694 = stablehlo.subtract %v1693, %v1689 : tensor<32x256x14x14xf32>
    %v1695 = stablehlo.multiply %v1684, %v1692 : tensor<32x256x14x14xf32>
    %v1696 = stablehlo.subtract %v1694, %v1695 : tensor<32x256x14x14xf32>
    %v1697 = stablehlo.divide %v1683, %v1672 : tensor<32x256x14x14xf32>
    %v1698 = stablehlo.multiply %v1697, %v1696 : tensor<32x256x14x14xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1701 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1702 = stablehlo.transpose %v1701, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1703 = stablehlo.convolution(%v1700, %v1702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1706 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1707 = stablehlo.add %v1705, %v1706 : tensor<32x256x14x14xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1709 = stablehlo.reshape %v826 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1710 = stablehlo.reshape %v1699 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1711 = stablehlo.transpose %v1709, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1712 = stablehlo.transpose %v1710, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1713 = stablehlo.convolution(%v1711, %v1712)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1714 = stablehlo.transpose %v1713, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1715 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1717 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1718 = stablehlo.reduce(%v1715 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1719 = stablehlo.broadcast_in_dim %v1718, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1720 = stablehlo.divide %v1719, %v1717 : tensor<32x256x14x14xf32>
    %v1721 = stablehlo.subtract %v1715, %v1720 : tensor<32x256x14x14xf32>
    %v1722 = stablehlo.multiply %v1721, %v1721 : tensor<32x256x14x14xf32>
    %v1723 = stablehlo.reduce(%v1722 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1724 = stablehlo.broadcast_in_dim %v1723, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1725 = stablehlo.divide %v1724, %v1717 : tensor<32x256x14x14xf32>
    %v1726 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<32x256x14x14xf32>
    %v1728 = stablehlo.rsqrt %v1727 : tensor<32x256x14x14xf32>
    %v1729 = stablehlo.multiply %v1721, %v1728 : tensor<32x256x14x14xf32>
    %v1730 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1731 = stablehlo.multiply %v1730, %v1729 : tensor<32x256x14x14xf32>
    %v1732 = stablehlo.reduce(%v1731 init: %v1716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1733 = stablehlo.reshape %v1669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1735 = stablehlo.reduce(%v1733 init: %v1734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1736 = stablehlo.reshape %v855 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1737 = stablehlo.reshape %v1658 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1738 = stablehlo.transpose %v1736, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1739 = stablehlo.transpose %v1737, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1740 = stablehlo.convolution(%v1738, %v1739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1741 = stablehlo.transpose %v1740, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1742 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1745 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1746 = stablehlo.broadcast_in_dim %v1745, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1747 = stablehlo.divide %v1746, %v1744 : tensor<32x256x14x14xf32>
    %v1748 = stablehlo.subtract %v1742, %v1747 : tensor<32x256x14x14xf32>
    %v1749 = stablehlo.multiply %v1748, %v1748 : tensor<32x256x14x14xf32>
    %v1750 = stablehlo.reduce(%v1749 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1752 = stablehlo.divide %v1751, %v1744 : tensor<32x256x14x14xf32>
    %v1753 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<32x256x14x14xf32>
    %v1755 = stablehlo.rsqrt %v1754 : tensor<32x256x14x14xf32>
    %v1756 = stablehlo.multiply %v1748, %v1755 : tensor<32x256x14x14xf32>
    %v1757 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1758 = stablehlo.multiply %v1757, %v1756 : tensor<32x256x14x14xf32>
    %v1759 = stablehlo.reduce(%v1758 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1760 = stablehlo.reshape %v1628 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1762 = stablehlo.reduce(%v1760 init: %v1761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1763 = stablehlo.reshape %v1708 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1764 = stablehlo.reshape %v822 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1765 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1766 = stablehlo.compare GT, %v1764, %v1765 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1767 = stablehlo.select %v1766, %v1763, %v1765 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1768 = stablehlo.reshape %v1767 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1769 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1771 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1772 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1773 = stablehlo.reduce(%v1769 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1774 = stablehlo.broadcast_in_dim %v1773, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1775 = stablehlo.divide %v1774, %v1771 : tensor<32x256x14x14xf32>
    %v1776 = stablehlo.subtract %v1769, %v1775 : tensor<32x256x14x14xf32>
    %v1777 = stablehlo.multiply %v1776, %v1776 : tensor<32x256x14x14xf32>
    %v1778 = stablehlo.reduce(%v1777 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1779 = stablehlo.broadcast_in_dim %v1778, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1780 = stablehlo.divide %v1779, %v1771 : tensor<32x256x14x14xf32>
    %v1781 = stablehlo.add %v1780, %v1772 : tensor<32x256x14x14xf32>
    %v1782 = stablehlo.rsqrt %v1781 : tensor<32x256x14x14xf32>
    %v1783 = stablehlo.multiply %v1776, %v1782 : tensor<32x256x14x14xf32>
    %v1784 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1785 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1786 = stablehlo.multiply %v1784, %v1785 : tensor<32x256x14x14xf32>
    %v1787 = stablehlo.reduce(%v1786 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1787, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1789 = stablehlo.multiply %v1783, %v1786 : tensor<32x256x14x14xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1792 = stablehlo.multiply %v1786, %v1771 : tensor<32x256x14x14xf32>
    %v1793 = stablehlo.subtract %v1792, %v1788 : tensor<32x256x14x14xf32>
    %v1794 = stablehlo.multiply %v1783, %v1791 : tensor<32x256x14x14xf32>
    %v1795 = stablehlo.subtract %v1793, %v1794 : tensor<32x256x14x14xf32>
    %v1796 = stablehlo.divide %v1782, %v1771 : tensor<32x256x14x14xf32>
    %v1797 = stablehlo.multiply %v1796, %v1795 : tensor<32x256x14x14xf32>
    %v1798 = stablehlo.reshape %v1797 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1800 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1801 = stablehlo.transpose %v1800, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1802 = stablehlo.convolution(%v1799, %v1801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1805 = stablehlo.reshape %v789 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1806 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1807 = stablehlo.compare GT, %v1805, %v1806 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1808 = stablehlo.select %v1807, %v1804, %v1806 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1810 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1812 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1813 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1814 = stablehlo.reduce(%v1810 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1815 = stablehlo.broadcast_in_dim %v1814, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1816 = stablehlo.divide %v1815, %v1812 : tensor<32x256x14x14xf32>
    %v1817 = stablehlo.subtract %v1810, %v1816 : tensor<32x256x14x14xf32>
    %v1818 = stablehlo.multiply %v1817, %v1817 : tensor<32x256x14x14xf32>
    %v1819 = stablehlo.reduce(%v1818 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1820 = stablehlo.broadcast_in_dim %v1819, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1821 = stablehlo.divide %v1820, %v1812 : tensor<32x256x14x14xf32>
    %v1822 = stablehlo.add %v1821, %v1813 : tensor<32x256x14x14xf32>
    %v1823 = stablehlo.rsqrt %v1822 : tensor<32x256x14x14xf32>
    %v1824 = stablehlo.multiply %v1817, %v1823 : tensor<32x256x14x14xf32>
    %v1825 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1826 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1825, %v1826 : tensor<32x256x14x14xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1830 = stablehlo.multiply %v1824, %v1827 : tensor<32x256x14x14xf32>
    %v1831 = stablehlo.reduce(%v1830 init: %v1811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1832 = stablehlo.broadcast_in_dim %v1831, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1833 = stablehlo.multiply %v1827, %v1812 : tensor<32x256x14x14xf32>
    %v1834 = stablehlo.subtract %v1833, %v1829 : tensor<32x256x14x14xf32>
    %v1835 = stablehlo.multiply %v1824, %v1832 : tensor<32x256x14x14xf32>
    %v1836 = stablehlo.subtract %v1834, %v1835 : tensor<32x256x14x14xf32>
    %v1837 = stablehlo.divide %v1823, %v1812 : tensor<32x256x14x14xf32>
    %v1838 = stablehlo.multiply %v1837, %v1836 : tensor<32x256x14x14xf32>
    %v1839 = stablehlo.reshape %v1838 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1840 = stablehlo.reshape %v1839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1841 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1842 = stablehlo.transpose %v1841, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1843 = stablehlo.convolution(%v1840, %v1842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1844 = stablehlo.reshape %v1843 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1845 = stablehlo.reshape %v1844 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1846 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1847 = stablehlo.add %v1845, %v1846 : tensor<32x256x14x14xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1849 = stablehlo.reshape %v764 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1850 = stablehlo.reshape %v1839 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1851 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1852 = stablehlo.transpose %v1850, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1853 = stablehlo.convolution(%v1851, %v1852)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1854 = stablehlo.transpose %v1853, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1855 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1857 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1858 = stablehlo.reduce(%v1855 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1859 = stablehlo.broadcast_in_dim %v1858, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1860 = stablehlo.divide %v1859, %v1857 : tensor<32x256x14x14xf32>
    %v1861 = stablehlo.subtract %v1855, %v1860 : tensor<32x256x14x14xf32>
    %v1862 = stablehlo.multiply %v1861, %v1861 : tensor<32x256x14x14xf32>
    %v1863 = stablehlo.reduce(%v1862 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1864 = stablehlo.broadcast_in_dim %v1863, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1865 = stablehlo.divide %v1864, %v1857 : tensor<32x256x14x14xf32>
    %v1866 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1867 = stablehlo.add %v1865, %v1866 : tensor<32x256x14x14xf32>
    %v1868 = stablehlo.rsqrt %v1867 : tensor<32x256x14x14xf32>
    %v1869 = stablehlo.multiply %v1861, %v1868 : tensor<32x256x14x14xf32>
    %v1870 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1871 = stablehlo.multiply %v1870, %v1869 : tensor<32x256x14x14xf32>
    %v1872 = stablehlo.reduce(%v1871 init: %v1856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1873 = stablehlo.reshape %v1809 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1875 = stablehlo.reduce(%v1873 init: %v1874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1876 = stablehlo.reshape %v793 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1877 = stablehlo.reshape %v1798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1878 = stablehlo.transpose %v1876, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1879 = stablehlo.transpose %v1877, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1880 = stablehlo.convolution(%v1878, %v1879)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1881 = stablehlo.transpose %v1880, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1882 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1884 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1885 = stablehlo.reduce(%v1882 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1886 = stablehlo.broadcast_in_dim %v1885, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1887 = stablehlo.divide %v1886, %v1884 : tensor<32x256x14x14xf32>
    %v1888 = stablehlo.subtract %v1882, %v1887 : tensor<32x256x14x14xf32>
    %v1889 = stablehlo.multiply %v1888, %v1888 : tensor<32x256x14x14xf32>
    %v1890 = stablehlo.reduce(%v1889 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1891 = stablehlo.broadcast_in_dim %v1890, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1892 = stablehlo.divide %v1891, %v1884 : tensor<32x256x14x14xf32>
    %v1893 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1894 = stablehlo.add %v1892, %v1893 : tensor<32x256x14x14xf32>
    %v1895 = stablehlo.rsqrt %v1894 : tensor<32x256x14x14xf32>
    %v1896 = stablehlo.multiply %v1888, %v1895 : tensor<32x256x14x14xf32>
    %v1897 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1898 = stablehlo.multiply %v1897, %v1896 : tensor<32x256x14x14xf32>
    %v1899 = stablehlo.reduce(%v1898 init: %v1883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1900 = stablehlo.reshape %v1768 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1902 = stablehlo.reduce(%v1900 init: %v1901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1903 = stablehlo.reshape %v1848 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1904 = stablehlo.reshape %v760 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1905 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1906 = stablehlo.compare GT, %v1904, %v1905 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1907 = stablehlo.select %v1906, %v1903, %v1905 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1908 = stablehlo.reshape %v1907 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1909 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1911 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1912 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1913 = stablehlo.reduce(%v1909 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1914 = stablehlo.broadcast_in_dim %v1913, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1915 = stablehlo.divide %v1914, %v1911 : tensor<32x256x14x14xf32>
    %v1916 = stablehlo.subtract %v1909, %v1915 : tensor<32x256x14x14xf32>
    %v1917 = stablehlo.multiply %v1916, %v1916 : tensor<32x256x14x14xf32>
    %v1918 = stablehlo.reduce(%v1917 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1919 = stablehlo.broadcast_in_dim %v1918, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1920 = stablehlo.divide %v1919, %v1911 : tensor<32x256x14x14xf32>
    %v1921 = stablehlo.add %v1920, %v1912 : tensor<32x256x14x14xf32>
    %v1922 = stablehlo.rsqrt %v1921 : tensor<32x256x14x14xf32>
    %v1923 = stablehlo.multiply %v1916, %v1922 : tensor<32x256x14x14xf32>
    %v1924 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1925 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<32x256x14x14xf32>
    %v1927 = stablehlo.reduce(%v1926 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1928 = stablehlo.broadcast_in_dim %v1927, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1929 = stablehlo.multiply %v1923, %v1926 : tensor<32x256x14x14xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1932 = stablehlo.multiply %v1926, %v1911 : tensor<32x256x14x14xf32>
    %v1933 = stablehlo.subtract %v1932, %v1928 : tensor<32x256x14x14xf32>
    %v1934 = stablehlo.multiply %v1923, %v1931 : tensor<32x256x14x14xf32>
    %v1935 = stablehlo.subtract %v1933, %v1934 : tensor<32x256x14x14xf32>
    %v1936 = stablehlo.divide %v1922, %v1911 : tensor<32x256x14x14xf32>
    %v1937 = stablehlo.multiply %v1936, %v1935 : tensor<32x256x14x14xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1940 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1941 = stablehlo.transpose %v1940, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1942 = stablehlo.convolution(%v1939, %v1941)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1943 = stablehlo.reshape %v1942 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1945 = stablehlo.reshape %v727 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1946 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v1947 = stablehlo.compare GT, %v1945, %v1946 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v1948 = stablehlo.select %v1947, %v1944, %v1946 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1950 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1952 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1953 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v1954 = stablehlo.reduce(%v1950 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1955 = stablehlo.broadcast_in_dim %v1954, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1956 = stablehlo.divide %v1955, %v1952 : tensor<32x256x14x14xf32>
    %v1957 = stablehlo.subtract %v1950, %v1956 : tensor<32x256x14x14xf32>
    %v1958 = stablehlo.multiply %v1957, %v1957 : tensor<32x256x14x14xf32>
    %v1959 = stablehlo.reduce(%v1958 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1961 = stablehlo.divide %v1960, %v1952 : tensor<32x256x14x14xf32>
    %v1962 = stablehlo.add %v1961, %v1953 : tensor<32x256x14x14xf32>
    %v1963 = stablehlo.rsqrt %v1962 : tensor<32x256x14x14xf32>
    %v1964 = stablehlo.multiply %v1957, %v1963 : tensor<32x256x14x14xf32>
    %v1965 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1966 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1967 = stablehlo.multiply %v1965, %v1966 : tensor<32x256x14x14xf32>
    %v1968 = stablehlo.reduce(%v1967 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1969 = stablehlo.broadcast_in_dim %v1968, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1970 = stablehlo.multiply %v1964, %v1967 : tensor<32x256x14x14xf32>
    %v1971 = stablehlo.reduce(%v1970 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v1973 = stablehlo.multiply %v1967, %v1952 : tensor<32x256x14x14xf32>
    %v1974 = stablehlo.subtract %v1973, %v1969 : tensor<32x256x14x14xf32>
    %v1975 = stablehlo.multiply %v1964, %v1972 : tensor<32x256x14x14xf32>
    %v1976 = stablehlo.subtract %v1974, %v1975 : tensor<32x256x14x14xf32>
    %v1977 = stablehlo.divide %v1963, %v1952 : tensor<32x256x14x14xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x256x14x14xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1980 = stablehlo.reshape %v1979 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1981 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1982 = stablehlo.transpose %v1981, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1983 = stablehlo.convolution(%v1980, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1986 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1987 = stablehlo.add %v1985, %v1986 : tensor<32x256x14x14xf32>
    %v1988 = stablehlo.reshape %v1987 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v1989 = stablehlo.reshape %v702 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1990 = stablehlo.reshape %v1979 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1991 = stablehlo.transpose %v1989, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1992 = stablehlo.transpose %v1990, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v1993 = stablehlo.convolution(%v1991, %v1992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1994 = stablehlo.transpose %v1993, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1995 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v1996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1997 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v1998 = stablehlo.reduce(%v1995 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1998, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2000 = stablehlo.divide %v1999, %v1997 : tensor<32x256x14x14xf32>
    %v2001 = stablehlo.subtract %v1995, %v2000 : tensor<32x256x14x14xf32>
    %v2002 = stablehlo.multiply %v2001, %v2001 : tensor<32x256x14x14xf32>
    %v2003 = stablehlo.reduce(%v2002 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2004 = stablehlo.broadcast_in_dim %v2003, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2005 = stablehlo.divide %v2004, %v1997 : tensor<32x256x14x14xf32>
    %v2006 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2007 = stablehlo.add %v2005, %v2006 : tensor<32x256x14x14xf32>
    %v2008 = stablehlo.rsqrt %v2007 : tensor<32x256x14x14xf32>
    %v2009 = stablehlo.multiply %v2001, %v2008 : tensor<32x256x14x14xf32>
    %v2010 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2011 = stablehlo.multiply %v2010, %v2009 : tensor<32x256x14x14xf32>
    %v2012 = stablehlo.reduce(%v2011 init: %v1996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2013 = stablehlo.reshape %v1949 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.reduce(%v2013 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2016 = stablehlo.reshape %v731 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2017 = stablehlo.reshape %v1938 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2018 = stablehlo.transpose %v2016, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2019 = stablehlo.transpose %v2017, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2020 = stablehlo.convolution(%v2018, %v2019)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2021 = stablehlo.transpose %v2020, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2022 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2024 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2025 = stablehlo.reduce(%v2022 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2026 = stablehlo.broadcast_in_dim %v2025, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2027 = stablehlo.divide %v2026, %v2024 : tensor<32x256x14x14xf32>
    %v2028 = stablehlo.subtract %v2022, %v2027 : tensor<32x256x14x14xf32>
    %v2029 = stablehlo.multiply %v2028, %v2028 : tensor<32x256x14x14xf32>
    %v2030 = stablehlo.reduce(%v2029 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2031 = stablehlo.broadcast_in_dim %v2030, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2032 = stablehlo.divide %v2031, %v2024 : tensor<32x256x14x14xf32>
    %v2033 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2034 = stablehlo.add %v2032, %v2033 : tensor<32x256x14x14xf32>
    %v2035 = stablehlo.rsqrt %v2034 : tensor<32x256x14x14xf32>
    %v2036 = stablehlo.multiply %v2028, %v2035 : tensor<32x256x14x14xf32>
    %v2037 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2038 = stablehlo.multiply %v2037, %v2036 : tensor<32x256x14x14xf32>
    %v2039 = stablehlo.reduce(%v2038 init: %v2023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2040 = stablehlo.reshape %v1908 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.reduce(%v2040 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2043 = stablehlo.reshape %v1988 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2044 = stablehlo.reshape %v698 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2045 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2046 = stablehlo.compare GT, %v2044, %v2045 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2047 = stablehlo.select %v2046, %v2043, %v2045 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2049 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2051 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2052 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2053 = stablehlo.reduce(%v2049 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2054 = stablehlo.broadcast_in_dim %v2053, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2055 = stablehlo.divide %v2054, %v2051 : tensor<32x256x14x14xf32>
    %v2056 = stablehlo.subtract %v2049, %v2055 : tensor<32x256x14x14xf32>
    %v2057 = stablehlo.multiply %v2056, %v2056 : tensor<32x256x14x14xf32>
    %v2058 = stablehlo.reduce(%v2057 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2060 = stablehlo.divide %v2059, %v2051 : tensor<32x256x14x14xf32>
    %v2061 = stablehlo.add %v2060, %v2052 : tensor<32x256x14x14xf32>
    %v2062 = stablehlo.rsqrt %v2061 : tensor<32x256x14x14xf32>
    %v2063 = stablehlo.multiply %v2056, %v2062 : tensor<32x256x14x14xf32>
    %v2064 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2065 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2066 = stablehlo.multiply %v2064, %v2065 : tensor<32x256x14x14xf32>
    %v2067 = stablehlo.reduce(%v2066 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2068 = stablehlo.broadcast_in_dim %v2067, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2069 = stablehlo.multiply %v2063, %v2066 : tensor<32x256x14x14xf32>
    %v2070 = stablehlo.reduce(%v2069 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2066, %v2051 : tensor<32x256x14x14xf32>
    %v2073 = stablehlo.subtract %v2072, %v2068 : tensor<32x256x14x14xf32>
    %v2074 = stablehlo.multiply %v2063, %v2071 : tensor<32x256x14x14xf32>
    %v2075 = stablehlo.subtract %v2073, %v2074 : tensor<32x256x14x14xf32>
    %v2076 = stablehlo.divide %v2062, %v2051 : tensor<32x256x14x14xf32>
    %v2077 = stablehlo.multiply %v2076, %v2075 : tensor<32x256x14x14xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2080 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2081 = stablehlo.transpose %v2080, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2082 = stablehlo.convolution(%v2079, %v2081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2085 = stablehlo.reshape %v665 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2086 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2087 = stablehlo.compare GT, %v2085, %v2086 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2088 = stablehlo.select %v2087, %v2084, %v2086 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2090 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2092 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2093 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2094 = stablehlo.reduce(%v2090 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2095 = stablehlo.broadcast_in_dim %v2094, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2096 = stablehlo.divide %v2095, %v2092 : tensor<32x256x14x14xf32>
    %v2097 = stablehlo.subtract %v2090, %v2096 : tensor<32x256x14x14xf32>
    %v2098 = stablehlo.multiply %v2097, %v2097 : tensor<32x256x14x14xf32>
    %v2099 = stablehlo.reduce(%v2098 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2100 = stablehlo.broadcast_in_dim %v2099, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2101 = stablehlo.divide %v2100, %v2092 : tensor<32x256x14x14xf32>
    %v2102 = stablehlo.add %v2101, %v2093 : tensor<32x256x14x14xf32>
    %v2103 = stablehlo.rsqrt %v2102 : tensor<32x256x14x14xf32>
    %v2104 = stablehlo.multiply %v2097, %v2103 : tensor<32x256x14x14xf32>
    %v2105 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2106 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2107 = stablehlo.multiply %v2105, %v2106 : tensor<32x256x14x14xf32>
    %v2108 = stablehlo.reduce(%v2107 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2109 = stablehlo.broadcast_in_dim %v2108, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2110 = stablehlo.multiply %v2104, %v2107 : tensor<32x256x14x14xf32>
    %v2111 = stablehlo.reduce(%v2110 init: %v2091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2112 = stablehlo.broadcast_in_dim %v2111, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2113 = stablehlo.multiply %v2107, %v2092 : tensor<32x256x14x14xf32>
    %v2114 = stablehlo.subtract %v2113, %v2109 : tensor<32x256x14x14xf32>
    %v2115 = stablehlo.multiply %v2104, %v2112 : tensor<32x256x14x14xf32>
    %v2116 = stablehlo.subtract %v2114, %v2115 : tensor<32x256x14x14xf32>
    %v2117 = stablehlo.divide %v2103, %v2092 : tensor<32x256x14x14xf32>
    %v2118 = stablehlo.multiply %v2117, %v2116 : tensor<32x256x14x14xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2121 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2122 = stablehlo.transpose %v2121, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2123 = stablehlo.convolution(%v2120, %v2122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2126 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2127 = stablehlo.add %v2125, %v2126 : tensor<32x256x14x14xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2129 = stablehlo.reshape %v640 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2130 = stablehlo.reshape %v2119 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2131 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2132 = stablehlo.transpose %v2130, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2133 = stablehlo.convolution(%v2131, %v2132)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2134 = stablehlo.transpose %v2133, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2135 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2137 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2138 = stablehlo.reduce(%v2135 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2139 = stablehlo.broadcast_in_dim %v2138, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2140 = stablehlo.divide %v2139, %v2137 : tensor<32x256x14x14xf32>
    %v2141 = stablehlo.subtract %v2135, %v2140 : tensor<32x256x14x14xf32>
    %v2142 = stablehlo.multiply %v2141, %v2141 : tensor<32x256x14x14xf32>
    %v2143 = stablehlo.reduce(%v2142 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2143, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2145 = stablehlo.divide %v2144, %v2137 : tensor<32x256x14x14xf32>
    %v2146 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2147 = stablehlo.add %v2145, %v2146 : tensor<32x256x14x14xf32>
    %v2148 = stablehlo.rsqrt %v2147 : tensor<32x256x14x14xf32>
    %v2149 = stablehlo.multiply %v2141, %v2148 : tensor<32x256x14x14xf32>
    %v2150 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2151 = stablehlo.multiply %v2150, %v2149 : tensor<32x256x14x14xf32>
    %v2152 = stablehlo.reduce(%v2151 init: %v2136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2153 = stablehlo.reshape %v2089 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2155 = stablehlo.reduce(%v2153 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2156 = stablehlo.reshape %v669 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2157 = stablehlo.reshape %v2078 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2158 = stablehlo.transpose %v2156, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2159 = stablehlo.transpose %v2157, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2160 = stablehlo.convolution(%v2158, %v2159)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2161 = stablehlo.transpose %v2160, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2162 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2164 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2165 = stablehlo.reduce(%v2162 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2166 = stablehlo.broadcast_in_dim %v2165, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2167 = stablehlo.divide %v2166, %v2164 : tensor<32x256x14x14xf32>
    %v2168 = stablehlo.subtract %v2162, %v2167 : tensor<32x256x14x14xf32>
    %v2169 = stablehlo.multiply %v2168, %v2168 : tensor<32x256x14x14xf32>
    %v2170 = stablehlo.reduce(%v2169 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2172 = stablehlo.divide %v2171, %v2164 : tensor<32x256x14x14xf32>
    %v2173 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2174 = stablehlo.add %v2172, %v2173 : tensor<32x256x14x14xf32>
    %v2175 = stablehlo.rsqrt %v2174 : tensor<32x256x14x14xf32>
    %v2176 = stablehlo.multiply %v2168, %v2175 : tensor<32x256x14x14xf32>
    %v2177 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2178 = stablehlo.multiply %v2177, %v2176 : tensor<32x256x14x14xf32>
    %v2179 = stablehlo.reduce(%v2178 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2180 = stablehlo.reshape %v2048 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2182 = stablehlo.reduce(%v2180 init: %v2181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2183 = stablehlo.reshape %v2128 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2184 = stablehlo.reshape %v636 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2185 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2186 = stablehlo.compare GT, %v2184, %v2185 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2187 = stablehlo.select %v2186, %v2183, %v2185 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2189 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2191 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2192 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2193 = stablehlo.reduce(%v2189 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2194 = stablehlo.broadcast_in_dim %v2193, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2195 = stablehlo.divide %v2194, %v2191 : tensor<32x256x14x14xf32>
    %v2196 = stablehlo.subtract %v2189, %v2195 : tensor<32x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2196, %v2196 : tensor<32x256x14x14xf32>
    %v2198 = stablehlo.reduce(%v2197 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2199 = stablehlo.broadcast_in_dim %v2198, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2200 = stablehlo.divide %v2199, %v2191 : tensor<32x256x14x14xf32>
    %v2201 = stablehlo.add %v2200, %v2192 : tensor<32x256x14x14xf32>
    %v2202 = stablehlo.rsqrt %v2201 : tensor<32x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2196, %v2202 : tensor<32x256x14x14xf32>
    %v2204 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2205 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2206 = stablehlo.multiply %v2204, %v2205 : tensor<32x256x14x14xf32>
    %v2207 = stablehlo.reduce(%v2206 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2208 = stablehlo.broadcast_in_dim %v2207, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2209 = stablehlo.multiply %v2203, %v2206 : tensor<32x256x14x14xf32>
    %v2210 = stablehlo.reduce(%v2209 init: %v2190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2211 = stablehlo.broadcast_in_dim %v2210, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2212 = stablehlo.multiply %v2206, %v2191 : tensor<32x256x14x14xf32>
    %v2213 = stablehlo.subtract %v2212, %v2208 : tensor<32x256x14x14xf32>
    %v2214 = stablehlo.multiply %v2203, %v2211 : tensor<32x256x14x14xf32>
    %v2215 = stablehlo.subtract %v2213, %v2214 : tensor<32x256x14x14xf32>
    %v2216 = stablehlo.divide %v2202, %v2191 : tensor<32x256x14x14xf32>
    %v2217 = stablehlo.multiply %v2216, %v2215 : tensor<32x256x14x14xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2219 = stablehlo.reshape %v2218 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2220 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2221 = stablehlo.transpose %v2220, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2222 = stablehlo.convolution(%v2219, %v2221)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2225 = stablehlo.reshape %v603 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2227 = stablehlo.compare GT, %v2225, %v2226 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2228 = stablehlo.select %v2227, %v2224, %v2226 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2229 = stablehlo.reshape %v2228 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2230 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2232 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2233 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2234 = stablehlo.reduce(%v2230 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2236 = stablehlo.divide %v2235, %v2232 : tensor<32x256x14x14xf32>
    %v2237 = stablehlo.subtract %v2230, %v2236 : tensor<32x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2237, %v2237 : tensor<32x256x14x14xf32>
    %v2239 = stablehlo.reduce(%v2238 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2240 = stablehlo.broadcast_in_dim %v2239, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2241 = stablehlo.divide %v2240, %v2232 : tensor<32x256x14x14xf32>
    %v2242 = stablehlo.add %v2241, %v2233 : tensor<32x256x14x14xf32>
    %v2243 = stablehlo.rsqrt %v2242 : tensor<32x256x14x14xf32>
    %v2244 = stablehlo.multiply %v2237, %v2243 : tensor<32x256x14x14xf32>
    %v2245 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2246 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2247 = stablehlo.multiply %v2245, %v2246 : tensor<32x256x14x14xf32>
    %v2248 = stablehlo.reduce(%v2247 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2249 = stablehlo.broadcast_in_dim %v2248, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2250 = stablehlo.multiply %v2244, %v2247 : tensor<32x256x14x14xf32>
    %v2251 = stablehlo.reduce(%v2250 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2252 = stablehlo.broadcast_in_dim %v2251, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2253 = stablehlo.multiply %v2247, %v2232 : tensor<32x256x14x14xf32>
    %v2254 = stablehlo.subtract %v2253, %v2249 : tensor<32x256x14x14xf32>
    %v2255 = stablehlo.multiply %v2244, %v2252 : tensor<32x256x14x14xf32>
    %v2256 = stablehlo.subtract %v2254, %v2255 : tensor<32x256x14x14xf32>
    %v2257 = stablehlo.divide %v2243, %v2232 : tensor<32x256x14x14xf32>
    %v2258 = stablehlo.multiply %v2257, %v2256 : tensor<32x256x14x14xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2260 = stablehlo.reshape %v2259 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2261 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2262 = stablehlo.transpose %v2261, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2263 = stablehlo.convolution(%v2260, %v2262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2266 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2267 = stablehlo.add %v2265, %v2266 : tensor<32x256x14x14xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2269 = stablehlo.reshape %v578 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2270 = stablehlo.reshape %v2259 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2271 = stablehlo.transpose %v2269, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2272 = stablehlo.transpose %v2270, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2273 = stablehlo.convolution(%v2271, %v2272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2274 = stablehlo.transpose %v2273, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2275 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2277 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2278 = stablehlo.reduce(%v2275 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2279 = stablehlo.broadcast_in_dim %v2278, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2280 = stablehlo.divide %v2279, %v2277 : tensor<32x256x14x14xf32>
    %v2281 = stablehlo.subtract %v2275, %v2280 : tensor<32x256x14x14xf32>
    %v2282 = stablehlo.multiply %v2281, %v2281 : tensor<32x256x14x14xf32>
    %v2283 = stablehlo.reduce(%v2282 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2285 = stablehlo.divide %v2284, %v2277 : tensor<32x256x14x14xf32>
    %v2286 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2287 = stablehlo.add %v2285, %v2286 : tensor<32x256x14x14xf32>
    %v2288 = stablehlo.rsqrt %v2287 : tensor<32x256x14x14xf32>
    %v2289 = stablehlo.multiply %v2281, %v2288 : tensor<32x256x14x14xf32>
    %v2290 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2291 = stablehlo.multiply %v2290, %v2289 : tensor<32x256x14x14xf32>
    %v2292 = stablehlo.reduce(%v2291 init: %v2276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2293 = stablehlo.reshape %v2229 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2295 = stablehlo.reduce(%v2293 init: %v2294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2296 = stablehlo.reshape %v607 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2297 = stablehlo.reshape %v2218 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2298 = stablehlo.transpose %v2296, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2299 = stablehlo.transpose %v2297, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2300 = stablehlo.convolution(%v2298, %v2299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2301 = stablehlo.transpose %v2300, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2302 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2304 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2305 = stablehlo.reduce(%v2302 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2306 = stablehlo.broadcast_in_dim %v2305, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2307 = stablehlo.divide %v2306, %v2304 : tensor<32x256x14x14xf32>
    %v2308 = stablehlo.subtract %v2302, %v2307 : tensor<32x256x14x14xf32>
    %v2309 = stablehlo.multiply %v2308, %v2308 : tensor<32x256x14x14xf32>
    %v2310 = stablehlo.reduce(%v2309 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2310, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2312 = stablehlo.divide %v2311, %v2304 : tensor<32x256x14x14xf32>
    %v2313 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2314 = stablehlo.add %v2312, %v2313 : tensor<32x256x14x14xf32>
    %v2315 = stablehlo.rsqrt %v2314 : tensor<32x256x14x14xf32>
    %v2316 = stablehlo.multiply %v2308, %v2315 : tensor<32x256x14x14xf32>
    %v2317 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2318 = stablehlo.multiply %v2317, %v2316 : tensor<32x256x14x14xf32>
    %v2319 = stablehlo.reduce(%v2318 init: %v2303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2320 = stablehlo.reshape %v2188 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2322 = stablehlo.reduce(%v2320 init: %v2321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2323 = stablehlo.reshape %v2268 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2324 = stablehlo.reshape %v574 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2326 = stablehlo.compare GT, %v2324, %v2325 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2327 = stablehlo.select %v2326, %v2323, %v2325 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2329 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2331 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2332 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2333 = stablehlo.reduce(%v2329 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2335 = stablehlo.divide %v2334, %v2331 : tensor<32x256x14x14xf32>
    %v2336 = stablehlo.subtract %v2329, %v2335 : tensor<32x256x14x14xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<32x256x14x14xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2340 = stablehlo.divide %v2339, %v2331 : tensor<32x256x14x14xf32>
    %v2341 = stablehlo.add %v2340, %v2332 : tensor<32x256x14x14xf32>
    %v2342 = stablehlo.rsqrt %v2341 : tensor<32x256x14x14xf32>
    %v2343 = stablehlo.multiply %v2336, %v2342 : tensor<32x256x14x14xf32>
    %v2344 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2345 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<32x256x14x14xf32>
    %v2347 = stablehlo.reduce(%v2346 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2348 = stablehlo.broadcast_in_dim %v2347, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2349 = stablehlo.multiply %v2343, %v2346 : tensor<32x256x14x14xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2351 = stablehlo.broadcast_in_dim %v2350, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2352 = stablehlo.multiply %v2346, %v2331 : tensor<32x256x14x14xf32>
    %v2353 = stablehlo.subtract %v2352, %v2348 : tensor<32x256x14x14xf32>
    %v2354 = stablehlo.multiply %v2343, %v2351 : tensor<32x256x14x14xf32>
    %v2355 = stablehlo.subtract %v2353, %v2354 : tensor<32x256x14x14xf32>
    %v2356 = stablehlo.divide %v2342, %v2331 : tensor<32x256x14x14xf32>
    %v2357 = stablehlo.multiply %v2356, %v2355 : tensor<32x256x14x14xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2360 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2361 = stablehlo.transpose %v2360, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2362 = stablehlo.convolution(%v2359, %v2361)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2364 = stablehlo.reshape %v2363 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2365 = stablehlo.reshape %v516 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2366 = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %v2367 = stablehlo.compare GT, %v2365, %v2366 : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %v2368 = stablehlo.select %v2367, %v2364, %v2366 : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %v2369 = stablehlo.reshape %v2368 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2370 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2372 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2373 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2374 = stablehlo.reduce(%v2370 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2376 = stablehlo.divide %v2375, %v2372 : tensor<32x256x14x14xf32>
    %v2377 = stablehlo.subtract %v2370, %v2376 : tensor<32x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2377, %v2377 : tensor<32x256x14x14xf32>
    %v2379 = stablehlo.reduce(%v2378 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2380 = stablehlo.broadcast_in_dim %v2379, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2381 = stablehlo.divide %v2380, %v2372 : tensor<32x256x14x14xf32>
    %v2382 = stablehlo.add %v2381, %v2373 : tensor<32x256x14x14xf32>
    %v2383 = stablehlo.rsqrt %v2382 : tensor<32x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2377, %v2383 : tensor<32x256x14x14xf32>
    %v2385 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2386 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2387 = stablehlo.multiply %v2385, %v2386 : tensor<32x256x14x14xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2390 = stablehlo.multiply %v2384, %v2387 : tensor<32x256x14x14xf32>
    %v2391 = stablehlo.reduce(%v2390 init: %v2371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2392 = stablehlo.broadcast_in_dim %v2391, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2393 = stablehlo.multiply %v2387, %v2372 : tensor<32x256x14x14xf32>
    %v2394 = stablehlo.subtract %v2393, %v2389 : tensor<32x256x14x14xf32>
    %v2395 = stablehlo.multiply %v2384, %v2392 : tensor<32x256x14x14xf32>
    %v2396 = stablehlo.subtract %v2394, %v2395 : tensor<32x256x14x14xf32>
    %v2397 = stablehlo.divide %v2383, %v2372 : tensor<32x256x14x14xf32>
    %v2398 = stablehlo.multiply %v2397, %v2396 : tensor<32x256x14x14xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2402 = stablehlo.pad %v2400, %v2401, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2403 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2404 = stablehlo.transpose %v2403, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2405 = stablehlo.convolution(%v2402, %v2404)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2407 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2408 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2409 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2410 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2411 = stablehlo.reduce(%v2407 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2412 = stablehlo.broadcast_in_dim %v2411, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2413 = stablehlo.divide %v2412, %v2409 : tensor<32x256x14x14xf32>
    %v2414 = stablehlo.subtract %v2407, %v2413 : tensor<32x256x14x14xf32>
    %v2415 = stablehlo.multiply %v2414, %v2414 : tensor<32x256x14x14xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2418 = stablehlo.divide %v2417, %v2409 : tensor<32x256x14x14xf32>
    %v2419 = stablehlo.add %v2418, %v2410 : tensor<32x256x14x14xf32>
    %v2420 = stablehlo.rsqrt %v2419 : tensor<32x256x14x14xf32>
    %v2421 = stablehlo.multiply %v2414, %v2420 : tensor<32x256x14x14xf32>
    %v2422 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2423 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2422, %v2423 : tensor<32x256x14x14xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2427 = stablehlo.multiply %v2421, %v2424 : tensor<32x256x14x14xf32>
    %v2428 = stablehlo.reduce(%v2427 init: %v2408) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2429 = stablehlo.broadcast_in_dim %v2428, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2430 = stablehlo.multiply %v2424, %v2409 : tensor<32x256x14x14xf32>
    %v2431 = stablehlo.subtract %v2430, %v2426 : tensor<32x256x14x14xf32>
    %v2432 = stablehlo.multiply %v2421, %v2429 : tensor<32x256x14x14xf32>
    %v2433 = stablehlo.subtract %v2431, %v2432 : tensor<32x256x14x14xf32>
    %v2434 = stablehlo.divide %v2420, %v2409 : tensor<32x256x14x14xf32>
    %v2435 = stablehlo.multiply %v2434, %v2433 : tensor<32x256x14x14xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<32x256x14x14xf32>) -> tensor<32x50176xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2439 = stablehlo.pad %v2437, %v2438, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2440 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2441 = stablehlo.transpose %v2440, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2442 = stablehlo.convolution(%v2439, %v2441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x1x1xf32>) -> tensor<32x128x28x28xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2444 = stablehlo.reshape %v2406 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2445 = stablehlo.reshape %v2443 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2446 = stablehlo.add %v2444, %v2445 : tensor<32x128x28x28xf32>
    %v2447 = stablehlo.reshape %v2446 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2448 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2449 = stablehlo.reshape %v2399 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2451 = stablehlo.pad %v2449, %v2450, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2452 = stablehlo.transpose %v2448, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2453 = stablehlo.transpose %v2451, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2454 = stablehlo.convolution(%v2452, %v2453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2455 = stablehlo.transpose %v2454, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2456 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2457 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2458 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2459 = stablehlo.reduce(%v2456 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2461 = stablehlo.divide %v2460, %v2458 : tensor<32x256x14x14xf32>
    %v2462 = stablehlo.subtract %v2456, %v2461 : tensor<32x256x14x14xf32>
    %v2463 = stablehlo.multiply %v2462, %v2462 : tensor<32x256x14x14xf32>
    %v2464 = stablehlo.reduce(%v2463 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2465 = stablehlo.broadcast_in_dim %v2464, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2466 = stablehlo.divide %v2465, %v2458 : tensor<32x256x14x14xf32>
    %v2467 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2468 = stablehlo.add %v2466, %v2467 : tensor<32x256x14x14xf32>
    %v2469 = stablehlo.rsqrt %v2468 : tensor<32x256x14x14xf32>
    %v2470 = stablehlo.multiply %v2462, %v2469 : tensor<32x256x14x14xf32>
    %v2471 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2472 = stablehlo.multiply %v2471, %v2470 : tensor<32x256x14x14xf32>
    %v2473 = stablehlo.reduce(%v2472 init: %v2457) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2474 = stablehlo.reshape %v2369 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2476 = stablehlo.reduce(%v2474 init: %v2475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2477 = stablehlo.reshape %v520 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2478 = stablehlo.reshape %v2358 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2479 = stablehlo.transpose %v2477, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2480 = stablehlo.transpose %v2478, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %v2481 = stablehlo.convolution(%v2479, %v2480)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2482 = stablehlo.transpose %v2481, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2483 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2485 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2483 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2488 = stablehlo.divide %v2487, %v2485 : tensor<32x256x14x14xf32>
    %v2489 = stablehlo.subtract %v2483, %v2488 : tensor<32x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2489, %v2489 : tensor<32x256x14x14xf32>
    %v2491 = stablehlo.reduce(%v2490 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2492 = stablehlo.broadcast_in_dim %v2491, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2493 = stablehlo.divide %v2492, %v2485 : tensor<32x256x14x14xf32>
    %v2494 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2495 = stablehlo.add %v2493, %v2494 : tensor<32x256x14x14xf32>
    %v2496 = stablehlo.rsqrt %v2495 : tensor<32x256x14x14xf32>
    %v2497 = stablehlo.multiply %v2489, %v2496 : tensor<32x256x14x14xf32>
    %v2498 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2499 = stablehlo.multiply %v2498, %v2497 : tensor<32x256x14x14xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2501 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2503 = stablehlo.reduce(%v2501 init: %v2502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.reshape %v491 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2505 = stablehlo.reshape %v2436 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2507 = stablehlo.pad %v2505, %v2506, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %v2508 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2509 = stablehlo.transpose %v2507, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %v2510 = stablehlo.convolution(%v2508, %v2509)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x1x1xf32>
    %v2511 = stablehlo.transpose %v2510, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2512 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2514 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v2515 = stablehlo.reduce(%v2512 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2516 = stablehlo.broadcast_in_dim %v2515, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2517 = stablehlo.divide %v2516, %v2514 : tensor<32x256x14x14xf32>
    %v2518 = stablehlo.subtract %v2512, %v2517 : tensor<32x256x14x14xf32>
    %v2519 = stablehlo.multiply %v2518, %v2518 : tensor<32x256x14x14xf32>
    %v2520 = stablehlo.reduce(%v2519 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2521 = stablehlo.broadcast_in_dim %v2520, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v2522 = stablehlo.divide %v2521, %v2514 : tensor<32x256x14x14xf32>
    %v2523 = stablehlo.constant dense<1.0e-05> : tensor<32x256x14x14xf32>
    %v2524 = stablehlo.add %v2522, %v2523 : tensor<32x256x14x14xf32>
    %v2525 = stablehlo.rsqrt %v2524 : tensor<32x256x14x14xf32>
    %v2526 = stablehlo.multiply %v2518, %v2525 : tensor<32x256x14x14xf32>
    %v2527 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2528 = stablehlo.multiply %v2527, %v2526 : tensor<32x256x14x14xf32>
    %v2529 = stablehlo.reduce(%v2528 init: %v2513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2530 = stablehlo.reshape %v2328 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.reduce(%v2530 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2533 = stablehlo.reshape %v2447 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2534 = stablehlo.reshape %v487 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2536 = stablehlo.compare GT, %v2534, %v2535 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2537 = stablehlo.select %v2536, %v2533, %v2535 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2538 = stablehlo.reshape %v2537 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2539 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2541 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2542 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2543 = stablehlo.reduce(%v2539 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2544 = stablehlo.broadcast_in_dim %v2543, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2545 = stablehlo.divide %v2544, %v2541 : tensor<32x128x28x28xf32>
    %v2546 = stablehlo.subtract %v2539, %v2545 : tensor<32x128x28x28xf32>
    %v2547 = stablehlo.multiply %v2546, %v2546 : tensor<32x128x28x28xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2550 = stablehlo.divide %v2549, %v2541 : tensor<32x128x28x28xf32>
    %v2551 = stablehlo.add %v2550, %v2542 : tensor<32x128x28x28xf32>
    %v2552 = stablehlo.rsqrt %v2551 : tensor<32x128x28x28xf32>
    %v2553 = stablehlo.multiply %v2546, %v2552 : tensor<32x128x28x28xf32>
    %v2554 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2555 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2556 = stablehlo.multiply %v2554, %v2555 : tensor<32x128x28x28xf32>
    %v2557 = stablehlo.reduce(%v2556 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2558 = stablehlo.broadcast_in_dim %v2557, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2559 = stablehlo.multiply %v2553, %v2556 : tensor<32x128x28x28xf32>
    %v2560 = stablehlo.reduce(%v2559 init: %v2540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2562 = stablehlo.multiply %v2556, %v2541 : tensor<32x128x28x28xf32>
    %v2563 = stablehlo.subtract %v2562, %v2558 : tensor<32x128x28x28xf32>
    %v2564 = stablehlo.multiply %v2553, %v2561 : tensor<32x128x28x28xf32>
    %v2565 = stablehlo.subtract %v2563, %v2564 : tensor<32x128x28x28xf32>
    %v2566 = stablehlo.divide %v2552, %v2541 : tensor<32x128x28x28xf32>
    %v2567 = stablehlo.multiply %v2566, %v2565 : tensor<32x128x28x28xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2569 = stablehlo.reshape %v2568 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2570 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2571 = stablehlo.transpose %v2570, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2572 = stablehlo.convolution(%v2569, %v2571)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2573 = stablehlo.reshape %v2572 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2574 = stablehlo.reshape %v2573 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2575 = stablehlo.reshape %v454 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2576 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2577 = stablehlo.compare GT, %v2575, %v2576 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2578 = stablehlo.select %v2577, %v2574, %v2576 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2580 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2582 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2583 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2584 = stablehlo.reduce(%v2580 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2585 = stablehlo.broadcast_in_dim %v2584, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2586 = stablehlo.divide %v2585, %v2582 : tensor<32x128x28x28xf32>
    %v2587 = stablehlo.subtract %v2580, %v2586 : tensor<32x128x28x28xf32>
    %v2588 = stablehlo.multiply %v2587, %v2587 : tensor<32x128x28x28xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2591 = stablehlo.divide %v2590, %v2582 : tensor<32x128x28x28xf32>
    %v2592 = stablehlo.add %v2591, %v2583 : tensor<32x128x28x28xf32>
    %v2593 = stablehlo.rsqrt %v2592 : tensor<32x128x28x28xf32>
    %v2594 = stablehlo.multiply %v2587, %v2593 : tensor<32x128x28x28xf32>
    %v2595 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2596 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2597 = stablehlo.multiply %v2595, %v2596 : tensor<32x128x28x28xf32>
    %v2598 = stablehlo.reduce(%v2597 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2599 = stablehlo.broadcast_in_dim %v2598, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2600 = stablehlo.multiply %v2594, %v2597 : tensor<32x128x28x28xf32>
    %v2601 = stablehlo.reduce(%v2600 init: %v2581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2602 = stablehlo.broadcast_in_dim %v2601, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2603 = stablehlo.multiply %v2597, %v2582 : tensor<32x128x28x28xf32>
    %v2604 = stablehlo.subtract %v2603, %v2599 : tensor<32x128x28x28xf32>
    %v2605 = stablehlo.multiply %v2594, %v2602 : tensor<32x128x28x28xf32>
    %v2606 = stablehlo.subtract %v2604, %v2605 : tensor<32x128x28x28xf32>
    %v2607 = stablehlo.divide %v2593, %v2582 : tensor<32x128x28x28xf32>
    %v2608 = stablehlo.multiply %v2607, %v2606 : tensor<32x128x28x28xf32>
    %v2609 = stablehlo.reshape %v2608 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2611 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2612 = stablehlo.transpose %v2611, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2613 = stablehlo.convolution(%v2610, %v2612)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2616 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2617 = stablehlo.add %v2615, %v2616 : tensor<32x128x28x28xf32>
    %v2618 = stablehlo.reshape %v2617 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2619 = stablehlo.reshape %v429 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2620 = stablehlo.reshape %v2609 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2621 = stablehlo.transpose %v2619, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2622 = stablehlo.transpose %v2620, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2623 = stablehlo.convolution(%v2621, %v2622)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2624 = stablehlo.transpose %v2623, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2625 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2627 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2628 = stablehlo.reduce(%v2625 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2630 = stablehlo.divide %v2629, %v2627 : tensor<32x128x28x28xf32>
    %v2631 = stablehlo.subtract %v2625, %v2630 : tensor<32x128x28x28xf32>
    %v2632 = stablehlo.multiply %v2631, %v2631 : tensor<32x128x28x28xf32>
    %v2633 = stablehlo.reduce(%v2632 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2635 = stablehlo.divide %v2634, %v2627 : tensor<32x128x28x28xf32>
    %v2636 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2637 = stablehlo.add %v2635, %v2636 : tensor<32x128x28x28xf32>
    %v2638 = stablehlo.rsqrt %v2637 : tensor<32x128x28x28xf32>
    %v2639 = stablehlo.multiply %v2631, %v2638 : tensor<32x128x28x28xf32>
    %v2640 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2641 = stablehlo.multiply %v2640, %v2639 : tensor<32x128x28x28xf32>
    %v2642 = stablehlo.reduce(%v2641 init: %v2626) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2643 = stablehlo.reshape %v2579 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2645 = stablehlo.reduce(%v2643 init: %v2644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2646 = stablehlo.reshape %v458 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2647 = stablehlo.reshape %v2568 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2648 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2649 = stablehlo.transpose %v2647, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2650 = stablehlo.convolution(%v2648, %v2649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2651 = stablehlo.transpose %v2650, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2652 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2654 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2655 = stablehlo.reduce(%v2652 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2657 = stablehlo.divide %v2656, %v2654 : tensor<32x128x28x28xf32>
    %v2658 = stablehlo.subtract %v2652, %v2657 : tensor<32x128x28x28xf32>
    %v2659 = stablehlo.multiply %v2658, %v2658 : tensor<32x128x28x28xf32>
    %v2660 = stablehlo.reduce(%v2659 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2660, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2662 = stablehlo.divide %v2661, %v2654 : tensor<32x128x28x28xf32>
    %v2663 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2664 = stablehlo.add %v2662, %v2663 : tensor<32x128x28x28xf32>
    %v2665 = stablehlo.rsqrt %v2664 : tensor<32x128x28x28xf32>
    %v2666 = stablehlo.multiply %v2658, %v2665 : tensor<32x128x28x28xf32>
    %v2667 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2668 = stablehlo.multiply %v2667, %v2666 : tensor<32x128x28x28xf32>
    %v2669 = stablehlo.reduce(%v2668 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2670 = stablehlo.reshape %v2538 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2672 = stablehlo.reduce(%v2670 init: %v2671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2673 = stablehlo.reshape %v2618 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2674 = stablehlo.reshape %v425 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2675 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2676 = stablehlo.compare GT, %v2674, %v2675 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2677 = stablehlo.select %v2676, %v2673, %v2675 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2678 = stablehlo.reshape %v2677 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2679 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2681 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2682 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2683 = stablehlo.reduce(%v2679 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2685 = stablehlo.divide %v2684, %v2681 : tensor<32x128x28x28xf32>
    %v2686 = stablehlo.subtract %v2679, %v2685 : tensor<32x128x28x28xf32>
    %v2687 = stablehlo.multiply %v2686, %v2686 : tensor<32x128x28x28xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2690 = stablehlo.divide %v2689, %v2681 : tensor<32x128x28x28xf32>
    %v2691 = stablehlo.add %v2690, %v2682 : tensor<32x128x28x28xf32>
    %v2692 = stablehlo.rsqrt %v2691 : tensor<32x128x28x28xf32>
    %v2693 = stablehlo.multiply %v2686, %v2692 : tensor<32x128x28x28xf32>
    %v2694 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2695 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2696 = stablehlo.multiply %v2694, %v2695 : tensor<32x128x28x28xf32>
    %v2697 = stablehlo.reduce(%v2696 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2698 = stablehlo.broadcast_in_dim %v2697, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2699 = stablehlo.multiply %v2693, %v2696 : tensor<32x128x28x28xf32>
    %v2700 = stablehlo.reduce(%v2699 init: %v2680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2701 = stablehlo.broadcast_in_dim %v2700, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2702 = stablehlo.multiply %v2696, %v2681 : tensor<32x128x28x28xf32>
    %v2703 = stablehlo.subtract %v2702, %v2698 : tensor<32x128x28x28xf32>
    %v2704 = stablehlo.multiply %v2693, %v2701 : tensor<32x128x28x28xf32>
    %v2705 = stablehlo.subtract %v2703, %v2704 : tensor<32x128x28x28xf32>
    %v2706 = stablehlo.divide %v2692, %v2681 : tensor<32x128x28x28xf32>
    %v2707 = stablehlo.multiply %v2706, %v2705 : tensor<32x128x28x28xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2710 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2711 = stablehlo.transpose %v2710, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2712 = stablehlo.convolution(%v2709, %v2711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2713 = stablehlo.reshape %v2712 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2714 = stablehlo.reshape %v2713 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2715 = stablehlo.reshape %v392 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2716 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2717 = stablehlo.compare GT, %v2715, %v2716 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2718 = stablehlo.select %v2717, %v2714, %v2716 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2719 = stablehlo.reshape %v2718 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2720 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2722 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2723 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2724 = stablehlo.reduce(%v2720 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2725 = stablehlo.broadcast_in_dim %v2724, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2726 = stablehlo.divide %v2725, %v2722 : tensor<32x128x28x28xf32>
    %v2727 = stablehlo.subtract %v2720, %v2726 : tensor<32x128x28x28xf32>
    %v2728 = stablehlo.multiply %v2727, %v2727 : tensor<32x128x28x28xf32>
    %v2729 = stablehlo.reduce(%v2728 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2731 = stablehlo.divide %v2730, %v2722 : tensor<32x128x28x28xf32>
    %v2732 = stablehlo.add %v2731, %v2723 : tensor<32x128x28x28xf32>
    %v2733 = stablehlo.rsqrt %v2732 : tensor<32x128x28x28xf32>
    %v2734 = stablehlo.multiply %v2727, %v2733 : tensor<32x128x28x28xf32>
    %v2735 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2736 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2737 = stablehlo.multiply %v2735, %v2736 : tensor<32x128x28x28xf32>
    %v2738 = stablehlo.reduce(%v2737 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2740 = stablehlo.multiply %v2734, %v2737 : tensor<32x128x28x28xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2742 = stablehlo.broadcast_in_dim %v2741, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2743 = stablehlo.multiply %v2737, %v2722 : tensor<32x128x28x28xf32>
    %v2744 = stablehlo.subtract %v2743, %v2739 : tensor<32x128x28x28xf32>
    %v2745 = stablehlo.multiply %v2734, %v2742 : tensor<32x128x28x28xf32>
    %v2746 = stablehlo.subtract %v2744, %v2745 : tensor<32x128x28x28xf32>
    %v2747 = stablehlo.divide %v2733, %v2722 : tensor<32x128x28x28xf32>
    %v2748 = stablehlo.multiply %v2747, %v2746 : tensor<32x128x28x28xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2751 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2752 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2753 = stablehlo.convolution(%v2750, %v2752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2756 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2757 = stablehlo.add %v2755, %v2756 : tensor<32x128x28x28xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2759 = stablehlo.reshape %v367 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2760 = stablehlo.reshape %v2749 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2761 = stablehlo.transpose %v2759, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2762 = stablehlo.transpose %v2760, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2763 = stablehlo.convolution(%v2761, %v2762)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2764 = stablehlo.transpose %v2763, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2765 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2767 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2768 = stablehlo.reduce(%v2765 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2768, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2770 = stablehlo.divide %v2769, %v2767 : tensor<32x128x28x28xf32>
    %v2771 = stablehlo.subtract %v2765, %v2770 : tensor<32x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2771, %v2771 : tensor<32x128x28x28xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2775 = stablehlo.divide %v2774, %v2767 : tensor<32x128x28x28xf32>
    %v2776 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2777 = stablehlo.add %v2775, %v2776 : tensor<32x128x28x28xf32>
    %v2778 = stablehlo.rsqrt %v2777 : tensor<32x128x28x28xf32>
    %v2779 = stablehlo.multiply %v2771, %v2778 : tensor<32x128x28x28xf32>
    %v2780 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2781 = stablehlo.multiply %v2780, %v2779 : tensor<32x128x28x28xf32>
    %v2782 = stablehlo.reduce(%v2781 init: %v2766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2783 = stablehlo.reshape %v2719 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2785 = stablehlo.reduce(%v2783 init: %v2784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2786 = stablehlo.reshape %v396 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2787 = stablehlo.reshape %v2708 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2788 = stablehlo.transpose %v2786, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2789 = stablehlo.transpose %v2787, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2790 = stablehlo.convolution(%v2788, %v2789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2791 = stablehlo.transpose %v2790, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2792 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2794 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2795 = stablehlo.reduce(%v2792 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2796 = stablehlo.broadcast_in_dim %v2795, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2797 = stablehlo.divide %v2796, %v2794 : tensor<32x128x28x28xf32>
    %v2798 = stablehlo.subtract %v2792, %v2797 : tensor<32x128x28x28xf32>
    %v2799 = stablehlo.multiply %v2798, %v2798 : tensor<32x128x28x28xf32>
    %v2800 = stablehlo.reduce(%v2799 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2800, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2802 = stablehlo.divide %v2801, %v2794 : tensor<32x128x28x28xf32>
    %v2803 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2804 = stablehlo.add %v2802, %v2803 : tensor<32x128x28x28xf32>
    %v2805 = stablehlo.rsqrt %v2804 : tensor<32x128x28x28xf32>
    %v2806 = stablehlo.multiply %v2798, %v2805 : tensor<32x128x28x28xf32>
    %v2807 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2808 = stablehlo.multiply %v2807, %v2806 : tensor<32x128x28x28xf32>
    %v2809 = stablehlo.reduce(%v2808 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2810 = stablehlo.reshape %v2678 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2811 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2812 = stablehlo.reduce(%v2810 init: %v2811) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2813 = stablehlo.reshape %v2758 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2814 = stablehlo.reshape %v363 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2815 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2816 = stablehlo.compare GT, %v2814, %v2815 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2817 = stablehlo.select %v2816, %v2813, %v2815 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2818 = stablehlo.reshape %v2817 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2819 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2821 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2822 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2823 = stablehlo.reduce(%v2819 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2823, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2825 = stablehlo.divide %v2824, %v2821 : tensor<32x128x28x28xf32>
    %v2826 = stablehlo.subtract %v2819, %v2825 : tensor<32x128x28x28xf32>
    %v2827 = stablehlo.multiply %v2826, %v2826 : tensor<32x128x28x28xf32>
    %v2828 = stablehlo.reduce(%v2827 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2829 = stablehlo.broadcast_in_dim %v2828, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2830 = stablehlo.divide %v2829, %v2821 : tensor<32x128x28x28xf32>
    %v2831 = stablehlo.add %v2830, %v2822 : tensor<32x128x28x28xf32>
    %v2832 = stablehlo.rsqrt %v2831 : tensor<32x128x28x28xf32>
    %v2833 = stablehlo.multiply %v2826, %v2832 : tensor<32x128x28x28xf32>
    %v2834 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2835 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2836 = stablehlo.multiply %v2834, %v2835 : tensor<32x128x28x28xf32>
    %v2837 = stablehlo.reduce(%v2836 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2838 = stablehlo.broadcast_in_dim %v2837, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2839 = stablehlo.multiply %v2833, %v2836 : tensor<32x128x28x28xf32>
    %v2840 = stablehlo.reduce(%v2839 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2841 = stablehlo.broadcast_in_dim %v2840, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2842 = stablehlo.multiply %v2836, %v2821 : tensor<32x128x28x28xf32>
    %v2843 = stablehlo.subtract %v2842, %v2838 : tensor<32x128x28x28xf32>
    %v2844 = stablehlo.multiply %v2833, %v2841 : tensor<32x128x28x28xf32>
    %v2845 = stablehlo.subtract %v2843, %v2844 : tensor<32x128x28x28xf32>
    %v2846 = stablehlo.divide %v2832, %v2821 : tensor<32x128x28x28xf32>
    %v2847 = stablehlo.multiply %v2846, %v2845 : tensor<32x128x28x28xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2850 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2851 = stablehlo.transpose %v2850, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2852 = stablehlo.convolution(%v2849, %v2851)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2853 = stablehlo.reshape %v2852 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2855 = stablehlo.reshape %v330 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2857 = stablehlo.compare GT, %v2855, %v2856 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2858 = stablehlo.select %v2857, %v2854, %v2856 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2860 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2862 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2863 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2864 = stablehlo.reduce(%v2860 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2866 = stablehlo.divide %v2865, %v2862 : tensor<32x128x28x28xf32>
    %v2867 = stablehlo.subtract %v2860, %v2866 : tensor<32x128x28x28xf32>
    %v2868 = stablehlo.multiply %v2867, %v2867 : tensor<32x128x28x28xf32>
    %v2869 = stablehlo.reduce(%v2868 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2870 = stablehlo.broadcast_in_dim %v2869, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2871 = stablehlo.divide %v2870, %v2862 : tensor<32x128x28x28xf32>
    %v2872 = stablehlo.add %v2871, %v2863 : tensor<32x128x28x28xf32>
    %v2873 = stablehlo.rsqrt %v2872 : tensor<32x128x28x28xf32>
    %v2874 = stablehlo.multiply %v2867, %v2873 : tensor<32x128x28x28xf32>
    %v2875 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2876 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2877 = stablehlo.multiply %v2875, %v2876 : tensor<32x128x28x28xf32>
    %v2878 = stablehlo.reduce(%v2877 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2880 = stablehlo.multiply %v2874, %v2877 : tensor<32x128x28x28xf32>
    %v2881 = stablehlo.reduce(%v2880 init: %v2861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2882 = stablehlo.broadcast_in_dim %v2881, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2883 = stablehlo.multiply %v2877, %v2862 : tensor<32x128x28x28xf32>
    %v2884 = stablehlo.subtract %v2883, %v2879 : tensor<32x128x28x28xf32>
    %v2885 = stablehlo.multiply %v2874, %v2882 : tensor<32x128x28x28xf32>
    %v2886 = stablehlo.subtract %v2884, %v2885 : tensor<32x128x28x28xf32>
    %v2887 = stablehlo.divide %v2873, %v2862 : tensor<32x128x28x28xf32>
    %v2888 = stablehlo.multiply %v2887, %v2886 : tensor<32x128x28x28xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2890 = stablehlo.reshape %v2889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2891 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2892 = stablehlo.transpose %v2891, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2893 = stablehlo.convolution(%v2890, %v2892)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2894 = stablehlo.reshape %v2893 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2896 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2897 = stablehlo.add %v2895, %v2896 : tensor<32x128x28x28xf32>
    %v2898 = stablehlo.reshape %v2897 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2899 = stablehlo.reshape %v305 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2900 = stablehlo.reshape %v2889 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2901 = stablehlo.transpose %v2899, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2902 = stablehlo.transpose %v2900, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2903 = stablehlo.convolution(%v2901, %v2902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2904 = stablehlo.transpose %v2903, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2905 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2907 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2908 = stablehlo.reduce(%v2905 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2910 = stablehlo.divide %v2909, %v2907 : tensor<32x128x28x28xf32>
    %v2911 = stablehlo.subtract %v2905, %v2910 : tensor<32x128x28x28xf32>
    %v2912 = stablehlo.multiply %v2911, %v2911 : tensor<32x128x28x28xf32>
    %v2913 = stablehlo.reduce(%v2912 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2914 = stablehlo.broadcast_in_dim %v2913, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2915 = stablehlo.divide %v2914, %v2907 : tensor<32x128x28x28xf32>
    %v2916 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2917 = stablehlo.add %v2915, %v2916 : tensor<32x128x28x28xf32>
    %v2918 = stablehlo.rsqrt %v2917 : tensor<32x128x28x28xf32>
    %v2919 = stablehlo.multiply %v2911, %v2918 : tensor<32x128x28x28xf32>
    %v2920 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2921 = stablehlo.multiply %v2920, %v2919 : tensor<32x128x28x28xf32>
    %v2922 = stablehlo.reduce(%v2921 init: %v2906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2923 = stablehlo.reshape %v2859 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2925 = stablehlo.reduce(%v2923 init: %v2924) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2926 = stablehlo.reshape %v334 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2927 = stablehlo.reshape %v2848 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2928 = stablehlo.transpose %v2926, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2929 = stablehlo.transpose %v2927, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v2930 = stablehlo.convolution(%v2928, %v2929)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2931 = stablehlo.transpose %v2930, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2932 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2935 = stablehlo.reduce(%v2932 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2936 = stablehlo.broadcast_in_dim %v2935, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2937 = stablehlo.divide %v2936, %v2934 : tensor<32x128x28x28xf32>
    %v2938 = stablehlo.subtract %v2932, %v2937 : tensor<32x128x28x28xf32>
    %v2939 = stablehlo.multiply %v2938, %v2938 : tensor<32x128x28x28xf32>
    %v2940 = stablehlo.reduce(%v2939 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2941 = stablehlo.broadcast_in_dim %v2940, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2942 = stablehlo.divide %v2941, %v2934 : tensor<32x128x28x28xf32>
    %v2943 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2944 = stablehlo.add %v2942, %v2943 : tensor<32x128x28x28xf32>
    %v2945 = stablehlo.rsqrt %v2944 : tensor<32x128x28x28xf32>
    %v2946 = stablehlo.multiply %v2938, %v2945 : tensor<32x128x28x28xf32>
    %v2947 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2948 = stablehlo.multiply %v2947, %v2946 : tensor<32x128x28x28xf32>
    %v2949 = stablehlo.reduce(%v2948 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2950 = stablehlo.reshape %v2818 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2952 = stablehlo.reduce(%v2950 init: %v2951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2953 = stablehlo.reshape %v2898 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2954 = stablehlo.reshape %v301 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2956 = stablehlo.compare GT, %v2954, %v2955 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2957 = stablehlo.select %v2956, %v2953, %v2955 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2958 = stablehlo.reshape %v2957 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2959 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2961 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v2962 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v2963 = stablehlo.reduce(%v2959 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2964 = stablehlo.broadcast_in_dim %v2963, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2965 = stablehlo.divide %v2964, %v2961 : tensor<32x128x28x28xf32>
    %v2966 = stablehlo.subtract %v2959, %v2965 : tensor<32x128x28x28xf32>
    %v2967 = stablehlo.multiply %v2966, %v2966 : tensor<32x128x28x28xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2970 = stablehlo.divide %v2969, %v2961 : tensor<32x128x28x28xf32>
    %v2971 = stablehlo.add %v2970, %v2962 : tensor<32x128x28x28xf32>
    %v2972 = stablehlo.rsqrt %v2971 : tensor<32x128x28x28xf32>
    %v2973 = stablehlo.multiply %v2966, %v2972 : tensor<32x128x28x28xf32>
    %v2974 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2975 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<32x128x28x28xf32>
    %v2977 = stablehlo.reduce(%v2976 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2979 = stablehlo.multiply %v2973, %v2976 : tensor<32x128x28x28xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v2982 = stablehlo.multiply %v2976, %v2961 : tensor<32x128x28x28xf32>
    %v2983 = stablehlo.subtract %v2982, %v2978 : tensor<32x128x28x28xf32>
    %v2984 = stablehlo.multiply %v2973, %v2981 : tensor<32x128x28x28xf32>
    %v2985 = stablehlo.subtract %v2983, %v2984 : tensor<32x128x28x28xf32>
    %v2986 = stablehlo.divide %v2972, %v2961 : tensor<32x128x28x28xf32>
    %v2987 = stablehlo.multiply %v2986, %v2985 : tensor<32x128x28x28xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2990 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2991 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2992 = stablehlo.convolution(%v2989, %v2991)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v2994 = stablehlo.reshape %v2993 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2995 = stablehlo.reshape %v243 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v2996 = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %v2997 = stablehlo.compare GT, %v2995, %v2996 : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %v2998 = stablehlo.select %v2997, %v2994, %v2996 : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %v2999 = stablehlo.reshape %v2998 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3000 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3002 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3003 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3004 = stablehlo.reduce(%v3000 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3005 = stablehlo.broadcast_in_dim %v3004, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3006 = stablehlo.divide %v3005, %v3002 : tensor<32x128x28x28xf32>
    %v3007 = stablehlo.subtract %v3000, %v3006 : tensor<32x128x28x28xf32>
    %v3008 = stablehlo.multiply %v3007, %v3007 : tensor<32x128x28x28xf32>
    %v3009 = stablehlo.reduce(%v3008 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3010 = stablehlo.broadcast_in_dim %v3009, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3011 = stablehlo.divide %v3010, %v3002 : tensor<32x128x28x28xf32>
    %v3012 = stablehlo.add %v3011, %v3003 : tensor<32x128x28x28xf32>
    %v3013 = stablehlo.rsqrt %v3012 : tensor<32x128x28x28xf32>
    %v3014 = stablehlo.multiply %v3007, %v3013 : tensor<32x128x28x28xf32>
    %v3015 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3016 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3017 = stablehlo.multiply %v3015, %v3016 : tensor<32x128x28x28xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3020 = stablehlo.multiply %v3014, %v3017 : tensor<32x128x28x28xf32>
    %v3021 = stablehlo.reduce(%v3020 init: %v3001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3022 = stablehlo.broadcast_in_dim %v3021, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3023 = stablehlo.multiply %v3017, %v3002 : tensor<32x128x28x28xf32>
    %v3024 = stablehlo.subtract %v3023, %v3019 : tensor<32x128x28x28xf32>
    %v3025 = stablehlo.multiply %v3014, %v3022 : tensor<32x128x28x28xf32>
    %v3026 = stablehlo.subtract %v3024, %v3025 : tensor<32x128x28x28xf32>
    %v3027 = stablehlo.divide %v3013, %v3002 : tensor<32x128x28x28xf32>
    %v3028 = stablehlo.multiply %v3027, %v3026 : tensor<32x128x28x28xf32>
    %v3029 = stablehlo.reshape %v3028 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3032 = stablehlo.pad %v3030, %v3031, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3033 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3034 = stablehlo.transpose %v3033, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3035 = stablehlo.convolution(%v3032, %v3034)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3037 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3039 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3040 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3041 = stablehlo.reduce(%v3037 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3043 = stablehlo.divide %v3042, %v3039 : tensor<32x128x28x28xf32>
    %v3044 = stablehlo.subtract %v3037, %v3043 : tensor<32x128x28x28xf32>
    %v3045 = stablehlo.multiply %v3044, %v3044 : tensor<32x128x28x28xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3048 = stablehlo.divide %v3047, %v3039 : tensor<32x128x28x28xf32>
    %v3049 = stablehlo.add %v3048, %v3040 : tensor<32x128x28x28xf32>
    %v3050 = stablehlo.rsqrt %v3049 : tensor<32x128x28x28xf32>
    %v3051 = stablehlo.multiply %v3044, %v3050 : tensor<32x128x28x28xf32>
    %v3052 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3053 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3054 = stablehlo.multiply %v3052, %v3053 : tensor<32x128x28x28xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3057 = stablehlo.multiply %v3051, %v3054 : tensor<32x128x28x28xf32>
    %v3058 = stablehlo.reduce(%v3057 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3059 = stablehlo.broadcast_in_dim %v3058, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3060 = stablehlo.multiply %v3054, %v3039 : tensor<32x128x28x28xf32>
    %v3061 = stablehlo.subtract %v3060, %v3056 : tensor<32x128x28x28xf32>
    %v3062 = stablehlo.multiply %v3051, %v3059 : tensor<32x128x28x28xf32>
    %v3063 = stablehlo.subtract %v3061, %v3062 : tensor<32x128x28x28xf32>
    %v3064 = stablehlo.divide %v3050, %v3039 : tensor<32x128x28x28xf32>
    %v3065 = stablehlo.multiply %v3064, %v3063 : tensor<32x128x28x28xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<32x128x28x28xf32>) -> tensor<32x100352xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3069 = stablehlo.pad %v3067, %v3068, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3070 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3071 = stablehlo.transpose %v3070, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3072 = stablehlo.convolution(%v3069, %v3071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x1x1xf32>) -> tensor<32x64x56x56xf32>
    %v3073 = stablehlo.reshape %v3072 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3074 = stablehlo.reshape %v3036 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3075 = stablehlo.reshape %v3073 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3076 = stablehlo.add %v3074, %v3075 : tensor<32x64x56x56xf32>
    %v3077 = stablehlo.reshape %v3076 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3078 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3079 = stablehlo.reshape %v3029 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3081 = stablehlo.pad %v3079, %v3080, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3082 = stablehlo.transpose %v3078, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3083 = stablehlo.transpose %v3081, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3084 = stablehlo.convolution(%v3082, %v3083)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v3085 = stablehlo.transpose %v3084, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3086 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3088 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3089 = stablehlo.reduce(%v3086 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3091 = stablehlo.divide %v3090, %v3088 : tensor<32x128x28x28xf32>
    %v3092 = stablehlo.subtract %v3086, %v3091 : tensor<32x128x28x28xf32>
    %v3093 = stablehlo.multiply %v3092, %v3092 : tensor<32x128x28x28xf32>
    %v3094 = stablehlo.reduce(%v3093 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3095 = stablehlo.broadcast_in_dim %v3094, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3096 = stablehlo.divide %v3095, %v3088 : tensor<32x128x28x28xf32>
    %v3097 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3098 = stablehlo.add %v3096, %v3097 : tensor<32x128x28x28xf32>
    %v3099 = stablehlo.rsqrt %v3098 : tensor<32x128x28x28xf32>
    %v3100 = stablehlo.multiply %v3092, %v3099 : tensor<32x128x28x28xf32>
    %v3101 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3102 = stablehlo.multiply %v3101, %v3100 : tensor<32x128x28x28xf32>
    %v3103 = stablehlo.reduce(%v3102 init: %v3087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3104 = stablehlo.reshape %v2999 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3106 = stablehlo.reduce(%v3104 init: %v3105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3107 = stablehlo.reshape %v247 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3108 = stablehlo.reshape %v2988 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3109 = stablehlo.transpose %v3107, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3110 = stablehlo.transpose %v3108, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %v3111 = stablehlo.convolution(%v3109, %v3110)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3112 = stablehlo.transpose %v3111, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3113 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3115 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3116 = stablehlo.reduce(%v3113 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3117 = stablehlo.broadcast_in_dim %v3116, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3118 = stablehlo.divide %v3117, %v3115 : tensor<32x128x28x28xf32>
    %v3119 = stablehlo.subtract %v3113, %v3118 : tensor<32x128x28x28xf32>
    %v3120 = stablehlo.multiply %v3119, %v3119 : tensor<32x128x28x28xf32>
    %v3121 = stablehlo.reduce(%v3120 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3122 = stablehlo.broadcast_in_dim %v3121, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3123 = stablehlo.divide %v3122, %v3115 : tensor<32x128x28x28xf32>
    %v3124 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3125 = stablehlo.add %v3123, %v3124 : tensor<32x128x28x28xf32>
    %v3126 = stablehlo.rsqrt %v3125 : tensor<32x128x28x28xf32>
    %v3127 = stablehlo.multiply %v3119, %v3126 : tensor<32x128x28x28xf32>
    %v3128 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3129 = stablehlo.multiply %v3128, %v3127 : tensor<32x128x28x28xf32>
    %v3130 = stablehlo.reduce(%v3129 init: %v3114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3131 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3133 = stablehlo.reduce(%v3131 init: %v3132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3134 = stablehlo.reshape %v218 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3135 = stablehlo.reshape %v3066 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3137 = stablehlo.pad %v3135, %v3136, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %v3138 = stablehlo.transpose %v3134, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3139 = stablehlo.transpose %v3137, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %v3140 = stablehlo.convolution(%v3138, %v3139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x1x1xf32>
    %v3141 = stablehlo.transpose %v3140, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3142 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3144 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3145 = stablehlo.reduce(%v3142 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3146 = stablehlo.broadcast_in_dim %v3145, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3147 = stablehlo.divide %v3146, %v3144 : tensor<32x128x28x28xf32>
    %v3148 = stablehlo.subtract %v3142, %v3147 : tensor<32x128x28x28xf32>
    %v3149 = stablehlo.multiply %v3148, %v3148 : tensor<32x128x28x28xf32>
    %v3150 = stablehlo.reduce(%v3149 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3151 = stablehlo.broadcast_in_dim %v3150, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3152 = stablehlo.divide %v3151, %v3144 : tensor<32x128x28x28xf32>
    %v3153 = stablehlo.constant dense<1.0e-05> : tensor<32x128x28x28xf32>
    %v3154 = stablehlo.add %v3152, %v3153 : tensor<32x128x28x28xf32>
    %v3155 = stablehlo.rsqrt %v3154 : tensor<32x128x28x28xf32>
    %v3156 = stablehlo.multiply %v3148, %v3155 : tensor<32x128x28x28xf32>
    %v3157 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3158 = stablehlo.multiply %v3157, %v3156 : tensor<32x128x28x28xf32>
    %v3159 = stablehlo.reduce(%v3158 init: %v3143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3160 = stablehlo.reshape %v2958 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3162 = stablehlo.reduce(%v3160 init: %v3161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3163 = stablehlo.reshape %v3077 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3164 = stablehlo.reshape %v214 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3165 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3166 = stablehlo.compare GT, %v3164, %v3165 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3167 = stablehlo.select %v3166, %v3163, %v3165 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3168 = stablehlo.reshape %v3167 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3169 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3171 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3172 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3173 = stablehlo.reduce(%v3169 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3174 = stablehlo.broadcast_in_dim %v3173, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3175 = stablehlo.divide %v3174, %v3171 : tensor<32x64x56x56xf32>
    %v3176 = stablehlo.subtract %v3169, %v3175 : tensor<32x64x56x56xf32>
    %v3177 = stablehlo.multiply %v3176, %v3176 : tensor<32x64x56x56xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3179 = stablehlo.broadcast_in_dim %v3178, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3180 = stablehlo.divide %v3179, %v3171 : tensor<32x64x56x56xf32>
    %v3181 = stablehlo.add %v3180, %v3172 : tensor<32x64x56x56xf32>
    %v3182 = stablehlo.rsqrt %v3181 : tensor<32x64x56x56xf32>
    %v3183 = stablehlo.multiply %v3176, %v3182 : tensor<32x64x56x56xf32>
    %v3184 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3185 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3186 = stablehlo.multiply %v3184, %v3185 : tensor<32x64x56x56xf32>
    %v3187 = stablehlo.reduce(%v3186 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3188 = stablehlo.broadcast_in_dim %v3187, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3189 = stablehlo.multiply %v3183, %v3186 : tensor<32x64x56x56xf32>
    %v3190 = stablehlo.reduce(%v3189 init: %v3170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3191 = stablehlo.broadcast_in_dim %v3190, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3192 = stablehlo.multiply %v3186, %v3171 : tensor<32x64x56x56xf32>
    %v3193 = stablehlo.subtract %v3192, %v3188 : tensor<32x64x56x56xf32>
    %v3194 = stablehlo.multiply %v3183, %v3191 : tensor<32x64x56x56xf32>
    %v3195 = stablehlo.subtract %v3193, %v3194 : tensor<32x64x56x56xf32>
    %v3196 = stablehlo.divide %v3182, %v3171 : tensor<32x64x56x56xf32>
    %v3197 = stablehlo.multiply %v3196, %v3195 : tensor<32x64x56x56xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3199 = stablehlo.reshape %v3198 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3200 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3201 = stablehlo.transpose %v3200, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3202 = stablehlo.convolution(%v3199, %v3201)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3203 = stablehlo.reshape %v3202 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3204 = stablehlo.reshape %v3203 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3205 = stablehlo.reshape %v181 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3206 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3207 = stablehlo.compare GT, %v3205, %v3206 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3208 = stablehlo.select %v3207, %v3204, %v3206 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3209 = stablehlo.reshape %v3208 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3210 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3213 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3214 = stablehlo.reduce(%v3210 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3215 = stablehlo.broadcast_in_dim %v3214, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3216 = stablehlo.divide %v3215, %v3212 : tensor<32x64x56x56xf32>
    %v3217 = stablehlo.subtract %v3210, %v3216 : tensor<32x64x56x56xf32>
    %v3218 = stablehlo.multiply %v3217, %v3217 : tensor<32x64x56x56xf32>
    %v3219 = stablehlo.reduce(%v3218 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3220 = stablehlo.broadcast_in_dim %v3219, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3221 = stablehlo.divide %v3220, %v3212 : tensor<32x64x56x56xf32>
    %v3222 = stablehlo.add %v3221, %v3213 : tensor<32x64x56x56xf32>
    %v3223 = stablehlo.rsqrt %v3222 : tensor<32x64x56x56xf32>
    %v3224 = stablehlo.multiply %v3217, %v3223 : tensor<32x64x56x56xf32>
    %v3225 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3227 = stablehlo.multiply %v3225, %v3226 : tensor<32x64x56x56xf32>
    %v3228 = stablehlo.reduce(%v3227 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3229 = stablehlo.broadcast_in_dim %v3228, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3230 = stablehlo.multiply %v3224, %v3227 : tensor<32x64x56x56xf32>
    %v3231 = stablehlo.reduce(%v3230 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3232 = stablehlo.broadcast_in_dim %v3231, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3233 = stablehlo.multiply %v3227, %v3212 : tensor<32x64x56x56xf32>
    %v3234 = stablehlo.subtract %v3233, %v3229 : tensor<32x64x56x56xf32>
    %v3235 = stablehlo.multiply %v3224, %v3232 : tensor<32x64x56x56xf32>
    %v3236 = stablehlo.subtract %v3234, %v3235 : tensor<32x64x56x56xf32>
    %v3237 = stablehlo.divide %v3223, %v3212 : tensor<32x64x56x56xf32>
    %v3238 = stablehlo.multiply %v3237, %v3236 : tensor<32x64x56x56xf32>
    %v3239 = stablehlo.reshape %v3238 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3240 = stablehlo.reshape %v3239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3241 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3242 = stablehlo.transpose %v3241, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3243 = stablehlo.convolution(%v3240, %v3242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3244 = stablehlo.reshape %v3243 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3245 = stablehlo.reshape %v3244 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3246 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3247 = stablehlo.add %v3245, %v3246 : tensor<32x64x56x56xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3249 = stablehlo.reshape %v156 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3250 = stablehlo.reshape %v3239 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3251 = stablehlo.transpose %v3249, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3252 = stablehlo.transpose %v3250, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3253 = stablehlo.convolution(%v3251, %v3252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3254 = stablehlo.transpose %v3253, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3255 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3258 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3259 = stablehlo.broadcast_in_dim %v3258, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3260 = stablehlo.divide %v3259, %v3257 : tensor<32x64x56x56xf32>
    %v3261 = stablehlo.subtract %v3255, %v3260 : tensor<32x64x56x56xf32>
    %v3262 = stablehlo.multiply %v3261, %v3261 : tensor<32x64x56x56xf32>
    %v3263 = stablehlo.reduce(%v3262 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3264 = stablehlo.broadcast_in_dim %v3263, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3265 = stablehlo.divide %v3264, %v3257 : tensor<32x64x56x56xf32>
    %v3266 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3267 = stablehlo.add %v3265, %v3266 : tensor<32x64x56x56xf32>
    %v3268 = stablehlo.rsqrt %v3267 : tensor<32x64x56x56xf32>
    %v3269 = stablehlo.multiply %v3261, %v3268 : tensor<32x64x56x56xf32>
    %v3270 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3271 = stablehlo.multiply %v3270, %v3269 : tensor<32x64x56x56xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3273 = stablehlo.reshape %v3209 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3275 = stablehlo.reduce(%v3273 init: %v3274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3276 = stablehlo.reshape %v185 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3277 = stablehlo.reshape %v3198 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3278 = stablehlo.transpose %v3276, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3279 = stablehlo.transpose %v3277, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3280 = stablehlo.convolution(%v3278, %v3279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3281 = stablehlo.transpose %v3280, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3282 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3284 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3285 = stablehlo.reduce(%v3282 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3286 = stablehlo.broadcast_in_dim %v3285, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3287 = stablehlo.divide %v3286, %v3284 : tensor<32x64x56x56xf32>
    %v3288 = stablehlo.subtract %v3282, %v3287 : tensor<32x64x56x56xf32>
    %v3289 = stablehlo.multiply %v3288, %v3288 : tensor<32x64x56x56xf32>
    %v3290 = stablehlo.reduce(%v3289 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3291 = stablehlo.broadcast_in_dim %v3290, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3292 = stablehlo.divide %v3291, %v3284 : tensor<32x64x56x56xf32>
    %v3293 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3294 = stablehlo.add %v3292, %v3293 : tensor<32x64x56x56xf32>
    %v3295 = stablehlo.rsqrt %v3294 : tensor<32x64x56x56xf32>
    %v3296 = stablehlo.multiply %v3288, %v3295 : tensor<32x64x56x56xf32>
    %v3297 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3298 = stablehlo.multiply %v3297, %v3296 : tensor<32x64x56x56xf32>
    %v3299 = stablehlo.reduce(%v3298 init: %v3283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3300 = stablehlo.reshape %v3168 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3302 = stablehlo.reduce(%v3300 init: %v3301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3303 = stablehlo.reshape %v3248 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3304 = stablehlo.reshape %v152 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3305 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3306 = stablehlo.compare GT, %v3304, %v3305 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3307 = stablehlo.select %v3306, %v3303, %v3305 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3309 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3311 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3312 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3313 = stablehlo.reduce(%v3309 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3314 = stablehlo.broadcast_in_dim %v3313, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3315 = stablehlo.divide %v3314, %v3311 : tensor<32x64x56x56xf32>
    %v3316 = stablehlo.subtract %v3309, %v3315 : tensor<32x64x56x56xf32>
    %v3317 = stablehlo.multiply %v3316, %v3316 : tensor<32x64x56x56xf32>
    %v3318 = stablehlo.reduce(%v3317 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3319 = stablehlo.broadcast_in_dim %v3318, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3320 = stablehlo.divide %v3319, %v3311 : tensor<32x64x56x56xf32>
    %v3321 = stablehlo.add %v3320, %v3312 : tensor<32x64x56x56xf32>
    %v3322 = stablehlo.rsqrt %v3321 : tensor<32x64x56x56xf32>
    %v3323 = stablehlo.multiply %v3316, %v3322 : tensor<32x64x56x56xf32>
    %v3324 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3326 = stablehlo.multiply %v3324, %v3325 : tensor<32x64x56x56xf32>
    %v3327 = stablehlo.reduce(%v3326 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3328 = stablehlo.broadcast_in_dim %v3327, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3329 = stablehlo.multiply %v3323, %v3326 : tensor<32x64x56x56xf32>
    %v3330 = stablehlo.reduce(%v3329 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3331 = stablehlo.broadcast_in_dim %v3330, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3332 = stablehlo.multiply %v3326, %v3311 : tensor<32x64x56x56xf32>
    %v3333 = stablehlo.subtract %v3332, %v3328 : tensor<32x64x56x56xf32>
    %v3334 = stablehlo.multiply %v3323, %v3331 : tensor<32x64x56x56xf32>
    %v3335 = stablehlo.subtract %v3333, %v3334 : tensor<32x64x56x56xf32>
    %v3336 = stablehlo.divide %v3322, %v3311 : tensor<32x64x56x56xf32>
    %v3337 = stablehlo.multiply %v3336, %v3335 : tensor<32x64x56x56xf32>
    %v3338 = stablehlo.reshape %v3337 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3339 = stablehlo.reshape %v3338 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3340 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3341 = stablehlo.transpose %v3340, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3342 = stablehlo.convolution(%v3339, %v3341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3343 = stablehlo.reshape %v3342 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3345 = stablehlo.reshape %v119 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3346 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3347 = stablehlo.compare GT, %v3345, %v3346 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3348 = stablehlo.select %v3347, %v3344, %v3346 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3349 = stablehlo.reshape %v3348 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3350 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3352 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3353 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3354 = stablehlo.reduce(%v3350 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3355 = stablehlo.broadcast_in_dim %v3354, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3356 = stablehlo.divide %v3355, %v3352 : tensor<32x64x56x56xf32>
    %v3357 = stablehlo.subtract %v3350, %v3356 : tensor<32x64x56x56xf32>
    %v3358 = stablehlo.multiply %v3357, %v3357 : tensor<32x64x56x56xf32>
    %v3359 = stablehlo.reduce(%v3358 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3360 = stablehlo.broadcast_in_dim %v3359, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3361 = stablehlo.divide %v3360, %v3352 : tensor<32x64x56x56xf32>
    %v3362 = stablehlo.add %v3361, %v3353 : tensor<32x64x56x56xf32>
    %v3363 = stablehlo.rsqrt %v3362 : tensor<32x64x56x56xf32>
    %v3364 = stablehlo.multiply %v3357, %v3363 : tensor<32x64x56x56xf32>
    %v3365 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3366 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3367 = stablehlo.multiply %v3365, %v3366 : tensor<32x64x56x56xf32>
    %v3368 = stablehlo.reduce(%v3367 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3368, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3370 = stablehlo.multiply %v3364, %v3367 : tensor<32x64x56x56xf32>
    %v3371 = stablehlo.reduce(%v3370 init: %v3351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3372 = stablehlo.broadcast_in_dim %v3371, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3373 = stablehlo.multiply %v3367, %v3352 : tensor<32x64x56x56xf32>
    %v3374 = stablehlo.subtract %v3373, %v3369 : tensor<32x64x56x56xf32>
    %v3375 = stablehlo.multiply %v3364, %v3372 : tensor<32x64x56x56xf32>
    %v3376 = stablehlo.subtract %v3374, %v3375 : tensor<32x64x56x56xf32>
    %v3377 = stablehlo.divide %v3363, %v3352 : tensor<32x64x56x56xf32>
    %v3378 = stablehlo.multiply %v3377, %v3376 : tensor<32x64x56x56xf32>
    %v3379 = stablehlo.reshape %v3378 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3380 = stablehlo.reshape %v3379 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3381 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3382 = stablehlo.transpose %v3381, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3383 = stablehlo.convolution(%v3380, %v3382)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3384 = stablehlo.reshape %v3383 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3386 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3387 = stablehlo.add %v3385, %v3386 : tensor<32x64x56x56xf32>
    %v3388 = stablehlo.reshape %v3387 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3389 = stablehlo.reshape %v94 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3390 = stablehlo.reshape %v3379 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3391 = stablehlo.transpose %v3389, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3392 = stablehlo.transpose %v3390, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3393 = stablehlo.convolution(%v3391, %v3392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3394 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3395 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3397 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3398 = stablehlo.reduce(%v3395 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3399 = stablehlo.broadcast_in_dim %v3398, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3400 = stablehlo.divide %v3399, %v3397 : tensor<32x64x56x56xf32>
    %v3401 = stablehlo.subtract %v3395, %v3400 : tensor<32x64x56x56xf32>
    %v3402 = stablehlo.multiply %v3401, %v3401 : tensor<32x64x56x56xf32>
    %v3403 = stablehlo.reduce(%v3402 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3404 = stablehlo.broadcast_in_dim %v3403, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3405 = stablehlo.divide %v3404, %v3397 : tensor<32x64x56x56xf32>
    %v3406 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3407 = stablehlo.add %v3405, %v3406 : tensor<32x64x56x56xf32>
    %v3408 = stablehlo.rsqrt %v3407 : tensor<32x64x56x56xf32>
    %v3409 = stablehlo.multiply %v3401, %v3408 : tensor<32x64x56x56xf32>
    %v3410 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3411 = stablehlo.multiply %v3410, %v3409 : tensor<32x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.reshape %v3349 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.reduce(%v3413 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3416 = stablehlo.reshape %v123 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3417 = stablehlo.reshape %v3338 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3418 = stablehlo.transpose %v3416, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3419 = stablehlo.transpose %v3417, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3420 = stablehlo.convolution(%v3418, %v3419)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3421 = stablehlo.transpose %v3420, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3422 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3424 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3425 = stablehlo.reduce(%v3422 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3425, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3427 = stablehlo.divide %v3426, %v3424 : tensor<32x64x56x56xf32>
    %v3428 = stablehlo.subtract %v3422, %v3427 : tensor<32x64x56x56xf32>
    %v3429 = stablehlo.multiply %v3428, %v3428 : tensor<32x64x56x56xf32>
    %v3430 = stablehlo.reduce(%v3429 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3432 = stablehlo.divide %v3431, %v3424 : tensor<32x64x56x56xf32>
    %v3433 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3434 = stablehlo.add %v3432, %v3433 : tensor<32x64x56x56xf32>
    %v3435 = stablehlo.rsqrt %v3434 : tensor<32x64x56x56xf32>
    %v3436 = stablehlo.multiply %v3428, %v3435 : tensor<32x64x56x56xf32>
    %v3437 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3438 = stablehlo.multiply %v3437, %v3436 : tensor<32x64x56x56xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3440 = stablehlo.reshape %v3308 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3442 = stablehlo.reduce(%v3440 init: %v3441) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3443 = stablehlo.reshape %v3388 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3444 = stablehlo.reshape %v90 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3445 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3446 = stablehlo.compare GT, %v3444, %v3445 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3447 = stablehlo.select %v3446, %v3443, %v3445 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3448 = stablehlo.reshape %v3447 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3449 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3451 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3452 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3453 = stablehlo.reduce(%v3449 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3454 = stablehlo.broadcast_in_dim %v3453, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3455 = stablehlo.divide %v3454, %v3451 : tensor<32x64x56x56xf32>
    %v3456 = stablehlo.subtract %v3449, %v3455 : tensor<32x64x56x56xf32>
    %v3457 = stablehlo.multiply %v3456, %v3456 : tensor<32x64x56x56xf32>
    %v3458 = stablehlo.reduce(%v3457 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3459 = stablehlo.broadcast_in_dim %v3458, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3460 = stablehlo.divide %v3459, %v3451 : tensor<32x64x56x56xf32>
    %v3461 = stablehlo.add %v3460, %v3452 : tensor<32x64x56x56xf32>
    %v3462 = stablehlo.rsqrt %v3461 : tensor<32x64x56x56xf32>
    %v3463 = stablehlo.multiply %v3456, %v3462 : tensor<32x64x56x56xf32>
    %v3464 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3465 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3466 = stablehlo.multiply %v3464, %v3465 : tensor<32x64x56x56xf32>
    %v3467 = stablehlo.reduce(%v3466 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3468 = stablehlo.broadcast_in_dim %v3467, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3469 = stablehlo.multiply %v3463, %v3466 : tensor<32x64x56x56xf32>
    %v3470 = stablehlo.reduce(%v3469 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3471 = stablehlo.broadcast_in_dim %v3470, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3472 = stablehlo.multiply %v3466, %v3451 : tensor<32x64x56x56xf32>
    %v3473 = stablehlo.subtract %v3472, %v3468 : tensor<32x64x56x56xf32>
    %v3474 = stablehlo.multiply %v3463, %v3471 : tensor<32x64x56x56xf32>
    %v3475 = stablehlo.subtract %v3473, %v3474 : tensor<32x64x56x56xf32>
    %v3476 = stablehlo.divide %v3462, %v3451 : tensor<32x64x56x56xf32>
    %v3477 = stablehlo.multiply %v3476, %v3475 : tensor<32x64x56x56xf32>
    %v3478 = stablehlo.reshape %v3477 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3479 = stablehlo.reshape %v3478 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3480 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3481 = stablehlo.transpose %v3480, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3482 = stablehlo.convolution(%v3479, %v3481)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3484 = stablehlo.reshape %v3483 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3485 = stablehlo.reshape %v57 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3486 = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %v3487 = stablehlo.compare GT, %v3485, %v3486 : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %v3488 = stablehlo.select %v3487, %v3484, %v3486 : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3490 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3493 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3494 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3496 = stablehlo.divide %v3495, %v3492 : tensor<32x64x56x56xf32>
    %v3497 = stablehlo.subtract %v3490, %v3496 : tensor<32x64x56x56xf32>
    %v3498 = stablehlo.multiply %v3497, %v3497 : tensor<32x64x56x56xf32>
    %v3499 = stablehlo.reduce(%v3498 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3501 = stablehlo.divide %v3500, %v3492 : tensor<32x64x56x56xf32>
    %v3502 = stablehlo.add %v3501, %v3493 : tensor<32x64x56x56xf32>
    %v3503 = stablehlo.rsqrt %v3502 : tensor<32x64x56x56xf32>
    %v3504 = stablehlo.multiply %v3497, %v3503 : tensor<32x64x56x56xf32>
    %v3505 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3506 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3507 = stablehlo.multiply %v3505, %v3506 : tensor<32x64x56x56xf32>
    %v3508 = stablehlo.reduce(%v3507 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3509 = stablehlo.broadcast_in_dim %v3508, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3510 = stablehlo.multiply %v3504, %v3507 : tensor<32x64x56x56xf32>
    %v3511 = stablehlo.reduce(%v3510 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3512 = stablehlo.broadcast_in_dim %v3511, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3513 = stablehlo.multiply %v3507, %v3492 : tensor<32x64x56x56xf32>
    %v3514 = stablehlo.subtract %v3513, %v3509 : tensor<32x64x56x56xf32>
    %v3515 = stablehlo.multiply %v3504, %v3512 : tensor<32x64x56x56xf32>
    %v3516 = stablehlo.subtract %v3514, %v3515 : tensor<32x64x56x56xf32>
    %v3517 = stablehlo.divide %v3503, %v3492 : tensor<32x64x56x56xf32>
    %v3518 = stablehlo.multiply %v3517, %v3516 : tensor<32x64x56x56xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3520 = stablehlo.reshape %v3519 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3521 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3522 = stablehlo.transpose %v3521, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3523 = stablehlo.convolution(%v3520, %v3522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3525 = stablehlo.reshape %v3524 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3526 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3527 = stablehlo.add %v3525, %v3526 : tensor<32x64x56x56xf32>
    %v3528 = stablehlo.reshape %v3527 : (tensor<32x64x56x56xf32>) -> tensor<32x200704xf32>
    %v3529 = stablehlo.reshape %v32 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3530 = stablehlo.reshape %v3519 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3531 = stablehlo.transpose %v3529, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3532 = stablehlo.transpose %v3530, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3533 = stablehlo.convolution(%v3531, %v3532)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3534 = stablehlo.transpose %v3533, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3535 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3537 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3538 = stablehlo.reduce(%v3535 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3539 = stablehlo.broadcast_in_dim %v3538, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3540 = stablehlo.divide %v3539, %v3537 : tensor<32x64x56x56xf32>
    %v3541 = stablehlo.subtract %v3535, %v3540 : tensor<32x64x56x56xf32>
    %v3542 = stablehlo.multiply %v3541, %v3541 : tensor<32x64x56x56xf32>
    %v3543 = stablehlo.reduce(%v3542 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3544 = stablehlo.broadcast_in_dim %v3543, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3545 = stablehlo.divide %v3544, %v3537 : tensor<32x64x56x56xf32>
    %v3546 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3547 = stablehlo.add %v3545, %v3546 : tensor<32x64x56x56xf32>
    %v3548 = stablehlo.rsqrt %v3547 : tensor<32x64x56x56xf32>
    %v3549 = stablehlo.multiply %v3541, %v3548 : tensor<32x64x56x56xf32>
    %v3550 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3551 = stablehlo.multiply %v3550, %v3549 : tensor<32x64x56x56xf32>
    %v3552 = stablehlo.reduce(%v3551 init: %v3536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3553 = stablehlo.reshape %v3489 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3556 = stablehlo.reshape %v61 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3557 = stablehlo.reshape %v3478 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3558 = stablehlo.transpose %v3556, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3559 = stablehlo.transpose %v3557, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %v3560 = stablehlo.convolution(%v3558, %v3559)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3561 = stablehlo.transpose %v3560, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3562 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3564 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3565 = stablehlo.reduce(%v3562 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3566 = stablehlo.broadcast_in_dim %v3565, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3567 = stablehlo.divide %v3566, %v3564 : tensor<32x64x56x56xf32>
    %v3568 = stablehlo.subtract %v3562, %v3567 : tensor<32x64x56x56xf32>
    %v3569 = stablehlo.multiply %v3568, %v3568 : tensor<32x64x56x56xf32>
    %v3570 = stablehlo.reduce(%v3569 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3571 = stablehlo.broadcast_in_dim %v3570, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3572 = stablehlo.divide %v3571, %v3564 : tensor<32x64x56x56xf32>
    %v3573 = stablehlo.constant dense<1.0e-05> : tensor<32x64x56x56xf32>
    %v3574 = stablehlo.add %v3572, %v3573 : tensor<32x64x56x56xf32>
    %v3575 = stablehlo.rsqrt %v3574 : tensor<32x64x56x56xf32>
    %v3576 = stablehlo.multiply %v3568, %v3575 : tensor<32x64x56x56xf32>
    %v3577 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3578 = stablehlo.multiply %v3577, %v3576 : tensor<32x64x56x56xf32>
    %v3579 = stablehlo.reduce(%v3578 init: %v3563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3580 = stablehlo.reshape %v3448 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3582 = stablehlo.reduce(%v3580 init: %v3581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3583 = stablehlo.reshape %v28 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3584 = stablehlo.reshape %v3528 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = "stablehlo.select_and_scatter"(%v3583, %v3584, %v3585) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3588 = stablehlo.reshape %v3587 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3589 = stablehlo.reshape %v24 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %v3591 = stablehlo.compare GT, %v3589, %v3590 : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %v3592 = stablehlo.select %v3591, %v3588, %v3590 : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3594 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3596 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3597 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3598 = stablehlo.reduce(%v3594 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3599 = stablehlo.broadcast_in_dim %v3598, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3600 = stablehlo.divide %v3599, %v3596 : tensor<32x64x112x112xf32>
    %v3601 = stablehlo.subtract %v3594, %v3600 : tensor<32x64x112x112xf32>
    %v3602 = stablehlo.multiply %v3601, %v3601 : tensor<32x64x112x112xf32>
    %v3603 = stablehlo.reduce(%v3602 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3604 = stablehlo.broadcast_in_dim %v3603, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3605 = stablehlo.divide %v3604, %v3596 : tensor<32x64x112x112xf32>
    %v3606 = stablehlo.add %v3605, %v3597 : tensor<32x64x112x112xf32>
    %v3607 = stablehlo.rsqrt %v3606 : tensor<32x64x112x112xf32>
    %v3608 = stablehlo.multiply %v3601, %v3607 : tensor<32x64x112x112xf32>
    %v3609 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3610 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3611 = stablehlo.multiply %v3609, %v3610 : tensor<32x64x112x112xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3614 = stablehlo.multiply %v3608, %v3611 : tensor<32x64x112x112xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3616 = stablehlo.broadcast_in_dim %v3615, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3617 = stablehlo.multiply %v3611, %v3596 : tensor<32x64x112x112xf32>
    %v3618 = stablehlo.subtract %v3617, %v3613 : tensor<32x64x112x112xf32>
    %v3619 = stablehlo.multiply %v3608, %v3616 : tensor<32x64x112x112xf32>
    %v3620 = stablehlo.subtract %v3618, %v3619 : tensor<32x64x112x112xf32>
    %v3621 = stablehlo.divide %v3607, %v3596 : tensor<32x64x112x112xf32>
    %v3622 = stablehlo.multiply %v3621, %v3620 : tensor<32x64x112x112xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x64x112x112xf32>) -> tensor<32x802816xf32>
    %v3624 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v3625 = stablehlo.reshape %v3623 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3626 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3627 = stablehlo.pad %v3625, %v3626, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %v3628 = stablehlo.transpose %v3624, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v3629 = stablehlo.transpose %v3627, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %v3630 = stablehlo.convolution(%v3628, %v3629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3631 = stablehlo.transpose %v3630, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3632 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3635 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3636 = stablehlo.broadcast_in_dim %v3635, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3637 = stablehlo.divide %v3636, %v3634 : tensor<32x64x112x112xf32>
    %v3638 = stablehlo.subtract %v3632, %v3637 : tensor<32x64x112x112xf32>
    %v3639 = stablehlo.multiply %v3638, %v3638 : tensor<32x64x112x112xf32>
    %v3640 = stablehlo.reduce(%v3639 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3640, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3642 = stablehlo.divide %v3641, %v3634 : tensor<32x64x112x112xf32>
    %v3643 = stablehlo.constant dense<1.0e-05> : tensor<32x64x112x112xf32>
    %v3644 = stablehlo.add %v3642, %v3643 : tensor<32x64x112x112xf32>
    %v3645 = stablehlo.rsqrt %v3644 : tensor<32x64x112x112xf32>
    %v3646 = stablehlo.multiply %v3638, %v3645 : tensor<32x64x112x112xf32>
    %v3647 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3648 = stablehlo.multiply %v3647, %v3646 : tensor<32x64x112x112xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3650 = stablehlo.reshape %v3593 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3651 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3652 = stablehlo.reduce(%v3650 init: %v3651) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3653 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3655 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3656 = stablehlo.reduce(%v3653 init: %v3654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3657 = stablehlo.divide %v3656, %v3655 : tensor<64xf32>
    %v3658 = stablehlo.reshape %v4 : (tensor<32x802816xf32>) -> tensor<32x64x112x112xf32>
    %v3659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3660 = stablehlo.constant dense<401408.0> : tensor<32x64x112x112xf32>
    %v3661 = stablehlo.reduce(%v3658 init: %v3659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %v3663 = stablehlo.divide %v3662, %v3660 : tensor<32x64x112x112xf32>
    %v3664 = stablehlo.subtract %v3658, %v3663 : tensor<32x64x112x112xf32>
    %v3665 = stablehlo.multiply %v3664, %v3664 : tensor<32x64x112x112xf32>
    %v3666 = stablehlo.reduce(%v3665 init: %v3659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3667 = stablehlo.constant dense<401408.0> : tensor<64xf32>
    %v3668 = stablehlo.divide %v3666, %v3667 : tensor<64xf32>
    %v3669 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3671 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3672 = stablehlo.reduce(%v3669 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3673 = stablehlo.divide %v3672, %v3671 : tensor<64xf32>
    %v3674 = stablehlo.reshape %v37 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3676 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3677 = stablehlo.reduce(%v3674 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3678 = stablehlo.broadcast_in_dim %v3677, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3679 = stablehlo.divide %v3678, %v3676 : tensor<32x64x56x56xf32>
    %v3680 = stablehlo.subtract %v3674, %v3679 : tensor<32x64x56x56xf32>
    %v3681 = stablehlo.multiply %v3680, %v3680 : tensor<32x64x56x56xf32>
    %v3682 = stablehlo.reduce(%v3681 init: %v3675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3683 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3684 = stablehlo.divide %v3682, %v3683 : tensor<64xf32>
    %v3685 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3687 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3688 = stablehlo.reduce(%v3685 init: %v3686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3689 = stablehlo.divide %v3688, %v3687 : tensor<64xf32>
    %v3690 = stablehlo.reshape %v66 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3692 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3693 = stablehlo.reduce(%v3690 init: %v3691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3694 = stablehlo.broadcast_in_dim %v3693, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3695 = stablehlo.divide %v3694, %v3692 : tensor<32x64x56x56xf32>
    %v3696 = stablehlo.subtract %v3690, %v3695 : tensor<32x64x56x56xf32>
    %v3697 = stablehlo.multiply %v3696, %v3696 : tensor<32x64x56x56xf32>
    %v3698 = stablehlo.reduce(%v3697 init: %v3691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3699 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3700 = stablehlo.divide %v3698, %v3699 : tensor<64xf32>
    %v3701 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3703 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3704 = stablehlo.reduce(%v3701 init: %v3702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3705 = stablehlo.divide %v3704, %v3703 : tensor<64xf32>
    %v3706 = stablehlo.reshape %v99 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3708 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3709 = stablehlo.reduce(%v3706 init: %v3707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3710 = stablehlo.broadcast_in_dim %v3709, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3711 = stablehlo.divide %v3710, %v3708 : tensor<32x64x56x56xf32>
    %v3712 = stablehlo.subtract %v3706, %v3711 : tensor<32x64x56x56xf32>
    %v3713 = stablehlo.multiply %v3712, %v3712 : tensor<32x64x56x56xf32>
    %v3714 = stablehlo.reduce(%v3713 init: %v3707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3715 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3716 = stablehlo.divide %v3714, %v3715 : tensor<64xf32>
    %v3717 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3719 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3720 = stablehlo.reduce(%v3717 init: %v3718) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3721 = stablehlo.divide %v3720, %v3719 : tensor<64xf32>
    %v3722 = stablehlo.reshape %v128 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3724 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3725 = stablehlo.reduce(%v3722 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3725, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3727 = stablehlo.divide %v3726, %v3724 : tensor<32x64x56x56xf32>
    %v3728 = stablehlo.subtract %v3722, %v3727 : tensor<32x64x56x56xf32>
    %v3729 = stablehlo.multiply %v3728, %v3728 : tensor<32x64x56x56xf32>
    %v3730 = stablehlo.reduce(%v3729 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3731 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3732 = stablehlo.divide %v3730, %v3731 : tensor<64xf32>
    %v3733 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3734 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3735 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3736 = stablehlo.reduce(%v3733 init: %v3734) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3737 = stablehlo.divide %v3736, %v3735 : tensor<64xf32>
    %v3738 = stablehlo.reshape %v161 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3740 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3741 = stablehlo.reduce(%v3738 init: %v3739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3742 = stablehlo.broadcast_in_dim %v3741, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3743 = stablehlo.divide %v3742, %v3740 : tensor<32x64x56x56xf32>
    %v3744 = stablehlo.subtract %v3738, %v3743 : tensor<32x64x56x56xf32>
    %v3745 = stablehlo.multiply %v3744, %v3744 : tensor<32x64x56x56xf32>
    %v3746 = stablehlo.reduce(%v3745 init: %v3739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3747 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3748 = stablehlo.divide %v3746, %v3747 : tensor<64xf32>
    %v3749 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3750 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3751 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3752 = stablehlo.reduce(%v3749 init: %v3750) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3753 = stablehlo.divide %v3752, %v3751 : tensor<64xf32>
    %v3754 = stablehlo.reshape %v190 : (tensor<32x200704xf32>) -> tensor<32x64x56x56xf32>
    %v3755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3756 = stablehlo.constant dense<100352.0> : tensor<32x64x56x56xf32>
    %v3757 = stablehlo.reduce(%v3754 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3758 = stablehlo.broadcast_in_dim %v3757, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %v3759 = stablehlo.divide %v3758, %v3756 : tensor<32x64x56x56xf32>
    %v3760 = stablehlo.subtract %v3754, %v3759 : tensor<32x64x56x56xf32>
    %v3761 = stablehlo.multiply %v3760, %v3760 : tensor<32x64x56x56xf32>
    %v3762 = stablehlo.reduce(%v3761 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3763 = stablehlo.constant dense<100352.0> : tensor<64xf32>
    %v3764 = stablehlo.divide %v3762, %v3763 : tensor<64xf32>
    %v3765 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3767 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3768 = stablehlo.reduce(%v3765 init: %v3766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3769 = stablehlo.divide %v3768, %v3767 : tensor<128xf32>
    %v3770 = stablehlo.reshape %v223 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3772 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3773 = stablehlo.reduce(%v3770 init: %v3771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3774 = stablehlo.broadcast_in_dim %v3773, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3775 = stablehlo.divide %v3774, %v3772 : tensor<32x128x28x28xf32>
    %v3776 = stablehlo.subtract %v3770, %v3775 : tensor<32x128x28x28xf32>
    %v3777 = stablehlo.multiply %v3776, %v3776 : tensor<32x128x28x28xf32>
    %v3778 = stablehlo.reduce(%v3777 init: %v3771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3779 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3780 = stablehlo.divide %v3778, %v3779 : tensor<128xf32>
    %v3781 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3783 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3784 = stablehlo.reduce(%v3781 init: %v3782) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3785 = stablehlo.divide %v3784, %v3783 : tensor<128xf32>
    %v3786 = stablehlo.reshape %v252 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3788 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3789 = stablehlo.reduce(%v3786 init: %v3787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3790 = stablehlo.broadcast_in_dim %v3789, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3791 = stablehlo.divide %v3790, %v3788 : tensor<32x128x28x28xf32>
    %v3792 = stablehlo.subtract %v3786, %v3791 : tensor<32x128x28x28xf32>
    %v3793 = stablehlo.multiply %v3792, %v3792 : tensor<32x128x28x28xf32>
    %v3794 = stablehlo.reduce(%v3793 init: %v3787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3795 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3796 = stablehlo.divide %v3794, %v3795 : tensor<128xf32>
    %v3797 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3800 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3801 = stablehlo.divide %v3800, %v3799 : tensor<128xf32>
    %v3802 = stablehlo.reshape %v277 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3803 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3804 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3805 = stablehlo.reduce(%v3802 init: %v3803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3806 = stablehlo.broadcast_in_dim %v3805, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3807 = stablehlo.divide %v3806, %v3804 : tensor<32x128x28x28xf32>
    %v3808 = stablehlo.subtract %v3802, %v3807 : tensor<32x128x28x28xf32>
    %v3809 = stablehlo.multiply %v3808, %v3808 : tensor<32x128x28x28xf32>
    %v3810 = stablehlo.reduce(%v3809 init: %v3803) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3811 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3812 = stablehlo.divide %v3810, %v3811 : tensor<128xf32>
    %v3813 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3816 = stablehlo.reduce(%v3813 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3817 = stablehlo.divide %v3816, %v3815 : tensor<128xf32>
    %v3818 = stablehlo.reshape %v310 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3820 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3821 = stablehlo.reduce(%v3818 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3822 = stablehlo.broadcast_in_dim %v3821, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3823 = stablehlo.divide %v3822, %v3820 : tensor<32x128x28x28xf32>
    %v3824 = stablehlo.subtract %v3818, %v3823 : tensor<32x128x28x28xf32>
    %v3825 = stablehlo.multiply %v3824, %v3824 : tensor<32x128x28x28xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3827 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3828 = stablehlo.divide %v3826, %v3827 : tensor<128xf32>
    %v3829 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3832 = stablehlo.reduce(%v3829 init: %v3830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3833 = stablehlo.divide %v3832, %v3831 : tensor<128xf32>
    %v3834 = stablehlo.reshape %v339 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3836 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3837 = stablehlo.reduce(%v3834 init: %v3835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3838 = stablehlo.broadcast_in_dim %v3837, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3839 = stablehlo.divide %v3838, %v3836 : tensor<32x128x28x28xf32>
    %v3840 = stablehlo.subtract %v3834, %v3839 : tensor<32x128x28x28xf32>
    %v3841 = stablehlo.multiply %v3840, %v3840 : tensor<32x128x28x28xf32>
    %v3842 = stablehlo.reduce(%v3841 init: %v3835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3843 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3844 = stablehlo.divide %v3842, %v3843 : tensor<128xf32>
    %v3845 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3847 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3848 = stablehlo.reduce(%v3845 init: %v3846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3849 = stablehlo.divide %v3848, %v3847 : tensor<128xf32>
    %v3850 = stablehlo.reshape %v372 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3852 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3853 = stablehlo.reduce(%v3850 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3854 = stablehlo.broadcast_in_dim %v3853, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3855 = stablehlo.divide %v3854, %v3852 : tensor<32x128x28x28xf32>
    %v3856 = stablehlo.subtract %v3850, %v3855 : tensor<32x128x28x28xf32>
    %v3857 = stablehlo.multiply %v3856, %v3856 : tensor<32x128x28x28xf32>
    %v3858 = stablehlo.reduce(%v3857 init: %v3851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3859 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3860 = stablehlo.divide %v3858, %v3859 : tensor<128xf32>
    %v3861 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3863 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3864 = stablehlo.reduce(%v3861 init: %v3862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3865 = stablehlo.divide %v3864, %v3863 : tensor<128xf32>
    %v3866 = stablehlo.reshape %v401 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3868 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3869 = stablehlo.reduce(%v3866 init: %v3867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3871 = stablehlo.divide %v3870, %v3868 : tensor<32x128x28x28xf32>
    %v3872 = stablehlo.subtract %v3866, %v3871 : tensor<32x128x28x28xf32>
    %v3873 = stablehlo.multiply %v3872, %v3872 : tensor<32x128x28x28xf32>
    %v3874 = stablehlo.reduce(%v3873 init: %v3867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3875 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3876 = stablehlo.divide %v3874, %v3875 : tensor<128xf32>
    %v3877 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3880 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3881 = stablehlo.divide %v3880, %v3879 : tensor<128xf32>
    %v3882 = stablehlo.reshape %v434 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3884 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3885 = stablehlo.reduce(%v3882 init: %v3883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3886 = stablehlo.broadcast_in_dim %v3885, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3887 = stablehlo.divide %v3886, %v3884 : tensor<32x128x28x28xf32>
    %v3888 = stablehlo.subtract %v3882, %v3887 : tensor<32x128x28x28xf32>
    %v3889 = stablehlo.multiply %v3888, %v3888 : tensor<32x128x28x28xf32>
    %v3890 = stablehlo.reduce(%v3889 init: %v3883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3891 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3892 = stablehlo.divide %v3890, %v3891 : tensor<128xf32>
    %v3893 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3896 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3897 = stablehlo.divide %v3896, %v3895 : tensor<128xf32>
    %v3898 = stablehlo.reshape %v463 : (tensor<32x100352xf32>) -> tensor<32x128x28x28xf32>
    %v3899 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3900 = stablehlo.constant dense<25088.0> : tensor<32x128x28x28xf32>
    %v3901 = stablehlo.reduce(%v3898 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3902 = stablehlo.broadcast_in_dim %v3901, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %v3903 = stablehlo.divide %v3902, %v3900 : tensor<32x128x28x28xf32>
    %v3904 = stablehlo.subtract %v3898, %v3903 : tensor<32x128x28x28xf32>
    %v3905 = stablehlo.multiply %v3904, %v3904 : tensor<32x128x28x28xf32>
    %v3906 = stablehlo.reduce(%v3905 init: %v3899) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3907 = stablehlo.constant dense<25088.0> : tensor<128xf32>
    %v3908 = stablehlo.divide %v3906, %v3907 : tensor<128xf32>
    %v3909 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3911 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3912 = stablehlo.reduce(%v3909 init: %v3910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3913 = stablehlo.divide %v3912, %v3911 : tensor<256xf32>
    %v3914 = stablehlo.reshape %v496 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3916 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3917 = stablehlo.reduce(%v3914 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3917, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3919 = stablehlo.divide %v3918, %v3916 : tensor<32x256x14x14xf32>
    %v3920 = stablehlo.subtract %v3914, %v3919 : tensor<32x256x14x14xf32>
    %v3921 = stablehlo.multiply %v3920, %v3920 : tensor<32x256x14x14xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3923 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3924 = stablehlo.divide %v3922, %v3923 : tensor<256xf32>
    %v3925 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3927 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3928 = stablehlo.reduce(%v3925 init: %v3926) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3929 = stablehlo.divide %v3928, %v3927 : tensor<256xf32>
    %v3930 = stablehlo.reshape %v525 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3933 = stablehlo.reduce(%v3930 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3934 = stablehlo.broadcast_in_dim %v3933, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3935 = stablehlo.divide %v3934, %v3932 : tensor<32x256x14x14xf32>
    %v3936 = stablehlo.subtract %v3930, %v3935 : tensor<32x256x14x14xf32>
    %v3937 = stablehlo.multiply %v3936, %v3936 : tensor<32x256x14x14xf32>
    %v3938 = stablehlo.reduce(%v3937 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3939 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3940 = stablehlo.divide %v3938, %v3939 : tensor<256xf32>
    %v3941 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3943 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3944 = stablehlo.reduce(%v3941 init: %v3942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3945 = stablehlo.divide %v3944, %v3943 : tensor<256xf32>
    %v3946 = stablehlo.reshape %v550 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3948 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3949 = stablehlo.reduce(%v3946 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3950 = stablehlo.broadcast_in_dim %v3949, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3951 = stablehlo.divide %v3950, %v3948 : tensor<32x256x14x14xf32>
    %v3952 = stablehlo.subtract %v3946, %v3951 : tensor<32x256x14x14xf32>
    %v3953 = stablehlo.multiply %v3952, %v3952 : tensor<32x256x14x14xf32>
    %v3954 = stablehlo.reduce(%v3953 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3955 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3956 = stablehlo.divide %v3954, %v3955 : tensor<256xf32>
    %v3957 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3959 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3960 = stablehlo.reduce(%v3957 init: %v3958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3961 = stablehlo.divide %v3960, %v3959 : tensor<256xf32>
    %v3962 = stablehlo.reshape %v583 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3964 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3965 = stablehlo.reduce(%v3962 init: %v3963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3966 = stablehlo.broadcast_in_dim %v3965, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3967 = stablehlo.divide %v3966, %v3964 : tensor<32x256x14x14xf32>
    %v3968 = stablehlo.subtract %v3962, %v3967 : tensor<32x256x14x14xf32>
    %v3969 = stablehlo.multiply %v3968, %v3968 : tensor<32x256x14x14xf32>
    %v3970 = stablehlo.reduce(%v3969 init: %v3963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3971 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3972 = stablehlo.divide %v3970, %v3971 : tensor<256xf32>
    %v3973 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3975 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3976 = stablehlo.reduce(%v3973 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3977 = stablehlo.divide %v3976, %v3975 : tensor<256xf32>
    %v3978 = stablehlo.reshape %v612 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3979 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3980 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3981 = stablehlo.reduce(%v3978 init: %v3979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3981, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3983 = stablehlo.divide %v3982, %v3980 : tensor<32x256x14x14xf32>
    %v3984 = stablehlo.subtract %v3978, %v3983 : tensor<32x256x14x14xf32>
    %v3985 = stablehlo.multiply %v3984, %v3984 : tensor<32x256x14x14xf32>
    %v3986 = stablehlo.reduce(%v3985 init: %v3979) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3987 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3988 = stablehlo.divide %v3986, %v3987 : tensor<256xf32>
    %v3989 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3991 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v3992 = stablehlo.reduce(%v3989 init: %v3990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3993 = stablehlo.divide %v3992, %v3991 : tensor<256xf32>
    %v3994 = stablehlo.reshape %v645 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v3995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3996 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v3997 = stablehlo.reduce(%v3994 init: %v3995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3998 = stablehlo.broadcast_in_dim %v3997, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v3999 = stablehlo.divide %v3998, %v3996 : tensor<32x256x14x14xf32>
    %v4000 = stablehlo.subtract %v3994, %v3999 : tensor<32x256x14x14xf32>
    %v4001 = stablehlo.multiply %v4000, %v4000 : tensor<32x256x14x14xf32>
    %v4002 = stablehlo.reduce(%v4001 init: %v3995) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4003 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4004 = stablehlo.divide %v4002, %v4003 : tensor<256xf32>
    %v4005 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4007 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4008 = stablehlo.reduce(%v4005 init: %v4006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4009 = stablehlo.divide %v4008, %v4007 : tensor<256xf32>
    %v4010 = stablehlo.reshape %v674 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4012 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4013 = stablehlo.reduce(%v4010 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4013, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4015 = stablehlo.divide %v4014, %v4012 : tensor<32x256x14x14xf32>
    %v4016 = stablehlo.subtract %v4010, %v4015 : tensor<32x256x14x14xf32>
    %v4017 = stablehlo.multiply %v4016, %v4016 : tensor<32x256x14x14xf32>
    %v4018 = stablehlo.reduce(%v4017 init: %v4011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4019 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4020 = stablehlo.divide %v4018, %v4019 : tensor<256xf32>
    %v4021 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4023 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4024 = stablehlo.reduce(%v4021 init: %v4022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4025 = stablehlo.divide %v4024, %v4023 : tensor<256xf32>
    %v4026 = stablehlo.reshape %v707 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4028 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4029 = stablehlo.reduce(%v4026 init: %v4027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4030 = stablehlo.broadcast_in_dim %v4029, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4031 = stablehlo.divide %v4030, %v4028 : tensor<32x256x14x14xf32>
    %v4032 = stablehlo.subtract %v4026, %v4031 : tensor<32x256x14x14xf32>
    %v4033 = stablehlo.multiply %v4032, %v4032 : tensor<32x256x14x14xf32>
    %v4034 = stablehlo.reduce(%v4033 init: %v4027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4035 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4036 = stablehlo.divide %v4034, %v4035 : tensor<256xf32>
    %v4037 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4039 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4040 = stablehlo.reduce(%v4037 init: %v4038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4041 = stablehlo.divide %v4040, %v4039 : tensor<256xf32>
    %v4042 = stablehlo.reshape %v736 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4044 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4045 = stablehlo.reduce(%v4042 init: %v4043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4046 = stablehlo.broadcast_in_dim %v4045, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4047 = stablehlo.divide %v4046, %v4044 : tensor<32x256x14x14xf32>
    %v4048 = stablehlo.subtract %v4042, %v4047 : tensor<32x256x14x14xf32>
    %v4049 = stablehlo.multiply %v4048, %v4048 : tensor<32x256x14x14xf32>
    %v4050 = stablehlo.reduce(%v4049 init: %v4043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4051 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4052 = stablehlo.divide %v4050, %v4051 : tensor<256xf32>
    %v4053 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4055 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4056 = stablehlo.reduce(%v4053 init: %v4054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4057 = stablehlo.divide %v4056, %v4055 : tensor<256xf32>
    %v4058 = stablehlo.reshape %v769 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4060 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4061 = stablehlo.reduce(%v4058 init: %v4059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4062 = stablehlo.broadcast_in_dim %v4061, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4063 = stablehlo.divide %v4062, %v4060 : tensor<32x256x14x14xf32>
    %v4064 = stablehlo.subtract %v4058, %v4063 : tensor<32x256x14x14xf32>
    %v4065 = stablehlo.multiply %v4064, %v4064 : tensor<32x256x14x14xf32>
    %v4066 = stablehlo.reduce(%v4065 init: %v4059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4067 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4068 = stablehlo.divide %v4066, %v4067 : tensor<256xf32>
    %v4069 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4071 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4072 = stablehlo.reduce(%v4069 init: %v4070) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4073 = stablehlo.divide %v4072, %v4071 : tensor<256xf32>
    %v4074 = stablehlo.reshape %v798 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4076 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4077 = stablehlo.reduce(%v4074 init: %v4075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4077, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4079 = stablehlo.divide %v4078, %v4076 : tensor<32x256x14x14xf32>
    %v4080 = stablehlo.subtract %v4074, %v4079 : tensor<32x256x14x14xf32>
    %v4081 = stablehlo.multiply %v4080, %v4080 : tensor<32x256x14x14xf32>
    %v4082 = stablehlo.reduce(%v4081 init: %v4075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4083 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4084 = stablehlo.divide %v4082, %v4083 : tensor<256xf32>
    %v4085 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4087 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4088 = stablehlo.reduce(%v4085 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4089 = stablehlo.divide %v4088, %v4087 : tensor<256xf32>
    %v4090 = stablehlo.reshape %v831 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4092 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4093 = stablehlo.reduce(%v4090 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4095 = stablehlo.divide %v4094, %v4092 : tensor<32x256x14x14xf32>
    %v4096 = stablehlo.subtract %v4090, %v4095 : tensor<32x256x14x14xf32>
    %v4097 = stablehlo.multiply %v4096, %v4096 : tensor<32x256x14x14xf32>
    %v4098 = stablehlo.reduce(%v4097 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4099 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4100 = stablehlo.divide %v4098, %v4099 : tensor<256xf32>
    %v4101 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4103 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4104 = stablehlo.reduce(%v4101 init: %v4102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4105 = stablehlo.divide %v4104, %v4103 : tensor<256xf32>
    %v4106 = stablehlo.reshape %v860 : (tensor<32x50176xf32>) -> tensor<32x256x14x14xf32>
    %v4107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4108 = stablehlo.constant dense<6272.0> : tensor<32x256x14x14xf32>
    %v4109 = stablehlo.reduce(%v4106 init: %v4107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4110 = stablehlo.broadcast_in_dim %v4109, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %v4111 = stablehlo.divide %v4110, %v4108 : tensor<32x256x14x14xf32>
    %v4112 = stablehlo.subtract %v4106, %v4111 : tensor<32x256x14x14xf32>
    %v4113 = stablehlo.multiply %v4112, %v4112 : tensor<32x256x14x14xf32>
    %v4114 = stablehlo.reduce(%v4113 init: %v4107) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4115 = stablehlo.constant dense<6272.0> : tensor<256xf32>
    %v4116 = stablehlo.divide %v4114, %v4115 : tensor<256xf32>
    %v4117 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4119 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4120 = stablehlo.reduce(%v4117 init: %v4118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4121 = stablehlo.divide %v4120, %v4119 : tensor<512xf32>
    %v4122 = stablehlo.reshape %v893 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4124 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4125 = stablehlo.reduce(%v4122 init: %v4123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4126 = stablehlo.broadcast_in_dim %v4125, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4127 = stablehlo.divide %v4126, %v4124 : tensor<32x512x7x7xf32>
    %v4128 = stablehlo.subtract %v4122, %v4127 : tensor<32x512x7x7xf32>
    %v4129 = stablehlo.multiply %v4128, %v4128 : tensor<32x512x7x7xf32>
    %v4130 = stablehlo.reduce(%v4129 init: %v4123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4131 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4132 = stablehlo.divide %v4130, %v4131 : tensor<512xf32>
    %v4133 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4135 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4136 = stablehlo.reduce(%v4133 init: %v4134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4137 = stablehlo.divide %v4136, %v4135 : tensor<512xf32>
    %v4138 = stablehlo.reshape %v922 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4140 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4141 = stablehlo.reduce(%v4138 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4142 = stablehlo.broadcast_in_dim %v4141, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4143 = stablehlo.divide %v4142, %v4140 : tensor<32x512x7x7xf32>
    %v4144 = stablehlo.subtract %v4138, %v4143 : tensor<32x512x7x7xf32>
    %v4145 = stablehlo.multiply %v4144, %v4144 : tensor<32x512x7x7xf32>
    %v4146 = stablehlo.reduce(%v4145 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4147 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4148 = stablehlo.divide %v4146, %v4147 : tensor<512xf32>
    %v4149 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4151 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4152 = stablehlo.reduce(%v4149 init: %v4150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4153 = stablehlo.divide %v4152, %v4151 : tensor<512xf32>
    %v4154 = stablehlo.reshape %v947 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4156 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4157 = stablehlo.reduce(%v4154 init: %v4155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4158 = stablehlo.broadcast_in_dim %v4157, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4159 = stablehlo.divide %v4158, %v4156 : tensor<32x512x7x7xf32>
    %v4160 = stablehlo.subtract %v4154, %v4159 : tensor<32x512x7x7xf32>
    %v4161 = stablehlo.multiply %v4160, %v4160 : tensor<32x512x7x7xf32>
    %v4162 = stablehlo.reduce(%v4161 init: %v4155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4163 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4164 = stablehlo.divide %v4162, %v4163 : tensor<512xf32>
    %v4165 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4167 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4168 = stablehlo.reduce(%v4165 init: %v4166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4169 = stablehlo.divide %v4168, %v4167 : tensor<512xf32>
    %v4170 = stablehlo.reshape %v980 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4172 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4173 = stablehlo.reduce(%v4170 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4174 = stablehlo.broadcast_in_dim %v4173, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4175 = stablehlo.divide %v4174, %v4172 : tensor<32x512x7x7xf32>
    %v4176 = stablehlo.subtract %v4170, %v4175 : tensor<32x512x7x7xf32>
    %v4177 = stablehlo.multiply %v4176, %v4176 : tensor<32x512x7x7xf32>
    %v4178 = stablehlo.reduce(%v4177 init: %v4171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4179 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4180 = stablehlo.divide %v4178, %v4179 : tensor<512xf32>
    %v4181 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4182 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4183 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4184 = stablehlo.reduce(%v4181 init: %v4182) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4185 = stablehlo.divide %v4184, %v4183 : tensor<512xf32>
    %v4186 = stablehlo.reshape %v1009 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4188 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4189 = stablehlo.reduce(%v4186 init: %v4187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4190 = stablehlo.broadcast_in_dim %v4189, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4191 = stablehlo.divide %v4190, %v4188 : tensor<32x512x7x7xf32>
    %v4192 = stablehlo.subtract %v4186, %v4191 : tensor<32x512x7x7xf32>
    %v4193 = stablehlo.multiply %v4192, %v4192 : tensor<32x512x7x7xf32>
    %v4194 = stablehlo.reduce(%v4193 init: %v4187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4195 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4196 = stablehlo.divide %v4194, %v4195 : tensor<512xf32>
    %v4197 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4199 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4200 = stablehlo.reduce(%v4197 init: %v4198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4201 = stablehlo.divide %v4200, %v4199 : tensor<512xf32>
    %v4202 = stablehlo.reshape %v1042 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4205 = stablehlo.reduce(%v4202 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4206 = stablehlo.broadcast_in_dim %v4205, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4207 = stablehlo.divide %v4206, %v4204 : tensor<32x512x7x7xf32>
    %v4208 = stablehlo.subtract %v4202, %v4207 : tensor<32x512x7x7xf32>
    %v4209 = stablehlo.multiply %v4208, %v4208 : tensor<32x512x7x7xf32>
    %v4210 = stablehlo.reduce(%v4209 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4211 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4212 = stablehlo.divide %v4210, %v4211 : tensor<512xf32>
    %v4213 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4215 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4216 = stablehlo.reduce(%v4213 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4217 = stablehlo.divide %v4216, %v4215 : tensor<512xf32>
    %v4218 = stablehlo.reshape %v1071 : (tensor<32x25088xf32>) -> tensor<32x512x7x7xf32>
    %v4219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4220 = stablehlo.constant dense<1568.0> : tensor<32x512x7x7xf32>
    %v4221 = stablehlo.reduce(%v4218 init: %v4219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4222 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %v4223 = stablehlo.divide %v4222, %v4220 : tensor<32x512x7x7xf32>
    %v4224 = stablehlo.subtract %v4218, %v4223 : tensor<32x512x7x7xf32>
    %v4225 = stablehlo.multiply %v4224, %v4224 : tensor<32x512x7x7xf32>
    %v4226 = stablehlo.reduce(%v4225 init: %v4219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4227 = stablehlo.constant dense<1568.0> : tensor<512xf32>
    %v4228 = stablehlo.divide %v4226, %v4227 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0> : tensor<f32>
    %v4229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4230 = stablehlo.multiply %v4229, %sW : tensor<64x3x7x7xf32>
    %v4231 = stablehlo.add %v4230, %v3631 : tensor<64x3x7x7xf32>
    %v4232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4233 = stablehlo.multiply %v4232, %sWv : tensor<64x3x7x7xf32>
    %v4234 = stablehlo.add %v4233, %v4231 : tensor<64x3x7x7xf32>
    %v4235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4236 = stablehlo.multiply %v4235, %v4234 : tensor<64x3x7x7xf32>
    %v4237 = stablehlo.subtract %sW, %v4236 : tensor<64x3x7x7xf32>
    %v4238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4239 = stablehlo.multiply %v4238, %sg : tensor<64xf32>
    %v4240 = stablehlo.add %v4239, %v3649 : tensor<64xf32>
    %v4241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4242 = stablehlo.multiply %v4241, %sgv : tensor<64xf32>
    %v4243 = stablehlo.add %v4242, %v4240 : tensor<64xf32>
    %v4244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4245 = stablehlo.multiply %v4244, %v4243 : tensor<64xf32>
    %v4246 = stablehlo.subtract %sg, %v4245 : tensor<64xf32>
    %v4247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4248 = stablehlo.multiply %v4247, %sbt : tensor<64xf32>
    %v4249 = stablehlo.add %v4248, %v3652 : tensor<64xf32>
    %v4250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4251 = stablehlo.multiply %v4250, %sbtv : tensor<64xf32>
    %v4252 = stablehlo.add %v4251, %v4249 : tensor<64xf32>
    %v4253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4254 = stablehlo.multiply %v4253, %v4252 : tensor<64xf32>
    %v4255 = stablehlo.subtract %sbt, %v4254 : tensor<64xf32>
    %v4256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4257 = stablehlo.multiply %v4256, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4258 = stablehlo.add %v4257, %v3534 : tensor<64x64x3x3xf32>
    %v4259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4260 = stablehlo.multiply %v4259, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4261 = stablehlo.add %v4260, %v4258 : tensor<64x64x3x3xf32>
    %v4262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4263 = stablehlo.multiply %v4262, %v4261 : tensor<64x64x3x3xf32>
    %v4264 = stablehlo.subtract %s1b0W1, %v4263 : tensor<64x64x3x3xf32>
    %v4265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4266 = stablehlo.multiply %v4265, %s1b0g1 : tensor<64xf32>
    %v4267 = stablehlo.add %v4266, %v3552 : tensor<64xf32>
    %v4268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4269 = stablehlo.multiply %v4268, %s1b0g1v : tensor<64xf32>
    %v4270 = stablehlo.add %v4269, %v4267 : tensor<64xf32>
    %v4271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4272 = stablehlo.multiply %v4271, %v4270 : tensor<64xf32>
    %v4273 = stablehlo.subtract %s1b0g1, %v4272 : tensor<64xf32>
    %v4274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4275 = stablehlo.multiply %v4274, %s1b0bt1 : tensor<64xf32>
    %v4276 = stablehlo.add %v4275, %v3555 : tensor<64xf32>
    %v4277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4278 = stablehlo.multiply %v4277, %s1b0bt1v : tensor<64xf32>
    %v4279 = stablehlo.add %v4278, %v4276 : tensor<64xf32>
    %v4280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4281 = stablehlo.multiply %v4280, %v4279 : tensor<64xf32>
    %v4282 = stablehlo.subtract %s1b0bt1, %v4281 : tensor<64xf32>
    %v4283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4284 = stablehlo.multiply %v4283, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4285 = stablehlo.add %v4284, %v3561 : tensor<64x64x3x3xf32>
    %v4286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4287 = stablehlo.multiply %v4286, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4288 = stablehlo.add %v4287, %v4285 : tensor<64x64x3x3xf32>
    %v4289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4290 = stablehlo.multiply %v4289, %v4288 : tensor<64x64x3x3xf32>
    %v4291 = stablehlo.subtract %s1b0W2, %v4290 : tensor<64x64x3x3xf32>
    %v4292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4293 = stablehlo.multiply %v4292, %s1b0g2 : tensor<64xf32>
    %v4294 = stablehlo.add %v4293, %v3579 : tensor<64xf32>
    %v4295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4296 = stablehlo.multiply %v4295, %s1b0g2v : tensor<64xf32>
    %v4297 = stablehlo.add %v4296, %v4294 : tensor<64xf32>
    %v4298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4299 = stablehlo.multiply %v4298, %v4297 : tensor<64xf32>
    %v4300 = stablehlo.subtract %s1b0g2, %v4299 : tensor<64xf32>
    %v4301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4302 = stablehlo.multiply %v4301, %s1b0bt2 : tensor<64xf32>
    %v4303 = stablehlo.add %v4302, %v3582 : tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4304, %s1b0bt2v : tensor<64xf32>
    %v4306 = stablehlo.add %v4305, %v4303 : tensor<64xf32>
    %v4307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4308 = stablehlo.multiply %v4307, %v4306 : tensor<64xf32>
    %v4309 = stablehlo.subtract %s1b0bt2, %v4308 : tensor<64xf32>
    %v4310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4311 = stablehlo.multiply %v4310, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4312 = stablehlo.add %v4311, %v3394 : tensor<64x64x3x3xf32>
    %v4313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4314 = stablehlo.multiply %v4313, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4315 = stablehlo.add %v4314, %v4312 : tensor<64x64x3x3xf32>
    %v4316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4317 = stablehlo.multiply %v4316, %v4315 : tensor<64x64x3x3xf32>
    %v4318 = stablehlo.subtract %s1b1W1, %v4317 : tensor<64x64x3x3xf32>
    %v4319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4320 = stablehlo.multiply %v4319, %s1b1g1 : tensor<64xf32>
    %v4321 = stablehlo.add %v4320, %v3412 : tensor<64xf32>
    %v4322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4323 = stablehlo.multiply %v4322, %s1b1g1v : tensor<64xf32>
    %v4324 = stablehlo.add %v4323, %v4321 : tensor<64xf32>
    %v4325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.multiply %v4325, %v4324 : tensor<64xf32>
    %v4327 = stablehlo.subtract %s1b1g1, %v4326 : tensor<64xf32>
    %v4328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4329 = stablehlo.multiply %v4328, %s1b1bt1 : tensor<64xf32>
    %v4330 = stablehlo.add %v4329, %v3415 : tensor<64xf32>
    %v4331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.multiply %v4331, %s1b1bt1v : tensor<64xf32>
    %v4333 = stablehlo.add %v4332, %v4330 : tensor<64xf32>
    %v4334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4335 = stablehlo.multiply %v4334, %v4333 : tensor<64xf32>
    %v4336 = stablehlo.subtract %s1b1bt1, %v4335 : tensor<64xf32>
    %v4337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4338 = stablehlo.multiply %v4337, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4339 = stablehlo.add %v4338, %v3421 : tensor<64x64x3x3xf32>
    %v4340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4341 = stablehlo.multiply %v4340, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4342 = stablehlo.add %v4341, %v4339 : tensor<64x64x3x3xf32>
    %v4343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4344 = stablehlo.multiply %v4343, %v4342 : tensor<64x64x3x3xf32>
    %v4345 = stablehlo.subtract %s1b1W2, %v4344 : tensor<64x64x3x3xf32>
    %v4346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4347 = stablehlo.multiply %v4346, %s1b1g2 : tensor<64xf32>
    %v4348 = stablehlo.add %v4347, %v3439 : tensor<64xf32>
    %v4349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4350 = stablehlo.multiply %v4349, %s1b1g2v : tensor<64xf32>
    %v4351 = stablehlo.add %v4350, %v4348 : tensor<64xf32>
    %v4352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4353 = stablehlo.multiply %v4352, %v4351 : tensor<64xf32>
    %v4354 = stablehlo.subtract %s1b1g2, %v4353 : tensor<64xf32>
    %v4355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4356 = stablehlo.multiply %v4355, %s1b1bt2 : tensor<64xf32>
    %v4357 = stablehlo.add %v4356, %v3442 : tensor<64xf32>
    %v4358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.multiply %v4358, %s1b1bt2v : tensor<64xf32>
    %v4360 = stablehlo.add %v4359, %v4357 : tensor<64xf32>
    %v4361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.multiply %v4361, %v4360 : tensor<64xf32>
    %v4363 = stablehlo.subtract %s1b1bt2, %v4362 : tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4365 = stablehlo.multiply %v4364, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4366 = stablehlo.add %v4365, %v3254 : tensor<64x64x3x3xf32>
    %v4367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4368 = stablehlo.multiply %v4367, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4369 = stablehlo.add %v4368, %v4366 : tensor<64x64x3x3xf32>
    %v4370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4371 = stablehlo.multiply %v4370, %v4369 : tensor<64x64x3x3xf32>
    %v4372 = stablehlo.subtract %s1b2W1, %v4371 : tensor<64x64x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4374 = stablehlo.multiply %v4373, %s1b2g1 : tensor<64xf32>
    %v4375 = stablehlo.add %v4374, %v3272 : tensor<64xf32>
    %v4376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4377 = stablehlo.multiply %v4376, %s1b2g1v : tensor<64xf32>
    %v4378 = stablehlo.add %v4377, %v4375 : tensor<64xf32>
    %v4379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4380 = stablehlo.multiply %v4379, %v4378 : tensor<64xf32>
    %v4381 = stablehlo.subtract %s1b2g1, %v4380 : tensor<64xf32>
    %v4382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4383 = stablehlo.multiply %v4382, %s1b2bt1 : tensor<64xf32>
    %v4384 = stablehlo.add %v4383, %v3275 : tensor<64xf32>
    %v4385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4386 = stablehlo.multiply %v4385, %s1b2bt1v : tensor<64xf32>
    %v4387 = stablehlo.add %v4386, %v4384 : tensor<64xf32>
    %v4388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4389 = stablehlo.multiply %v4388, %v4387 : tensor<64xf32>
    %v4390 = stablehlo.subtract %s1b2bt1, %v4389 : tensor<64xf32>
    %v4391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4392 = stablehlo.multiply %v4391, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4393 = stablehlo.add %v4392, %v3281 : tensor<64x64x3x3xf32>
    %v4394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4395 = stablehlo.multiply %v4394, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4396 = stablehlo.add %v4395, %v4393 : tensor<64x64x3x3xf32>
    %v4397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4398 = stablehlo.multiply %v4397, %v4396 : tensor<64x64x3x3xf32>
    %v4399 = stablehlo.subtract %s1b2W2, %v4398 : tensor<64x64x3x3xf32>
    %v4400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4401 = stablehlo.multiply %v4400, %s1b2g2 : tensor<64xf32>
    %v4402 = stablehlo.add %v4401, %v3299 : tensor<64xf32>
    %v4403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4404 = stablehlo.multiply %v4403, %s1b2g2v : tensor<64xf32>
    %v4405 = stablehlo.add %v4404, %v4402 : tensor<64xf32>
    %v4406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4407 = stablehlo.multiply %v4406, %v4405 : tensor<64xf32>
    %v4408 = stablehlo.subtract %s1b2g2, %v4407 : tensor<64xf32>
    %v4409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4410 = stablehlo.multiply %v4409, %s1b2bt2 : tensor<64xf32>
    %v4411 = stablehlo.add %v4410, %v3302 : tensor<64xf32>
    %v4412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4413 = stablehlo.multiply %v4412, %s1b2bt2v : tensor<64xf32>
    %v4414 = stablehlo.add %v4413, %v4411 : tensor<64xf32>
    %v4415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4416 = stablehlo.multiply %v4415, %v4414 : tensor<64xf32>
    %v4417 = stablehlo.subtract %s1b2bt2, %v4416 : tensor<64xf32>
    %v4418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4419 = stablehlo.multiply %v4418, %d2W1 : tensor<128x64x3x3xf32>
    %v4420 = stablehlo.add %v4419, %v3085 : tensor<128x64x3x3xf32>
    %v4421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4422 = stablehlo.multiply %v4421, %d2W1v : tensor<128x64x3x3xf32>
    %v4423 = stablehlo.add %v4422, %v4420 : tensor<128x64x3x3xf32>
    %v4424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4425 = stablehlo.multiply %v4424, %v4423 : tensor<128x64x3x3xf32>
    %v4426 = stablehlo.subtract %d2W1, %v4425 : tensor<128x64x3x3xf32>
    %v4427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4428 = stablehlo.multiply %v4427, %d2g1 : tensor<128xf32>
    %v4429 = stablehlo.add %v4428, %v3103 : tensor<128xf32>
    %v4430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4431 = stablehlo.multiply %v4430, %d2g1v : tensor<128xf32>
    %v4432 = stablehlo.add %v4431, %v4429 : tensor<128xf32>
    %v4433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4434 = stablehlo.multiply %v4433, %v4432 : tensor<128xf32>
    %v4435 = stablehlo.subtract %d2g1, %v4434 : tensor<128xf32>
    %v4436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4437 = stablehlo.multiply %v4436, %d2bt1 : tensor<128xf32>
    %v4438 = stablehlo.add %v4437, %v3106 : tensor<128xf32>
    %v4439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4440 = stablehlo.multiply %v4439, %d2bt1v : tensor<128xf32>
    %v4441 = stablehlo.add %v4440, %v4438 : tensor<128xf32>
    %v4442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4443 = stablehlo.multiply %v4442, %v4441 : tensor<128xf32>
    %v4444 = stablehlo.subtract %d2bt1, %v4443 : tensor<128xf32>
    %v4445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4446 = stablehlo.multiply %v4445, %d2W2 : tensor<128x128x3x3xf32>
    %v4447 = stablehlo.add %v4446, %v3112 : tensor<128x128x3x3xf32>
    %v4448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4449 = stablehlo.multiply %v4448, %d2W2v : tensor<128x128x3x3xf32>
    %v4450 = stablehlo.add %v4449, %v4447 : tensor<128x128x3x3xf32>
    %v4451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4452 = stablehlo.multiply %v4451, %v4450 : tensor<128x128x3x3xf32>
    %v4453 = stablehlo.subtract %d2W2, %v4452 : tensor<128x128x3x3xf32>
    %v4454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4455 = stablehlo.multiply %v4454, %d2g2 : tensor<128xf32>
    %v4456 = stablehlo.add %v4455, %v3130 : tensor<128xf32>
    %v4457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4458 = stablehlo.multiply %v4457, %d2g2v : tensor<128xf32>
    %v4459 = stablehlo.add %v4458, %v4456 : tensor<128xf32>
    %v4460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4461 = stablehlo.multiply %v4460, %v4459 : tensor<128xf32>
    %v4462 = stablehlo.subtract %d2g2, %v4461 : tensor<128xf32>
    %v4463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4464 = stablehlo.multiply %v4463, %d2bt2 : tensor<128xf32>
    %v4465 = stablehlo.add %v4464, %v3133 : tensor<128xf32>
    %v4466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4467 = stablehlo.multiply %v4466, %d2bt2v : tensor<128xf32>
    %v4468 = stablehlo.add %v4467, %v4465 : tensor<128xf32>
    %v4469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4470 = stablehlo.multiply %v4469, %v4468 : tensor<128xf32>
    %v4471 = stablehlo.subtract %d2bt2, %v4470 : tensor<128xf32>
    %v4472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4473 = stablehlo.multiply %v4472, %d2Wp : tensor<128x64x1x1xf32>
    %v4474 = stablehlo.add %v4473, %v3141 : tensor<128x64x1x1xf32>
    %v4475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4476 = stablehlo.multiply %v4475, %d2Wpv : tensor<128x64x1x1xf32>
    %v4477 = stablehlo.add %v4476, %v4474 : tensor<128x64x1x1xf32>
    %v4478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4479 = stablehlo.multiply %v4478, %v4477 : tensor<128x64x1x1xf32>
    %v4480 = stablehlo.subtract %d2Wp, %v4479 : tensor<128x64x1x1xf32>
    %v4481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4482 = stablehlo.multiply %v4481, %d2gp : tensor<128xf32>
    %v4483 = stablehlo.add %v4482, %v3159 : tensor<128xf32>
    %v4484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4485 = stablehlo.multiply %v4484, %d2gpv : tensor<128xf32>
    %v4486 = stablehlo.add %v4485, %v4483 : tensor<128xf32>
    %v4487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4488 = stablehlo.multiply %v4487, %v4486 : tensor<128xf32>
    %v4489 = stablehlo.subtract %d2gp, %v4488 : tensor<128xf32>
    %v4490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4491 = stablehlo.multiply %v4490, %d2btp : tensor<128xf32>
    %v4492 = stablehlo.add %v4491, %v3162 : tensor<128xf32>
    %v4493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4494 = stablehlo.multiply %v4493, %d2btpv : tensor<128xf32>
    %v4495 = stablehlo.add %v4494, %v4492 : tensor<128xf32>
    %v4496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4497 = stablehlo.multiply %v4496, %v4495 : tensor<128xf32>
    %v4498 = stablehlo.subtract %d2btp, %v4497 : tensor<128xf32>
    %v4499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4500 = stablehlo.multiply %v4499, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4501 = stablehlo.add %v4500, %v2904 : tensor<128x128x3x3xf32>
    %v4502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4503 = stablehlo.multiply %v4502, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4504 = stablehlo.add %v4503, %v4501 : tensor<128x128x3x3xf32>
    %v4505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4506 = stablehlo.multiply %v4505, %v4504 : tensor<128x128x3x3xf32>
    %v4507 = stablehlo.subtract %s2b0W1, %v4506 : tensor<128x128x3x3xf32>
    %v4508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4509 = stablehlo.multiply %v4508, %s2b0g1 : tensor<128xf32>
    %v4510 = stablehlo.add %v4509, %v2922 : tensor<128xf32>
    %v4511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4512 = stablehlo.multiply %v4511, %s2b0g1v : tensor<128xf32>
    %v4513 = stablehlo.add %v4512, %v4510 : tensor<128xf32>
    %v4514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4515 = stablehlo.multiply %v4514, %v4513 : tensor<128xf32>
    %v4516 = stablehlo.subtract %s2b0g1, %v4515 : tensor<128xf32>
    %v4517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4518 = stablehlo.multiply %v4517, %s2b0bt1 : tensor<128xf32>
    %v4519 = stablehlo.add %v4518, %v2925 : tensor<128xf32>
    %v4520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4521 = stablehlo.multiply %v4520, %s2b0bt1v : tensor<128xf32>
    %v4522 = stablehlo.add %v4521, %v4519 : tensor<128xf32>
    %v4523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4524 = stablehlo.multiply %v4523, %v4522 : tensor<128xf32>
    %v4525 = stablehlo.subtract %s2b0bt1, %v4524 : tensor<128xf32>
    %v4526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4527 = stablehlo.multiply %v4526, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4528 = stablehlo.add %v4527, %v2931 : tensor<128x128x3x3xf32>
    %v4529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4530 = stablehlo.multiply %v4529, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4531 = stablehlo.add %v4530, %v4528 : tensor<128x128x3x3xf32>
    %v4532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4533 = stablehlo.multiply %v4532, %v4531 : tensor<128x128x3x3xf32>
    %v4534 = stablehlo.subtract %s2b0W2, %v4533 : tensor<128x128x3x3xf32>
    %v4535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4536 = stablehlo.multiply %v4535, %s2b0g2 : tensor<128xf32>
    %v4537 = stablehlo.add %v4536, %v2949 : tensor<128xf32>
    %v4538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4539 = stablehlo.multiply %v4538, %s2b0g2v : tensor<128xf32>
    %v4540 = stablehlo.add %v4539, %v4537 : tensor<128xf32>
    %v4541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4542 = stablehlo.multiply %v4541, %v4540 : tensor<128xf32>
    %v4543 = stablehlo.subtract %s2b0g2, %v4542 : tensor<128xf32>
    %v4544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4545 = stablehlo.multiply %v4544, %s2b0bt2 : tensor<128xf32>
    %v4546 = stablehlo.add %v4545, %v2952 : tensor<128xf32>
    %v4547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4548 = stablehlo.multiply %v4547, %s2b0bt2v : tensor<128xf32>
    %v4549 = stablehlo.add %v4548, %v4546 : tensor<128xf32>
    %v4550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4551 = stablehlo.multiply %v4550, %v4549 : tensor<128xf32>
    %v4552 = stablehlo.subtract %s2b0bt2, %v4551 : tensor<128xf32>
    %v4553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4554 = stablehlo.multiply %v4553, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4555 = stablehlo.add %v4554, %v2764 : tensor<128x128x3x3xf32>
    %v4556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4557 = stablehlo.multiply %v4556, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4558 = stablehlo.add %v4557, %v4555 : tensor<128x128x3x3xf32>
    %v4559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4560 = stablehlo.multiply %v4559, %v4558 : tensor<128x128x3x3xf32>
    %v4561 = stablehlo.subtract %s2b1W1, %v4560 : tensor<128x128x3x3xf32>
    %v4562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4563 = stablehlo.multiply %v4562, %s2b1g1 : tensor<128xf32>
    %v4564 = stablehlo.add %v4563, %v2782 : tensor<128xf32>
    %v4565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4566 = stablehlo.multiply %v4565, %s2b1g1v : tensor<128xf32>
    %v4567 = stablehlo.add %v4566, %v4564 : tensor<128xf32>
    %v4568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4569 = stablehlo.multiply %v4568, %v4567 : tensor<128xf32>
    %v4570 = stablehlo.subtract %s2b1g1, %v4569 : tensor<128xf32>
    %v4571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4572 = stablehlo.multiply %v4571, %s2b1bt1 : tensor<128xf32>
    %v4573 = stablehlo.add %v4572, %v2785 : tensor<128xf32>
    %v4574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4575 = stablehlo.multiply %v4574, %s2b1bt1v : tensor<128xf32>
    %v4576 = stablehlo.add %v4575, %v4573 : tensor<128xf32>
    %v4577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4578 = stablehlo.multiply %v4577, %v4576 : tensor<128xf32>
    %v4579 = stablehlo.subtract %s2b1bt1, %v4578 : tensor<128xf32>
    %v4580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4581 = stablehlo.multiply %v4580, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4582 = stablehlo.add %v4581, %v2791 : tensor<128x128x3x3xf32>
    %v4583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4584 = stablehlo.multiply %v4583, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4585 = stablehlo.add %v4584, %v4582 : tensor<128x128x3x3xf32>
    %v4586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4587 = stablehlo.multiply %v4586, %v4585 : tensor<128x128x3x3xf32>
    %v4588 = stablehlo.subtract %s2b1W2, %v4587 : tensor<128x128x3x3xf32>
    %v4589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4590 = stablehlo.multiply %v4589, %s2b1g2 : tensor<128xf32>
    %v4591 = stablehlo.add %v4590, %v2809 : tensor<128xf32>
    %v4592 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4593 = stablehlo.multiply %v4592, %s2b1g2v : tensor<128xf32>
    %v4594 = stablehlo.add %v4593, %v4591 : tensor<128xf32>
    %v4595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4596 = stablehlo.multiply %v4595, %v4594 : tensor<128xf32>
    %v4597 = stablehlo.subtract %s2b1g2, %v4596 : tensor<128xf32>
    %v4598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4599 = stablehlo.multiply %v4598, %s2b1bt2 : tensor<128xf32>
    %v4600 = stablehlo.add %v4599, %v2812 : tensor<128xf32>
    %v4601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4602 = stablehlo.multiply %v4601, %s2b1bt2v : tensor<128xf32>
    %v4603 = stablehlo.add %v4602, %v4600 : tensor<128xf32>
    %v4604 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4605 = stablehlo.multiply %v4604, %v4603 : tensor<128xf32>
    %v4606 = stablehlo.subtract %s2b1bt2, %v4605 : tensor<128xf32>
    %v4607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4608 = stablehlo.multiply %v4607, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4609 = stablehlo.add %v4608, %v2624 : tensor<128x128x3x3xf32>
    %v4610 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4611 = stablehlo.multiply %v4610, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4612 = stablehlo.add %v4611, %v4609 : tensor<128x128x3x3xf32>
    %v4613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4614 = stablehlo.multiply %v4613, %v4612 : tensor<128x128x3x3xf32>
    %v4615 = stablehlo.subtract %s2b2W1, %v4614 : tensor<128x128x3x3xf32>
    %v4616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4617 = stablehlo.multiply %v4616, %s2b2g1 : tensor<128xf32>
    %v4618 = stablehlo.add %v4617, %v2642 : tensor<128xf32>
    %v4619 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4620 = stablehlo.multiply %v4619, %s2b2g1v : tensor<128xf32>
    %v4621 = stablehlo.add %v4620, %v4618 : tensor<128xf32>
    %v4622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4623 = stablehlo.multiply %v4622, %v4621 : tensor<128xf32>
    %v4624 = stablehlo.subtract %s2b2g1, %v4623 : tensor<128xf32>
    %v4625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4626 = stablehlo.multiply %v4625, %s2b2bt1 : tensor<128xf32>
    %v4627 = stablehlo.add %v4626, %v2645 : tensor<128xf32>
    %v4628 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4629 = stablehlo.multiply %v4628, %s2b2bt1v : tensor<128xf32>
    %v4630 = stablehlo.add %v4629, %v4627 : tensor<128xf32>
    %v4631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4632 = stablehlo.multiply %v4631, %v4630 : tensor<128xf32>
    %v4633 = stablehlo.subtract %s2b2bt1, %v4632 : tensor<128xf32>
    %v4634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4635 = stablehlo.multiply %v4634, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4636 = stablehlo.add %v4635, %v2651 : tensor<128x128x3x3xf32>
    %v4637 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4638 = stablehlo.multiply %v4637, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4639 = stablehlo.add %v4638, %v4636 : tensor<128x128x3x3xf32>
    %v4640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4641 = stablehlo.multiply %v4640, %v4639 : tensor<128x128x3x3xf32>
    %v4642 = stablehlo.subtract %s2b2W2, %v4641 : tensor<128x128x3x3xf32>
    %v4643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4644 = stablehlo.multiply %v4643, %s2b2g2 : tensor<128xf32>
    %v4645 = stablehlo.add %v4644, %v2669 : tensor<128xf32>
    %v4646 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4647 = stablehlo.multiply %v4646, %s2b2g2v : tensor<128xf32>
    %v4648 = stablehlo.add %v4647, %v4645 : tensor<128xf32>
    %v4649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4650 = stablehlo.multiply %v4649, %v4648 : tensor<128xf32>
    %v4651 = stablehlo.subtract %s2b2g2, %v4650 : tensor<128xf32>
    %v4652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4653 = stablehlo.multiply %v4652, %s2b2bt2 : tensor<128xf32>
    %v4654 = stablehlo.add %v4653, %v2672 : tensor<128xf32>
    %v4655 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4656 = stablehlo.multiply %v4655, %s2b2bt2v : tensor<128xf32>
    %v4657 = stablehlo.add %v4656, %v4654 : tensor<128xf32>
    %v4658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4659 = stablehlo.multiply %v4658, %v4657 : tensor<128xf32>
    %v4660 = stablehlo.subtract %s2b2bt2, %v4659 : tensor<128xf32>
    %v4661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4662 = stablehlo.multiply %v4661, %d3W1 : tensor<256x128x3x3xf32>
    %v4663 = stablehlo.add %v4662, %v2455 : tensor<256x128x3x3xf32>
    %v4664 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4665 = stablehlo.multiply %v4664, %d3W1v : tensor<256x128x3x3xf32>
    %v4666 = stablehlo.add %v4665, %v4663 : tensor<256x128x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4668 = stablehlo.multiply %v4667, %v4666 : tensor<256x128x3x3xf32>
    %v4669 = stablehlo.subtract %d3W1, %v4668 : tensor<256x128x3x3xf32>
    %v4670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4671 = stablehlo.multiply %v4670, %d3g1 : tensor<256xf32>
    %v4672 = stablehlo.add %v4671, %v2473 : tensor<256xf32>
    %v4673 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4674 = stablehlo.multiply %v4673, %d3g1v : tensor<256xf32>
    %v4675 = stablehlo.add %v4674, %v4672 : tensor<256xf32>
    %v4676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4677 = stablehlo.multiply %v4676, %v4675 : tensor<256xf32>
    %v4678 = stablehlo.subtract %d3g1, %v4677 : tensor<256xf32>
    %v4679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4680 = stablehlo.multiply %v4679, %d3bt1 : tensor<256xf32>
    %v4681 = stablehlo.add %v4680, %v2476 : tensor<256xf32>
    %v4682 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4683 = stablehlo.multiply %v4682, %d3bt1v : tensor<256xf32>
    %v4684 = stablehlo.add %v4683, %v4681 : tensor<256xf32>
    %v4685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4686 = stablehlo.multiply %v4685, %v4684 : tensor<256xf32>
    %v4687 = stablehlo.subtract %d3bt1, %v4686 : tensor<256xf32>
    %v4688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4689 = stablehlo.multiply %v4688, %d3W2 : tensor<256x256x3x3xf32>
    %v4690 = stablehlo.add %v4689, %v2482 : tensor<256x256x3x3xf32>
    %v4691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4692 = stablehlo.multiply %v4691, %d3W2v : tensor<256x256x3x3xf32>
    %v4693 = stablehlo.add %v4692, %v4690 : tensor<256x256x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %v4693 : tensor<256x256x3x3xf32>
    %v4696 = stablehlo.subtract %d3W2, %v4695 : tensor<256x256x3x3xf32>
    %v4697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4698 = stablehlo.multiply %v4697, %d3g2 : tensor<256xf32>
    %v4699 = stablehlo.add %v4698, %v2500 : tensor<256xf32>
    %v4700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4701 = stablehlo.multiply %v4700, %d3g2v : tensor<256xf32>
    %v4702 = stablehlo.add %v4701, %v4699 : tensor<256xf32>
    %v4703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4704 = stablehlo.multiply %v4703, %v4702 : tensor<256xf32>
    %v4705 = stablehlo.subtract %d3g2, %v4704 : tensor<256xf32>
    %v4706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4707 = stablehlo.multiply %v4706, %d3bt2 : tensor<256xf32>
    %v4708 = stablehlo.add %v4707, %v2503 : tensor<256xf32>
    %v4709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4710 = stablehlo.multiply %v4709, %d3bt2v : tensor<256xf32>
    %v4711 = stablehlo.add %v4710, %v4708 : tensor<256xf32>
    %v4712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4713 = stablehlo.multiply %v4712, %v4711 : tensor<256xf32>
    %v4714 = stablehlo.subtract %d3bt2, %v4713 : tensor<256xf32>
    %v4715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4716 = stablehlo.multiply %v4715, %d3Wp : tensor<256x128x1x1xf32>
    %v4717 = stablehlo.add %v4716, %v2511 : tensor<256x128x1x1xf32>
    %v4718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4719 = stablehlo.multiply %v4718, %d3Wpv : tensor<256x128x1x1xf32>
    %v4720 = stablehlo.add %v4719, %v4717 : tensor<256x128x1x1xf32>
    %v4721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4722 = stablehlo.multiply %v4721, %v4720 : tensor<256x128x1x1xf32>
    %v4723 = stablehlo.subtract %d3Wp, %v4722 : tensor<256x128x1x1xf32>
    %v4724 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4725 = stablehlo.multiply %v4724, %d3gp : tensor<256xf32>
    %v4726 = stablehlo.add %v4725, %v2529 : tensor<256xf32>
    %v4727 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4728 = stablehlo.multiply %v4727, %d3gpv : tensor<256xf32>
    %v4729 = stablehlo.add %v4728, %v4726 : tensor<256xf32>
    %v4730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4731 = stablehlo.multiply %v4730, %v4729 : tensor<256xf32>
    %v4732 = stablehlo.subtract %d3gp, %v4731 : tensor<256xf32>
    %v4733 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4734 = stablehlo.multiply %v4733, %d3btp : tensor<256xf32>
    %v4735 = stablehlo.add %v4734, %v2532 : tensor<256xf32>
    %v4736 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.multiply %v4736, %d3btpv : tensor<256xf32>
    %v4738 = stablehlo.add %v4737, %v4735 : tensor<256xf32>
    %v4739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4740 = stablehlo.multiply %v4739, %v4738 : tensor<256xf32>
    %v4741 = stablehlo.subtract %d3btp, %v4740 : tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4743 = stablehlo.multiply %v4742, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4744 = stablehlo.add %v4743, %v2274 : tensor<256x256x3x3xf32>
    %v4745 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4746 = stablehlo.multiply %v4745, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4747 = stablehlo.add %v4746, %v4744 : tensor<256x256x3x3xf32>
    %v4748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4749 = stablehlo.multiply %v4748, %v4747 : tensor<256x256x3x3xf32>
    %v4750 = stablehlo.subtract %s3b0W1, %v4749 : tensor<256x256x3x3xf32>
    %v4751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4752 = stablehlo.multiply %v4751, %s3b0g1 : tensor<256xf32>
    %v4753 = stablehlo.add %v4752, %v2292 : tensor<256xf32>
    %v4754 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4755 = stablehlo.multiply %v4754, %s3b0g1v : tensor<256xf32>
    %v4756 = stablehlo.add %v4755, %v4753 : tensor<256xf32>
    %v4757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4758 = stablehlo.multiply %v4757, %v4756 : tensor<256xf32>
    %v4759 = stablehlo.subtract %s3b0g1, %v4758 : tensor<256xf32>
    %v4760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4761 = stablehlo.multiply %v4760, %s3b0bt1 : tensor<256xf32>
    %v4762 = stablehlo.add %v4761, %v2295 : tensor<256xf32>
    %v4763 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4764 = stablehlo.multiply %v4763, %s3b0bt1v : tensor<256xf32>
    %v4765 = stablehlo.add %v4764, %v4762 : tensor<256xf32>
    %v4766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4767 = stablehlo.multiply %v4766, %v4765 : tensor<256xf32>
    %v4768 = stablehlo.subtract %s3b0bt1, %v4767 : tensor<256xf32>
    %v4769 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4770 = stablehlo.multiply %v4769, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4771 = stablehlo.add %v4770, %v2301 : tensor<256x256x3x3xf32>
    %v4772 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4773 = stablehlo.multiply %v4772, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4774 = stablehlo.add %v4773, %v4771 : tensor<256x256x3x3xf32>
    %v4775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4776 = stablehlo.multiply %v4775, %v4774 : tensor<256x256x3x3xf32>
    %v4777 = stablehlo.subtract %s3b0W2, %v4776 : tensor<256x256x3x3xf32>
    %v4778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4779 = stablehlo.multiply %v4778, %s3b0g2 : tensor<256xf32>
    %v4780 = stablehlo.add %v4779, %v2319 : tensor<256xf32>
    %v4781 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4782 = stablehlo.multiply %v4781, %s3b0g2v : tensor<256xf32>
    %v4783 = stablehlo.add %v4782, %v4780 : tensor<256xf32>
    %v4784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4785 = stablehlo.multiply %v4784, %v4783 : tensor<256xf32>
    %v4786 = stablehlo.subtract %s3b0g2, %v4785 : tensor<256xf32>
    %v4787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4788 = stablehlo.multiply %v4787, %s3b0bt2 : tensor<256xf32>
    %v4789 = stablehlo.add %v4788, %v2322 : tensor<256xf32>
    %v4790 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4791 = stablehlo.multiply %v4790, %s3b0bt2v : tensor<256xf32>
    %v4792 = stablehlo.add %v4791, %v4789 : tensor<256xf32>
    %v4793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4794 = stablehlo.multiply %v4793, %v4792 : tensor<256xf32>
    %v4795 = stablehlo.subtract %s3b0bt2, %v4794 : tensor<256xf32>
    %v4796 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4797 = stablehlo.multiply %v4796, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4798 = stablehlo.add %v4797, %v2134 : tensor<256x256x3x3xf32>
    %v4799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4800 = stablehlo.multiply %v4799, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4801 = stablehlo.add %v4800, %v4798 : tensor<256x256x3x3xf32>
    %v4802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4803 = stablehlo.multiply %v4802, %v4801 : tensor<256x256x3x3xf32>
    %v4804 = stablehlo.subtract %s3b1W1, %v4803 : tensor<256x256x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4806 = stablehlo.multiply %v4805, %s3b1g1 : tensor<256xf32>
    %v4807 = stablehlo.add %v4806, %v2152 : tensor<256xf32>
    %v4808 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4809 = stablehlo.multiply %v4808, %s3b1g1v : tensor<256xf32>
    %v4810 = stablehlo.add %v4809, %v4807 : tensor<256xf32>
    %v4811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4812 = stablehlo.multiply %v4811, %v4810 : tensor<256xf32>
    %v4813 = stablehlo.subtract %s3b1g1, %v4812 : tensor<256xf32>
    %v4814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4815 = stablehlo.multiply %v4814, %s3b1bt1 : tensor<256xf32>
    %v4816 = stablehlo.add %v4815, %v2155 : tensor<256xf32>
    %v4817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4818 = stablehlo.multiply %v4817, %s3b1bt1v : tensor<256xf32>
    %v4819 = stablehlo.add %v4818, %v4816 : tensor<256xf32>
    %v4820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4821 = stablehlo.multiply %v4820, %v4819 : tensor<256xf32>
    %v4822 = stablehlo.subtract %s3b1bt1, %v4821 : tensor<256xf32>
    %v4823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4824 = stablehlo.multiply %v4823, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4825 = stablehlo.add %v4824, %v2161 : tensor<256x256x3x3xf32>
    %v4826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4827 = stablehlo.multiply %v4826, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4828 = stablehlo.add %v4827, %v4825 : tensor<256x256x3x3xf32>
    %v4829 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4830 = stablehlo.multiply %v4829, %v4828 : tensor<256x256x3x3xf32>
    %v4831 = stablehlo.subtract %s3b1W2, %v4830 : tensor<256x256x3x3xf32>
    %v4832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4833 = stablehlo.multiply %v4832, %s3b1g2 : tensor<256xf32>
    %v4834 = stablehlo.add %v4833, %v2179 : tensor<256xf32>
    %v4835 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4836 = stablehlo.multiply %v4835, %s3b1g2v : tensor<256xf32>
    %v4837 = stablehlo.add %v4836, %v4834 : tensor<256xf32>
    %v4838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4839 = stablehlo.multiply %v4838, %v4837 : tensor<256xf32>
    %v4840 = stablehlo.subtract %s3b1g2, %v4839 : tensor<256xf32>
    %v4841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4842 = stablehlo.multiply %v4841, %s3b1bt2 : tensor<256xf32>
    %v4843 = stablehlo.add %v4842, %v2182 : tensor<256xf32>
    %v4844 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4845 = stablehlo.multiply %v4844, %s3b1bt2v : tensor<256xf32>
    %v4846 = stablehlo.add %v4845, %v4843 : tensor<256xf32>
    %v4847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4848 = stablehlo.multiply %v4847, %v4846 : tensor<256xf32>
    %v4849 = stablehlo.subtract %s3b1bt2, %v4848 : tensor<256xf32>
    %v4850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4851 = stablehlo.multiply %v4850, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4852 = stablehlo.add %v4851, %v1994 : tensor<256x256x3x3xf32>
    %v4853 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4854 = stablehlo.multiply %v4853, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4855 = stablehlo.add %v4854, %v4852 : tensor<256x256x3x3xf32>
    %v4856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4857 = stablehlo.multiply %v4856, %v4855 : tensor<256x256x3x3xf32>
    %v4858 = stablehlo.subtract %s3b2W1, %v4857 : tensor<256x256x3x3xf32>
    %v4859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4860 = stablehlo.multiply %v4859, %s3b2g1 : tensor<256xf32>
    %v4861 = stablehlo.add %v4860, %v2012 : tensor<256xf32>
    %v4862 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4863 = stablehlo.multiply %v4862, %s3b2g1v : tensor<256xf32>
    %v4864 = stablehlo.add %v4863, %v4861 : tensor<256xf32>
    %v4865 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4866 = stablehlo.multiply %v4865, %v4864 : tensor<256xf32>
    %v4867 = stablehlo.subtract %s3b2g1, %v4866 : tensor<256xf32>
    %v4868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4869 = stablehlo.multiply %v4868, %s3b2bt1 : tensor<256xf32>
    %v4870 = stablehlo.add %v4869, %v2015 : tensor<256xf32>
    %v4871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4872 = stablehlo.multiply %v4871, %s3b2bt1v : tensor<256xf32>
    %v4873 = stablehlo.add %v4872, %v4870 : tensor<256xf32>
    %v4874 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4875 = stablehlo.multiply %v4874, %v4873 : tensor<256xf32>
    %v4876 = stablehlo.subtract %s3b2bt1, %v4875 : tensor<256xf32>
    %v4877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4878 = stablehlo.multiply %v4877, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4879 = stablehlo.add %v4878, %v2021 : tensor<256x256x3x3xf32>
    %v4880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4881 = stablehlo.multiply %v4880, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4882 = stablehlo.add %v4881, %v4879 : tensor<256x256x3x3xf32>
    %v4883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4884 = stablehlo.multiply %v4883, %v4882 : tensor<256x256x3x3xf32>
    %v4885 = stablehlo.subtract %s3b2W2, %v4884 : tensor<256x256x3x3xf32>
    %v4886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4887 = stablehlo.multiply %v4886, %s3b2g2 : tensor<256xf32>
    %v4888 = stablehlo.add %v4887, %v2039 : tensor<256xf32>
    %v4889 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4890 = stablehlo.multiply %v4889, %s3b2g2v : tensor<256xf32>
    %v4891 = stablehlo.add %v4890, %v4888 : tensor<256xf32>
    %v4892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4893 = stablehlo.multiply %v4892, %v4891 : tensor<256xf32>
    %v4894 = stablehlo.subtract %s3b2g2, %v4893 : tensor<256xf32>
    %v4895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4896 = stablehlo.multiply %v4895, %s3b2bt2 : tensor<256xf32>
    %v4897 = stablehlo.add %v4896, %v2042 : tensor<256xf32>
    %v4898 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4899 = stablehlo.multiply %v4898, %s3b2bt2v : tensor<256xf32>
    %v4900 = stablehlo.add %v4899, %v4897 : tensor<256xf32>
    %v4901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4902 = stablehlo.multiply %v4901, %v4900 : tensor<256xf32>
    %v4903 = stablehlo.subtract %s3b2bt2, %v4902 : tensor<256xf32>
    %v4904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4905 = stablehlo.multiply %v4904, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4906 = stablehlo.add %v4905, %v1854 : tensor<256x256x3x3xf32>
    %v4907 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4908 = stablehlo.multiply %v4907, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4909 = stablehlo.add %v4908, %v4906 : tensor<256x256x3x3xf32>
    %v4910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4911 = stablehlo.multiply %v4910, %v4909 : tensor<256x256x3x3xf32>
    %v4912 = stablehlo.subtract %s3b3W1, %v4911 : tensor<256x256x3x3xf32>
    %v4913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4914 = stablehlo.multiply %v4913, %s3b3g1 : tensor<256xf32>
    %v4915 = stablehlo.add %v4914, %v1872 : tensor<256xf32>
    %v4916 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4917 = stablehlo.multiply %v4916, %s3b3g1v : tensor<256xf32>
    %v4918 = stablehlo.add %v4917, %v4915 : tensor<256xf32>
    %v4919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4920 = stablehlo.multiply %v4919, %v4918 : tensor<256xf32>
    %v4921 = stablehlo.subtract %s3b3g1, %v4920 : tensor<256xf32>
    %v4922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4923 = stablehlo.multiply %v4922, %s3b3bt1 : tensor<256xf32>
    %v4924 = stablehlo.add %v4923, %v1875 : tensor<256xf32>
    %v4925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4926 = stablehlo.multiply %v4925, %s3b3bt1v : tensor<256xf32>
    %v4927 = stablehlo.add %v4926, %v4924 : tensor<256xf32>
    %v4928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4929 = stablehlo.multiply %v4928, %v4927 : tensor<256xf32>
    %v4930 = stablehlo.subtract %s3b3bt1, %v4929 : tensor<256xf32>
    %v4931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4932 = stablehlo.multiply %v4931, %s3b3W2 : tensor<256x256x3x3xf32>
    %v4933 = stablehlo.add %v4932, %v1881 : tensor<256x256x3x3xf32>
    %v4934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4935 = stablehlo.multiply %v4934, %s3b3W2v : tensor<256x256x3x3xf32>
    %v4936 = stablehlo.add %v4935, %v4933 : tensor<256x256x3x3xf32>
    %v4937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4938 = stablehlo.multiply %v4937, %v4936 : tensor<256x256x3x3xf32>
    %v4939 = stablehlo.subtract %s3b3W2, %v4938 : tensor<256x256x3x3xf32>
    %v4940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4941 = stablehlo.multiply %v4940, %s3b3g2 : tensor<256xf32>
    %v4942 = stablehlo.add %v4941, %v1899 : tensor<256xf32>
    %v4943 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4944 = stablehlo.multiply %v4943, %s3b3g2v : tensor<256xf32>
    %v4945 = stablehlo.add %v4944, %v4942 : tensor<256xf32>
    %v4946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4947 = stablehlo.multiply %v4946, %v4945 : tensor<256xf32>
    %v4948 = stablehlo.subtract %s3b3g2, %v4947 : tensor<256xf32>
    %v4949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4950 = stablehlo.multiply %v4949, %s3b3bt2 : tensor<256xf32>
    %v4951 = stablehlo.add %v4950, %v1902 : tensor<256xf32>
    %v4952 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4953 = stablehlo.multiply %v4952, %s3b3bt2v : tensor<256xf32>
    %v4954 = stablehlo.add %v4953, %v4951 : tensor<256xf32>
    %v4955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4956 = stablehlo.multiply %v4955, %v4954 : tensor<256xf32>
    %v4957 = stablehlo.subtract %s3b3bt2, %v4956 : tensor<256xf32>
    %v4958 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4959 = stablehlo.multiply %v4958, %s3b4W1 : tensor<256x256x3x3xf32>
    %v4960 = stablehlo.add %v4959, %v1714 : tensor<256x256x3x3xf32>
    %v4961 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4962 = stablehlo.multiply %v4961, %s3b4W1v : tensor<256x256x3x3xf32>
    %v4963 = stablehlo.add %v4962, %v4960 : tensor<256x256x3x3xf32>
    %v4964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4965 = stablehlo.multiply %v4964, %v4963 : tensor<256x256x3x3xf32>
    %v4966 = stablehlo.subtract %s3b4W1, %v4965 : tensor<256x256x3x3xf32>
    %v4967 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4968 = stablehlo.multiply %v4967, %s3b4g1 : tensor<256xf32>
    %v4969 = stablehlo.add %v4968, %v1732 : tensor<256xf32>
    %v4970 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4971 = stablehlo.multiply %v4970, %s3b4g1v : tensor<256xf32>
    %v4972 = stablehlo.add %v4971, %v4969 : tensor<256xf32>
    %v4973 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4974 = stablehlo.multiply %v4973, %v4972 : tensor<256xf32>
    %v4975 = stablehlo.subtract %s3b4g1, %v4974 : tensor<256xf32>
    %v4976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4977 = stablehlo.multiply %v4976, %s3b4bt1 : tensor<256xf32>
    %v4978 = stablehlo.add %v4977, %v1735 : tensor<256xf32>
    %v4979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4980 = stablehlo.multiply %v4979, %s3b4bt1v : tensor<256xf32>
    %v4981 = stablehlo.add %v4980, %v4978 : tensor<256xf32>
    %v4982 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4983 = stablehlo.multiply %v4982, %v4981 : tensor<256xf32>
    %v4984 = stablehlo.subtract %s3b4bt1, %v4983 : tensor<256xf32>
    %v4985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4986 = stablehlo.multiply %v4985, %s3b4W2 : tensor<256x256x3x3xf32>
    %v4987 = stablehlo.add %v4986, %v1741 : tensor<256x256x3x3xf32>
    %v4988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4989 = stablehlo.multiply %v4988, %s3b4W2v : tensor<256x256x3x3xf32>
    %v4990 = stablehlo.add %v4989, %v4987 : tensor<256x256x3x3xf32>
    %v4991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4992 = stablehlo.multiply %v4991, %v4990 : tensor<256x256x3x3xf32>
    %v4993 = stablehlo.subtract %s3b4W2, %v4992 : tensor<256x256x3x3xf32>
    %v4994 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4995 = stablehlo.multiply %v4994, %s3b4g2 : tensor<256xf32>
    %v4996 = stablehlo.add %v4995, %v1759 : tensor<256xf32>
    %v4997 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4998 = stablehlo.multiply %v4997, %s3b4g2v : tensor<256xf32>
    %v4999 = stablehlo.add %v4998, %v4996 : tensor<256xf32>
    %v5000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5001 = stablehlo.multiply %v5000, %v4999 : tensor<256xf32>
    %v5002 = stablehlo.subtract %s3b4g2, %v5001 : tensor<256xf32>
    %v5003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5004 = stablehlo.multiply %v5003, %s3b4bt2 : tensor<256xf32>
    %v5005 = stablehlo.add %v5004, %v1762 : tensor<256xf32>
    %v5006 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5007 = stablehlo.multiply %v5006, %s3b4bt2v : tensor<256xf32>
    %v5008 = stablehlo.add %v5007, %v5005 : tensor<256xf32>
    %v5009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5010 = stablehlo.multiply %v5009, %v5008 : tensor<256xf32>
    %v5011 = stablehlo.subtract %s3b4bt2, %v5010 : tensor<256xf32>
    %v5012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5013 = stablehlo.multiply %v5012, %d4W1 : tensor<512x256x3x3xf32>
    %v5014 = stablehlo.add %v5013, %v1545 : tensor<512x256x3x3xf32>
    %v5015 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5016 = stablehlo.multiply %v5015, %d4W1v : tensor<512x256x3x3xf32>
    %v5017 = stablehlo.add %v5016, %v5014 : tensor<512x256x3x3xf32>
    %v5018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %v5017 : tensor<512x256x3x3xf32>
    %v5020 = stablehlo.subtract %d4W1, %v5019 : tensor<512x256x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5022 = stablehlo.multiply %v5021, %d4g1 : tensor<512xf32>
    %v5023 = stablehlo.add %v5022, %v1563 : tensor<512xf32>
    %v5024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5025 = stablehlo.multiply %v5024, %d4g1v : tensor<512xf32>
    %v5026 = stablehlo.add %v5025, %v5023 : tensor<512xf32>
    %v5027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5028 = stablehlo.multiply %v5027, %v5026 : tensor<512xf32>
    %v5029 = stablehlo.subtract %d4g1, %v5028 : tensor<512xf32>
    %v5030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5031 = stablehlo.multiply %v5030, %d4bt1 : tensor<512xf32>
    %v5032 = stablehlo.add %v5031, %v1566 : tensor<512xf32>
    %v5033 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5034 = stablehlo.multiply %v5033, %d4bt1v : tensor<512xf32>
    %v5035 = stablehlo.add %v5034, %v5032 : tensor<512xf32>
    %v5036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5037 = stablehlo.multiply %v5036, %v5035 : tensor<512xf32>
    %v5038 = stablehlo.subtract %d4bt1, %v5037 : tensor<512xf32>
    %v5039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5040 = stablehlo.multiply %v5039, %d4W2 : tensor<512x512x3x3xf32>
    %v5041 = stablehlo.add %v5040, %v1572 : tensor<512x512x3x3xf32>
    %v5042 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5043 = stablehlo.multiply %v5042, %d4W2v : tensor<512x512x3x3xf32>
    %v5044 = stablehlo.add %v5043, %v5041 : tensor<512x512x3x3xf32>
    %v5045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5046 = stablehlo.multiply %v5045, %v5044 : tensor<512x512x3x3xf32>
    %v5047 = stablehlo.subtract %d4W2, %v5046 : tensor<512x512x3x3xf32>
    %v5048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5049 = stablehlo.multiply %v5048, %d4g2 : tensor<512xf32>
    %v5050 = stablehlo.add %v5049, %v1590 : tensor<512xf32>
    %v5051 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5052 = stablehlo.multiply %v5051, %d4g2v : tensor<512xf32>
    %v5053 = stablehlo.add %v5052, %v5050 : tensor<512xf32>
    %v5054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5055 = stablehlo.multiply %v5054, %v5053 : tensor<512xf32>
    %v5056 = stablehlo.subtract %d4g2, %v5055 : tensor<512xf32>
    %v5057 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5058 = stablehlo.multiply %v5057, %d4bt2 : tensor<512xf32>
    %v5059 = stablehlo.add %v5058, %v1593 : tensor<512xf32>
    %v5060 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5061 = stablehlo.multiply %v5060, %d4bt2v : tensor<512xf32>
    %v5062 = stablehlo.add %v5061, %v5059 : tensor<512xf32>
    %v5063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5064 = stablehlo.multiply %v5063, %v5062 : tensor<512xf32>
    %v5065 = stablehlo.subtract %d4bt2, %v5064 : tensor<512xf32>
    %v5066 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5067 = stablehlo.multiply %v5066, %d4Wp : tensor<512x256x1x1xf32>
    %v5068 = stablehlo.add %v5067, %v1601 : tensor<512x256x1x1xf32>
    %v5069 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5070 = stablehlo.multiply %v5069, %d4Wpv : tensor<512x256x1x1xf32>
    %v5071 = stablehlo.add %v5070, %v5068 : tensor<512x256x1x1xf32>
    %v5072 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5073 = stablehlo.multiply %v5072, %v5071 : tensor<512x256x1x1xf32>
    %v5074 = stablehlo.subtract %d4Wp, %v5073 : tensor<512x256x1x1xf32>
    %v5075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5076 = stablehlo.multiply %v5075, %d4gp : tensor<512xf32>
    %v5077 = stablehlo.add %v5076, %v1619 : tensor<512xf32>
    %v5078 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5079 = stablehlo.multiply %v5078, %d4gpv : tensor<512xf32>
    %v5080 = stablehlo.add %v5079, %v5077 : tensor<512xf32>
    %v5081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5082 = stablehlo.multiply %v5081, %v5080 : tensor<512xf32>
    %v5083 = stablehlo.subtract %d4gp, %v5082 : tensor<512xf32>
    %v5084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5085 = stablehlo.multiply %v5084, %d4btp : tensor<512xf32>
    %v5086 = stablehlo.add %v5085, %v1622 : tensor<512xf32>
    %v5087 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5088 = stablehlo.multiply %v5087, %d4btpv : tensor<512xf32>
    %v5089 = stablehlo.add %v5088, %v5086 : tensor<512xf32>
    %v5090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5091 = stablehlo.multiply %v5090, %v5089 : tensor<512xf32>
    %v5092 = stablehlo.subtract %d4btp, %v5091 : tensor<512xf32>
    %v5093 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5094 = stablehlo.multiply %v5093, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5095 = stablehlo.add %v5094, %v1364 : tensor<512x512x3x3xf32>
    %v5096 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5097 = stablehlo.multiply %v5096, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5098 = stablehlo.add %v5097, %v5095 : tensor<512x512x3x3xf32>
    %v5099 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5100 = stablehlo.multiply %v5099, %v5098 : tensor<512x512x3x3xf32>
    %v5101 = stablehlo.subtract %s4b0W1, %v5100 : tensor<512x512x3x3xf32>
    %v5102 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5103 = stablehlo.multiply %v5102, %s4b0g1 : tensor<512xf32>
    %v5104 = stablehlo.add %v5103, %v1382 : tensor<512xf32>
    %v5105 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5106 = stablehlo.multiply %v5105, %s4b0g1v : tensor<512xf32>
    %v5107 = stablehlo.add %v5106, %v5104 : tensor<512xf32>
    %v5108 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5109 = stablehlo.multiply %v5108, %v5107 : tensor<512xf32>
    %v5110 = stablehlo.subtract %s4b0g1, %v5109 : tensor<512xf32>
    %v5111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5112 = stablehlo.multiply %v5111, %s4b0bt1 : tensor<512xf32>
    %v5113 = stablehlo.add %v5112, %v1385 : tensor<512xf32>
    %v5114 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5115 = stablehlo.multiply %v5114, %s4b0bt1v : tensor<512xf32>
    %v5116 = stablehlo.add %v5115, %v5113 : tensor<512xf32>
    %v5117 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5118 = stablehlo.multiply %v5117, %v5116 : tensor<512xf32>
    %v5119 = stablehlo.subtract %s4b0bt1, %v5118 : tensor<512xf32>
    %v5120 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5121 = stablehlo.multiply %v5120, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5122 = stablehlo.add %v5121, %v1391 : tensor<512x512x3x3xf32>
    %v5123 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5124 = stablehlo.multiply %v5123, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5125 = stablehlo.add %v5124, %v5122 : tensor<512x512x3x3xf32>
    %v5126 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5127 = stablehlo.multiply %v5126, %v5125 : tensor<512x512x3x3xf32>
    %v5128 = stablehlo.subtract %s4b0W2, %v5127 : tensor<512x512x3x3xf32>
    %v5129 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5130 = stablehlo.multiply %v5129, %s4b0g2 : tensor<512xf32>
    %v5131 = stablehlo.add %v5130, %v1409 : tensor<512xf32>
    %v5132 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5133 = stablehlo.multiply %v5132, %s4b0g2v : tensor<512xf32>
    %v5134 = stablehlo.add %v5133, %v5131 : tensor<512xf32>
    %v5135 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5136 = stablehlo.multiply %v5135, %v5134 : tensor<512xf32>
    %v5137 = stablehlo.subtract %s4b0g2, %v5136 : tensor<512xf32>
    %v5138 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5139 = stablehlo.multiply %v5138, %s4b0bt2 : tensor<512xf32>
    %v5140 = stablehlo.add %v5139, %v1412 : tensor<512xf32>
    %v5141 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5142 = stablehlo.multiply %v5141, %s4b0bt2v : tensor<512xf32>
    %v5143 = stablehlo.add %v5142, %v5140 : tensor<512xf32>
    %v5144 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5145 = stablehlo.multiply %v5144, %v5143 : tensor<512xf32>
    %v5146 = stablehlo.subtract %s4b0bt2, %v5145 : tensor<512xf32>
    %v5147 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5148 = stablehlo.multiply %v5147, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5149 = stablehlo.add %v5148, %v1224 : tensor<512x512x3x3xf32>
    %v5150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5151 = stablehlo.multiply %v5150, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5152 = stablehlo.add %v5151, %v5149 : tensor<512x512x3x3xf32>
    %v5153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5154 = stablehlo.multiply %v5153, %v5152 : tensor<512x512x3x3xf32>
    %v5155 = stablehlo.subtract %s4b1W1, %v5154 : tensor<512x512x3x3xf32>
    %v5156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5157 = stablehlo.multiply %v5156, %s4b1g1 : tensor<512xf32>
    %v5158 = stablehlo.add %v5157, %v1242 : tensor<512xf32>
    %v5159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5160 = stablehlo.multiply %v5159, %s4b1g1v : tensor<512xf32>
    %v5161 = stablehlo.add %v5160, %v5158 : tensor<512xf32>
    %v5162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5163 = stablehlo.multiply %v5162, %v5161 : tensor<512xf32>
    %v5164 = stablehlo.subtract %s4b1g1, %v5163 : tensor<512xf32>
    %v5165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5166 = stablehlo.multiply %v5165, %s4b1bt1 : tensor<512xf32>
    %v5167 = stablehlo.add %v5166, %v1245 : tensor<512xf32>
    %v5168 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5169 = stablehlo.multiply %v5168, %s4b1bt1v : tensor<512xf32>
    %v5170 = stablehlo.add %v5169, %v5167 : tensor<512xf32>
    %v5171 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5172 = stablehlo.multiply %v5171, %v5170 : tensor<512xf32>
    %v5173 = stablehlo.subtract %s4b1bt1, %v5172 : tensor<512xf32>
    %v5174 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5175 = stablehlo.multiply %v5174, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5176 = stablehlo.add %v5175, %v1251 : tensor<512x512x3x3xf32>
    %v5177 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5178 = stablehlo.multiply %v5177, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5179 = stablehlo.add %v5178, %v5176 : tensor<512x512x3x3xf32>
    %v5180 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5181 = stablehlo.multiply %v5180, %v5179 : tensor<512x512x3x3xf32>
    %v5182 = stablehlo.subtract %s4b1W2, %v5181 : tensor<512x512x3x3xf32>
    %v5183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5184 = stablehlo.multiply %v5183, %s4b1g2 : tensor<512xf32>
    %v5185 = stablehlo.add %v5184, %v1269 : tensor<512xf32>
    %v5186 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5187 = stablehlo.multiply %v5186, %s4b1g2v : tensor<512xf32>
    %v5188 = stablehlo.add %v5187, %v5185 : tensor<512xf32>
    %v5189 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5190 = stablehlo.multiply %v5189, %v5188 : tensor<512xf32>
    %v5191 = stablehlo.subtract %s4b1g2, %v5190 : tensor<512xf32>
    %v5192 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5193 = stablehlo.multiply %v5192, %s4b1bt2 : tensor<512xf32>
    %v5194 = stablehlo.add %v5193, %v1272 : tensor<512xf32>
    %v5195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5196 = stablehlo.multiply %v5195, %s4b1bt2v : tensor<512xf32>
    %v5197 = stablehlo.add %v5196, %v5194 : tensor<512xf32>
    %v5198 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5199 = stablehlo.multiply %v5198, %v5197 : tensor<512xf32>
    %v5200 = stablehlo.subtract %s4b1bt2, %v5199 : tensor<512xf32>
    %v5201 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5202 = stablehlo.multiply %v5201, %Wd : tensor<512x10xf32>
    %v5203 = stablehlo.add %v5202, %v1126 : tensor<512x10xf32>
    %v5204 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5205 = stablehlo.multiply %v5204, %Wdv : tensor<512x10xf32>
    %v5206 = stablehlo.add %v5205, %v5203 : tensor<512x10xf32>
    %v5207 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %v5208 = stablehlo.multiply %v5207, %v5206 : tensor<512x10xf32>
    %v5209 = stablehlo.subtract %Wd, %v5208 : tensor<512x10xf32>
    %v5210 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5211 = stablehlo.multiply %v5210, %bd : tensor<10xf32>
    %v5212 = stablehlo.add %v5211, %v1128 : tensor<10xf32>
    %v5213 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5214 = stablehlo.multiply %v5213, %bdv : tensor<10xf32>
    %v5215 = stablehlo.add %v5214, %v5212 : tensor<10xf32>
    %v5216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v5217 = stablehlo.multiply %v5216, %v5215 : tensor<10xf32>
    %v5218 = stablehlo.subtract %bd, %v5217 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1114 : tensor<32x10xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<32x10xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<32xf32>
    %laKc = stablehlo.constant dense<0.010000> : tensor<32xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<32xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<32xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<32xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<32.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v4237, %v4246, %v4255, %v4264, %v4273, %v4282, %v4291, %v4300, %v4309, %v4318, %v4327, %v4336, %v4345, %v4354, %v4363, %v4372, %v4381, %v4390, %v4399, %v4408, %v4417, %v4426, %v4435, %v4444, %v4453, %v4462, %v4471, %v4480, %v4489, %v4498, %v4507, %v4516, %v4525, %v4534, %v4543, %v4552, %v4561, %v4570, %v4579, %v4588, %v4597, %v4606, %v4615, %v4624, %v4633, %v4642, %v4651, %v4660, %v4669, %v4678, %v4687, %v4696, %v4705, %v4714, %v4723, %v4732, %v4741, %v4750, %v4759, %v4768, %v4777, %v4786, %v4795, %v4804, %v4813, %v4822, %v4831, %v4840, %v4849, %v4858, %v4867, %v4876, %v4885, %v4894, %v4903, %v4912, %v4921, %v4930, %v4939, %v4948, %v4957, %v4966, %v4975, %v4984, %v4993, %v5002, %v5011, %v5020, %v5029, %v5038, %v5047, %v5056, %v5065, %v5074, %v5083, %v5092, %v5101, %v5110, %v5119, %v5128, %v5137, %v5146, %v5155, %v5164, %v5173, %v5182, %v5191, %v5200, %v5209, %v5218, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4234, %v4243, %v4252, %v4261, %v4270, %v4279, %v4288, %v4297, %v4306, %v4315, %v4324, %v4333, %v4342, %v4351, %v4360, %v4369, %v4378, %v4387, %v4396, %v4405, %v4414, %v4423, %v4432, %v4441, %v4450, %v4459, %v4468, %v4477, %v4486, %v4495, %v4504, %v4513, %v4522, %v4531, %v4540, %v4549, %v4558, %v4567, %v4576, %v4585, %v4594, %v4603, %v4612, %v4621, %v4630, %v4639, %v4648, %v4657, %v4666, %v4675, %v4684, %v4693, %v4702, %v4711, %v4720, %v4729, %v4738, %v4747, %v4756, %v4765, %v4774, %v4783, %v4792, %v4801, %v4810, %v4819, %v4828, %v4837, %v4846, %v4855, %v4864, %v4873, %v4882, %v4891, %v4900, %v4909, %v4918, %v4927, %v4936, %v4945, %v4954, %v4963, %v4972, %v4981, %v4990, %v4999, %v5008, %v5017, %v5026, %v5035, %v5044, %v5053, %v5062, %v5071, %v5080, %v5089, %v5098, %v5107, %v5116, %v5125, %v5134, %v5143, %v5152, %v5161, %v5170, %v5179, %v5188, %v5197, %v5206, %v5215, %loss, %bc1, %bc2, %v3657, %v3668, %v3673, %v3684, %v3689, %v3700, %v3705, %v3716, %v3721, %v3732, %v3737, %v3748, %v3753, %v3764, %v3769, %v3780, %v3785, %v3796, %v3801, %v3812, %v3817, %v3828, %v3833, %v3844, %v3849, %v3860, %v3865, %v3876, %v3881, %v3892, %v3897, %v3908, %v3913, %v3924, %v3929, %v3940, %v3945, %v3956, %v3961, %v3972, %v3977, %v3988, %v3993, %v4004, %v4009, %v4020, %v4025, %v4036, %v4041, %v4052, %v4057, %v4068, %v4073, %v4084, %v4089, %v4100, %v4105, %v4116, %v4121, %v4132, %v4137, %v4148, %v4153, %v4164, %v4169, %v4180, %v4185, %v4196, %v4201, %v4212, %v4217, %v4228 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}
