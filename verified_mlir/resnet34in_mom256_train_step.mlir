module @m {
  func.func @resnet34in_mom256_train_step(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sbi: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sbim: tensor<64xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0b1m: tensor<64xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0b2m: tensor<64xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1b1m: tensor<64xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1b2m: tensor<64xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2b1m: tensor<64xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2b2m: tensor<64xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2b1m: tensor<128xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2b2m: tensor<128xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x3x3xf32>, %d2bpm: tensor<128xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0b1m: tensor<128xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0b2m: tensor<128xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1b1m: tensor<128xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1b2m: tensor<128xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2b1m: tensor<128xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2b2m: tensor<128xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3b1m: tensor<256xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3b2m: tensor<256xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x3x3xf32>, %d3bpm: tensor<256xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0b1m: tensor<256xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0b2m: tensor<256xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1b1m: tensor<256xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1b2m: tensor<256xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2b1m: tensor<256xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2b2m: tensor<256xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3b1m: tensor<256xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3b2m: tensor<256xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4b1m: tensor<256xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4b2m: tensor<256xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4b1m: tensor<512xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4b2m: tensor<512xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x3x3xf32>, %d4bpm: tensor<512xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0b1m: tensor<512xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0b2m: tensor<512xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1b1m: tensor<512xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1b2m: tensor<512xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sbiv: tensor<64xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0b1v: tensor<64xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0b2v: tensor<64xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1b1v: tensor<64xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1b2v: tensor<64xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2b1v: tensor<64xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2b2v: tensor<64xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2b1v: tensor<128xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2b2v: tensor<128xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x3x3xf32>, %d2bpv: tensor<128xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0b1v: tensor<128xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0b2v: tensor<128xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1b1v: tensor<128xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1b2v: tensor<128xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2b1v: tensor<128xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2b2v: tensor<128xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3b1v: tensor<256xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3b2v: tensor<256xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x3x3xf32>, %d3bpv: tensor<256xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0b1v: tensor<256xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0b2v: tensor<256xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1b1v: tensor<256xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1b2v: tensor<256xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2b1v: tensor<256xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2b2v: tensor<256xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3b1v: tensor<256xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3b2v: tensor<256xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4b1v: tensor<256xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4b2v: tensor<256xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4b1v: tensor<512xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4b2v: tensor<512xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x3x3xf32>, %d4bpv: tensor<512xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0b1v: tensor<512xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0b2v: tensor<512xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1b1v: tensor<512xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1b2v: tensor<512xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<256x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step: every line is pretty(verified AST node) ──
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<256x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %sbi, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
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
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %s1b0b1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
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
    %v60 = stablehlo.broadcast_in_dim %s1b0b2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
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
    %v83 = stablehlo.add %v82, %v30 : tensor<256x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<256x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %s1b1b1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v94 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v95 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v96 = stablehlo.broadcast_in_dim %v95, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v97 = stablehlo.divide %v96, %v93 : tensor<256x64x56x56xf32>
    %v98 = stablehlo.subtract %v91, %v97 : tensor<256x64x56x56xf32>
    %v99 = stablehlo.multiply %v98, %v98 : tensor<256x64x56x56xf32>
    %v100 = stablehlo.reduce(%v99 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v102 = stablehlo.divide %v101, %v93 : tensor<256x64x56x56xf32>
    %v103 = stablehlo.add %v102, %v94 : tensor<256x64x56x56xf32>
    %v104 = stablehlo.rsqrt %v103 : tensor<256x64x56x56xf32>
    %v105 = stablehlo.multiply %v98, %v104 : tensor<256x64x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v107 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v108 = stablehlo.multiply %v105, %v106 : tensor<256x64x56x56xf32>
    %v109 = stablehlo.add %v108, %v107 : tensor<256x64x56x56xf32>
    %v110 = stablehlo.reshape %v109 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v111 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v112 = stablehlo.maximum %v110, %v111 : tensor<256x200704xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v114 = stablehlo.convolution(%v113, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v115 = stablehlo.broadcast_in_dim %s1b1b2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<256x64x56x56xf32>
    %v117 = stablehlo.reshape %v116 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v118 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v120 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v121 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v122 = stablehlo.reduce(%v118 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v123 = stablehlo.broadcast_in_dim %v122, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v124 = stablehlo.divide %v123, %v120 : tensor<256x64x56x56xf32>
    %v125 = stablehlo.subtract %v118, %v124 : tensor<256x64x56x56xf32>
    %v126 = stablehlo.multiply %v125, %v125 : tensor<256x64x56x56xf32>
    %v127 = stablehlo.reduce(%v126 init: %v119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v128 = stablehlo.broadcast_in_dim %v127, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v129 = stablehlo.divide %v128, %v120 : tensor<256x64x56x56xf32>
    %v130 = stablehlo.add %v129, %v121 : tensor<256x64x56x56xf32>
    %v131 = stablehlo.rsqrt %v130 : tensor<256x64x56x56xf32>
    %v132 = stablehlo.multiply %v125, %v131 : tensor<256x64x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v134 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v135 = stablehlo.multiply %v132, %v133 : tensor<256x64x56x56xf32>
    %v136 = stablehlo.add %v135, %v134 : tensor<256x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v138 = stablehlo.add %v137, %v85 : tensor<256x200704xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<256x200704xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v142 = stablehlo.convolution(%v141, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v143 = stablehlo.broadcast_in_dim %s1b2b1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v144 = stablehlo.add %v142, %v143 : tensor<256x64x56x56xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v148 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v149 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v150 = stablehlo.reduce(%v146 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v151 = stablehlo.broadcast_in_dim %v150, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v152 = stablehlo.divide %v151, %v148 : tensor<256x64x56x56xf32>
    %v153 = stablehlo.subtract %v146, %v152 : tensor<256x64x56x56xf32>
    %v154 = stablehlo.multiply %v153, %v153 : tensor<256x64x56x56xf32>
    %v155 = stablehlo.reduce(%v154 init: %v147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v156 = stablehlo.broadcast_in_dim %v155, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v157 = stablehlo.divide %v156, %v148 : tensor<256x64x56x56xf32>
    %v158 = stablehlo.add %v157, %v149 : tensor<256x64x56x56xf32>
    %v159 = stablehlo.rsqrt %v158 : tensor<256x64x56x56xf32>
    %v160 = stablehlo.multiply %v153, %v159 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<256x64x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<256x64x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v166 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v167 = stablehlo.maximum %v165, %v166 : tensor<256x200704xf32>
    %v168 = stablehlo.reshape %v167 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v169 = stablehlo.convolution(%v168, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %s1b2b2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.add %v169, %v170 : tensor<256x64x56x56xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v175 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v176 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v177 = stablehlo.reduce(%v173 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v178 = stablehlo.broadcast_in_dim %v177, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.divide %v178, %v175 : tensor<256x64x56x56xf32>
    %v180 = stablehlo.subtract %v173, %v179 : tensor<256x64x56x56xf32>
    %v181 = stablehlo.multiply %v180, %v180 : tensor<256x64x56x56xf32>
    %v182 = stablehlo.reduce(%v181 init: %v174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v183 = stablehlo.broadcast_in_dim %v182, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v184 = stablehlo.divide %v183, %v175 : tensor<256x64x56x56xf32>
    %v185 = stablehlo.add %v184, %v176 : tensor<256x64x56x56xf32>
    %v186 = stablehlo.rsqrt %v185 : tensor<256x64x56x56xf32>
    %v187 = stablehlo.multiply %v180, %v186 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v189 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v190 = stablehlo.multiply %v187, %v188 : tensor<256x64x56x56xf32>
    %v191 = stablehlo.add %v190, %v189 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v193 = stablehlo.add %v192, %v140 : tensor<256x200704xf32>
    %v194 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v195 = stablehlo.maximum %v193, %v194 : tensor<256x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v197 = stablehlo.convolution(%v196, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v198 = stablehlo.broadcast_in_dim %d2b1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v199 = stablehlo.add %v197, %v198 : tensor<256x128x28x28xf32>
    %v200 = stablehlo.reshape %v199 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v201 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v203 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v204 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v205 = stablehlo.reduce(%v201 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v206 = stablehlo.broadcast_in_dim %v205, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v207 = stablehlo.divide %v206, %v203 : tensor<256x128x28x28xf32>
    %v208 = stablehlo.subtract %v201, %v207 : tensor<256x128x28x28xf32>
    %v209 = stablehlo.multiply %v208, %v208 : tensor<256x128x28x28xf32>
    %v210 = stablehlo.reduce(%v209 init: %v202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v211 = stablehlo.broadcast_in_dim %v210, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v212 = stablehlo.divide %v211, %v203 : tensor<256x128x28x28xf32>
    %v213 = stablehlo.add %v212, %v204 : tensor<256x128x28x28xf32>
    %v214 = stablehlo.rsqrt %v213 : tensor<256x128x28x28xf32>
    %v215 = stablehlo.multiply %v208, %v214 : tensor<256x128x28x28xf32>
    %v216 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v217 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v218 = stablehlo.multiply %v215, %v216 : tensor<256x128x28x28xf32>
    %v219 = stablehlo.add %v218, %v217 : tensor<256x128x28x28xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<256x100352xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v224 = stablehlo.convolution(%v223, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v225 = stablehlo.broadcast_in_dim %d2b2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v226 = stablehlo.add %v224, %v225 : tensor<256x128x28x28xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v230 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v231 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v232 = stablehlo.reduce(%v228 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v233 = stablehlo.broadcast_in_dim %v232, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v234 = stablehlo.divide %v233, %v230 : tensor<256x128x28x28xf32>
    %v235 = stablehlo.subtract %v228, %v234 : tensor<256x128x28x28xf32>
    %v236 = stablehlo.multiply %v235, %v235 : tensor<256x128x28x28xf32>
    %v237 = stablehlo.reduce(%v236 init: %v229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v238 = stablehlo.broadcast_in_dim %v237, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v239 = stablehlo.divide %v238, %v230 : tensor<256x128x28x28xf32>
    %v240 = stablehlo.add %v239, %v231 : tensor<256x128x28x28xf32>
    %v241 = stablehlo.rsqrt %v240 : tensor<256x128x28x28xf32>
    %v242 = stablehlo.multiply %v235, %v241 : tensor<256x128x28x28xf32>
    %v243 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v244 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v245 = stablehlo.multiply %v242, %v243 : tensor<256x128x28x28xf32>
    %v246 = stablehlo.add %v245, %v244 : tensor<256x128x28x28xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v248 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v249 = stablehlo.convolution(%v248, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v250 = stablehlo.broadcast_in_dim %d2bp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v251 = stablehlo.add %v249, %v250 : tensor<256x128x28x28xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v255 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v256 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v257 = stablehlo.reduce(%v253 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v258 = stablehlo.broadcast_in_dim %v257, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v259 = stablehlo.divide %v258, %v255 : tensor<256x128x28x28xf32>
    %v260 = stablehlo.subtract %v253, %v259 : tensor<256x128x28x28xf32>
    %v261 = stablehlo.multiply %v260, %v260 : tensor<256x128x28x28xf32>
    %v262 = stablehlo.reduce(%v261 init: %v254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v263 = stablehlo.broadcast_in_dim %v262, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v264 = stablehlo.divide %v263, %v255 : tensor<256x128x28x28xf32>
    %v265 = stablehlo.add %v264, %v256 : tensor<256x128x28x28xf32>
    %v266 = stablehlo.rsqrt %v265 : tensor<256x128x28x28xf32>
    %v267 = stablehlo.multiply %v260, %v266 : tensor<256x128x28x28xf32>
    %v268 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v269 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v270 = stablehlo.multiply %v267, %v268 : tensor<256x128x28x28xf32>
    %v271 = stablehlo.add %v270, %v269 : tensor<256x128x28x28xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v273 = stablehlo.add %v247, %v272 : tensor<256x100352xf32>
    %v274 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v275 = stablehlo.maximum %v273, %v274 : tensor<256x100352xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v277 = stablehlo.convolution(%v276, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v278 = stablehlo.broadcast_in_dim %s2b0b1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<256x128x28x28xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v283 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v284 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v285 = stablehlo.reduce(%v281 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v286 = stablehlo.broadcast_in_dim %v285, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v287 = stablehlo.divide %v286, %v283 : tensor<256x128x28x28xf32>
    %v288 = stablehlo.subtract %v281, %v287 : tensor<256x128x28x28xf32>
    %v289 = stablehlo.multiply %v288, %v288 : tensor<256x128x28x28xf32>
    %v290 = stablehlo.reduce(%v289 init: %v282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v291 = stablehlo.broadcast_in_dim %v290, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v292 = stablehlo.divide %v291, %v283 : tensor<256x128x28x28xf32>
    %v293 = stablehlo.add %v292, %v284 : tensor<256x128x28x28xf32>
    %v294 = stablehlo.rsqrt %v293 : tensor<256x128x28x28xf32>
    %v295 = stablehlo.multiply %v288, %v294 : tensor<256x128x28x28xf32>
    %v296 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v297 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v298 = stablehlo.multiply %v295, %v296 : tensor<256x128x28x28xf32>
    %v299 = stablehlo.add %v298, %v297 : tensor<256x128x28x28xf32>
    %v300 = stablehlo.reshape %v299 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v302 = stablehlo.maximum %v300, %v301 : tensor<256x100352xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v304 = stablehlo.convolution(%v303, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v305 = stablehlo.broadcast_in_dim %s2b0b2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v306 = stablehlo.add %v304, %v305 : tensor<256x128x28x28xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v310 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v311 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v312 = stablehlo.reduce(%v308 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v314 = stablehlo.divide %v313, %v310 : tensor<256x128x28x28xf32>
    %v315 = stablehlo.subtract %v308, %v314 : tensor<256x128x28x28xf32>
    %v316 = stablehlo.multiply %v315, %v315 : tensor<256x128x28x28xf32>
    %v317 = stablehlo.reduce(%v316 init: %v309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v318 = stablehlo.broadcast_in_dim %v317, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v319 = stablehlo.divide %v318, %v310 : tensor<256x128x28x28xf32>
    %v320 = stablehlo.add %v319, %v311 : tensor<256x128x28x28xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<256x128x28x28xf32>
    %v322 = stablehlo.multiply %v315, %v321 : tensor<256x128x28x28xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v324 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v325 = stablehlo.multiply %v322, %v323 : tensor<256x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v324 : tensor<256x128x28x28xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v328 = stablehlo.add %v327, %v275 : tensor<256x100352xf32>
    %v329 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v330 = stablehlo.maximum %v328, %v329 : tensor<256x100352xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v332 = stablehlo.convolution(%v331, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v333 = stablehlo.broadcast_in_dim %s2b1b1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<256x128x28x28xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<256x128x28x28xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<256x128x28x28xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<256x128x28x28xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<256x128x28x28xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<256x128x28x28xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<256x128x28x28xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<256x128x28x28xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v356 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v357 = stablehlo.maximum %v355, %v356 : tensor<256x100352xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v359 = stablehlo.convolution(%v358, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s2b1b2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v361 = stablehlo.add %v359, %v360 : tensor<256x128x28x28xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v365 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v367 = stablehlo.reduce(%v363 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v368 = stablehlo.broadcast_in_dim %v367, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v369 = stablehlo.divide %v368, %v365 : tensor<256x128x28x28xf32>
    %v370 = stablehlo.subtract %v363, %v369 : tensor<256x128x28x28xf32>
    %v371 = stablehlo.multiply %v370, %v370 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.reduce(%v371 init: %v364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v373 = stablehlo.broadcast_in_dim %v372, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v374 = stablehlo.divide %v373, %v365 : tensor<256x128x28x28xf32>
    %v375 = stablehlo.add %v374, %v366 : tensor<256x128x28x28xf32>
    %v376 = stablehlo.rsqrt %v375 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.multiply %v370, %v376 : tensor<256x128x28x28xf32>
    %v378 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v380 = stablehlo.multiply %v377, %v378 : tensor<256x128x28x28xf32>
    %v381 = stablehlo.add %v380, %v379 : tensor<256x128x28x28xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v383 = stablehlo.add %v382, %v330 : tensor<256x100352xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v385 = stablehlo.maximum %v383, %v384 : tensor<256x100352xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v387 = stablehlo.convolution(%v386, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b2b1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v389 = stablehlo.add %v387, %v388 : tensor<256x128x28x28xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v393 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v394 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v395 = stablehlo.reduce(%v391 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v397 = stablehlo.divide %v396, %v393 : tensor<256x128x28x28xf32>
    %v398 = stablehlo.subtract %v391, %v397 : tensor<256x128x28x28xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<256x128x28x28xf32>
    %v400 = stablehlo.reduce(%v399 init: %v392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v401 = stablehlo.broadcast_in_dim %v400, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v402 = stablehlo.divide %v401, %v393 : tensor<256x128x28x28xf32>
    %v403 = stablehlo.add %v402, %v394 : tensor<256x128x28x28xf32>
    %v404 = stablehlo.rsqrt %v403 : tensor<256x128x28x28xf32>
    %v405 = stablehlo.multiply %v398, %v404 : tensor<256x128x28x28xf32>
    %v406 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v407 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v408 = stablehlo.multiply %v405, %v406 : tensor<256x128x28x28xf32>
    %v409 = stablehlo.add %v408, %v407 : tensor<256x128x28x28xf32>
    %v410 = stablehlo.reshape %v409 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v411 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v412 = stablehlo.maximum %v410, %v411 : tensor<256x100352xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v414 = stablehlo.convolution(%v413, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v415 = stablehlo.broadcast_in_dim %s2b2b2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v416 = stablehlo.add %v414, %v415 : tensor<256x128x28x28xf32>
    %v417 = stablehlo.reshape %v416 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v421 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v422 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v424 = stablehlo.divide %v423, %v420 : tensor<256x128x28x28xf32>
    %v425 = stablehlo.subtract %v418, %v424 : tensor<256x128x28x28xf32>
    %v426 = stablehlo.multiply %v425, %v425 : tensor<256x128x28x28xf32>
    %v427 = stablehlo.reduce(%v426 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v428 = stablehlo.broadcast_in_dim %v427, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v429 = stablehlo.divide %v428, %v420 : tensor<256x128x28x28xf32>
    %v430 = stablehlo.add %v429, %v421 : tensor<256x128x28x28xf32>
    %v431 = stablehlo.rsqrt %v430 : tensor<256x128x28x28xf32>
    %v432 = stablehlo.multiply %v425, %v431 : tensor<256x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v435 = stablehlo.multiply %v432, %v433 : tensor<256x128x28x28xf32>
    %v436 = stablehlo.add %v435, %v434 : tensor<256x128x28x28xf32>
    %v437 = stablehlo.reshape %v436 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v438 = stablehlo.add %v437, %v385 : tensor<256x100352xf32>
    %v439 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v440 = stablehlo.maximum %v438, %v439 : tensor<256x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v442 = stablehlo.convolution(%v441, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v443 = stablehlo.broadcast_in_dim %d3b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v444 = stablehlo.add %v442, %v443 : tensor<256x256x14x14xf32>
    %v445 = stablehlo.reshape %v444 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v448 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v449 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v450 = stablehlo.reduce(%v446 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v452 = stablehlo.divide %v451, %v448 : tensor<256x256x14x14xf32>
    %v453 = stablehlo.subtract %v446, %v452 : tensor<256x256x14x14xf32>
    %v454 = stablehlo.multiply %v453, %v453 : tensor<256x256x14x14xf32>
    %v455 = stablehlo.reduce(%v454 init: %v447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v456 = stablehlo.broadcast_in_dim %v455, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v457 = stablehlo.divide %v456, %v448 : tensor<256x256x14x14xf32>
    %v458 = stablehlo.add %v457, %v449 : tensor<256x256x14x14xf32>
    %v459 = stablehlo.rsqrt %v458 : tensor<256x256x14x14xf32>
    %v460 = stablehlo.multiply %v453, %v459 : tensor<256x256x14x14xf32>
    %v461 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v462 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v463 = stablehlo.multiply %v460, %v461 : tensor<256x256x14x14xf32>
    %v464 = stablehlo.add %v463, %v462 : tensor<256x256x14x14xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v466 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v467 = stablehlo.maximum %v465, %v466 : tensor<256x50176xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v469 = stablehlo.convolution(%v468, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v470 = stablehlo.broadcast_in_dim %d3b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v471 = stablehlo.add %v469, %v470 : tensor<256x256x14x14xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v473 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v475 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v476 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v477 = stablehlo.reduce(%v473 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v479 = stablehlo.divide %v478, %v475 : tensor<256x256x14x14xf32>
    %v480 = stablehlo.subtract %v473, %v479 : tensor<256x256x14x14xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<256x256x14x14xf32>
    %v482 = stablehlo.reduce(%v481 init: %v474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v483 = stablehlo.broadcast_in_dim %v482, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v484 = stablehlo.divide %v483, %v475 : tensor<256x256x14x14xf32>
    %v485 = stablehlo.add %v484, %v476 : tensor<256x256x14x14xf32>
    %v486 = stablehlo.rsqrt %v485 : tensor<256x256x14x14xf32>
    %v487 = stablehlo.multiply %v480, %v486 : tensor<256x256x14x14xf32>
    %v488 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v489 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v490 = stablehlo.multiply %v487, %v488 : tensor<256x256x14x14xf32>
    %v491 = stablehlo.add %v490, %v489 : tensor<256x256x14x14xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v493 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v494 = stablehlo.convolution(%v493, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v495 = stablehlo.broadcast_in_dim %d3bp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<256x256x14x14xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v499 = stablehlo.constant dense<0.0> : tensor<f32>
    %v500 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v501 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v502 = stablehlo.reduce(%v498 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v503 = stablehlo.broadcast_in_dim %v502, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v504 = stablehlo.divide %v503, %v500 : tensor<256x256x14x14xf32>
    %v505 = stablehlo.subtract %v498, %v504 : tensor<256x256x14x14xf32>
    %v506 = stablehlo.multiply %v505, %v505 : tensor<256x256x14x14xf32>
    %v507 = stablehlo.reduce(%v506 init: %v499) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v508 = stablehlo.broadcast_in_dim %v507, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v509 = stablehlo.divide %v508, %v500 : tensor<256x256x14x14xf32>
    %v510 = stablehlo.add %v509, %v501 : tensor<256x256x14x14xf32>
    %v511 = stablehlo.rsqrt %v510 : tensor<256x256x14x14xf32>
    %v512 = stablehlo.multiply %v505, %v511 : tensor<256x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v514 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v515 = stablehlo.multiply %v512, %v513 : tensor<256x256x14x14xf32>
    %v516 = stablehlo.add %v515, %v514 : tensor<256x256x14x14xf32>
    %v517 = stablehlo.reshape %v516 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v518 = stablehlo.add %v492, %v517 : tensor<256x50176xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v520 = stablehlo.maximum %v518, %v519 : tensor<256x50176xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v522 = stablehlo.convolution(%v521, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v523 = stablehlo.broadcast_in_dim %s3b0b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v524 = stablehlo.add %v522, %v523 : tensor<256x256x14x14xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v528 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v529 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v530 = stablehlo.reduce(%v526 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v531 = stablehlo.broadcast_in_dim %v530, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v532 = stablehlo.divide %v531, %v528 : tensor<256x256x14x14xf32>
    %v533 = stablehlo.subtract %v526, %v532 : tensor<256x256x14x14xf32>
    %v534 = stablehlo.multiply %v533, %v533 : tensor<256x256x14x14xf32>
    %v535 = stablehlo.reduce(%v534 init: %v527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v536 = stablehlo.broadcast_in_dim %v535, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v537 = stablehlo.divide %v536, %v528 : tensor<256x256x14x14xf32>
    %v538 = stablehlo.add %v537, %v529 : tensor<256x256x14x14xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<256x256x14x14xf32>
    %v540 = stablehlo.multiply %v533, %v539 : tensor<256x256x14x14xf32>
    %v541 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<256x256x14x14xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<256x256x14x14xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<256x50176xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v549 = stablehlo.convolution(%v548, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v550 = stablehlo.broadcast_in_dim %s3b0b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<256x256x14x14xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<256x256x14x14xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<256x256x14x14xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<256x256x14x14xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<256x256x14x14xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<256x256x14x14xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<256x256x14x14xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<256x256x14x14xf32>
    %v568 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v569 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<256x256x14x14xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<256x256x14x14xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v573 = stablehlo.add %v572, %v520 : tensor<256x50176xf32>
    %v574 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v575 = stablehlo.maximum %v573, %v574 : tensor<256x50176xf32>
    %v576 = stablehlo.reshape %v575 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v577 = stablehlo.convolution(%v576, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v578 = stablehlo.broadcast_in_dim %s3b1b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v579 = stablehlo.add %v577, %v578 : tensor<256x256x14x14xf32>
    %v580 = stablehlo.reshape %v579 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v581 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v583 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v584 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v585 = stablehlo.reduce(%v581 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v586 = stablehlo.broadcast_in_dim %v585, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v587 = stablehlo.divide %v586, %v583 : tensor<256x256x14x14xf32>
    %v588 = stablehlo.subtract %v581, %v587 : tensor<256x256x14x14xf32>
    %v589 = stablehlo.multiply %v588, %v588 : tensor<256x256x14x14xf32>
    %v590 = stablehlo.reduce(%v589 init: %v582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v583 : tensor<256x256x14x14xf32>
    %v593 = stablehlo.add %v592, %v584 : tensor<256x256x14x14xf32>
    %v594 = stablehlo.rsqrt %v593 : tensor<256x256x14x14xf32>
    %v595 = stablehlo.multiply %v588, %v594 : tensor<256x256x14x14xf32>
    %v596 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v597 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v598 = stablehlo.multiply %v595, %v596 : tensor<256x256x14x14xf32>
    %v599 = stablehlo.add %v598, %v597 : tensor<256x256x14x14xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v602 = stablehlo.maximum %v600, %v601 : tensor<256x50176xf32>
    %v603 = stablehlo.reshape %v602 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v604 = stablehlo.convolution(%v603, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v605 = stablehlo.broadcast_in_dim %s3b1b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v606 = stablehlo.add %v604, %v605 : tensor<256x256x14x14xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v611 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<256x256x14x14xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<256x256x14x14xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<256x256x14x14xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<256x256x14x14xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<256x256x14x14xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<256x256x14x14xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<256x256x14x14xf32>
    %v623 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v624 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<256x256x14x14xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<256x256x14x14xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v628 = stablehlo.add %v627, %v575 : tensor<256x50176xf32>
    %v629 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v630 = stablehlo.maximum %v628, %v629 : tensor<256x50176xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v632 = stablehlo.convolution(%v631, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v633 = stablehlo.broadcast_in_dim %s3b2b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<256x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v638 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v639 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v640 = stablehlo.reduce(%v636 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v641 = stablehlo.broadcast_in_dim %v640, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v642 = stablehlo.divide %v641, %v638 : tensor<256x256x14x14xf32>
    %v643 = stablehlo.subtract %v636, %v642 : tensor<256x256x14x14xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<256x256x14x14xf32>
    %v645 = stablehlo.reduce(%v644 init: %v637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v646 = stablehlo.broadcast_in_dim %v645, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v647 = stablehlo.divide %v646, %v638 : tensor<256x256x14x14xf32>
    %v648 = stablehlo.add %v647, %v639 : tensor<256x256x14x14xf32>
    %v649 = stablehlo.rsqrt %v648 : tensor<256x256x14x14xf32>
    %v650 = stablehlo.multiply %v643, %v649 : tensor<256x256x14x14xf32>
    %v651 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v652 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v653 = stablehlo.multiply %v650, %v651 : tensor<256x256x14x14xf32>
    %v654 = stablehlo.add %v653, %v652 : tensor<256x256x14x14xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v656 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v657 = stablehlo.maximum %v655, %v656 : tensor<256x50176xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v659 = stablehlo.convolution(%v658, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v660 = stablehlo.broadcast_in_dim %s3b2b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<256x256x14x14xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<256x256x14x14xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<256x256x14x14xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<256x256x14x14xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<256x256x14x14xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<256x256x14x14xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<256x256x14x14xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<256x256x14x14xf32>
    %v678 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v679 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<256x256x14x14xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<256x256x14x14xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v683 = stablehlo.add %v682, %v630 : tensor<256x50176xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<256x50176xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v687 = stablehlo.convolution(%v686, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v688 = stablehlo.broadcast_in_dim %s3b3b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v689 = stablehlo.add %v687, %v688 : tensor<256x256x14x14xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v692 = stablehlo.constant dense<0.0> : tensor<f32>
    %v693 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v694 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v695 = stablehlo.reduce(%v691 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v696 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v697 = stablehlo.divide %v696, %v693 : tensor<256x256x14x14xf32>
    %v698 = stablehlo.subtract %v691, %v697 : tensor<256x256x14x14xf32>
    %v699 = stablehlo.multiply %v698, %v698 : tensor<256x256x14x14xf32>
    %v700 = stablehlo.reduce(%v699 init: %v692) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v701 = stablehlo.broadcast_in_dim %v700, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v702 = stablehlo.divide %v701, %v693 : tensor<256x256x14x14xf32>
    %v703 = stablehlo.add %v702, %v694 : tensor<256x256x14x14xf32>
    %v704 = stablehlo.rsqrt %v703 : tensor<256x256x14x14xf32>
    %v705 = stablehlo.multiply %v698, %v704 : tensor<256x256x14x14xf32>
    %v706 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v707 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v708 = stablehlo.multiply %v705, %v706 : tensor<256x256x14x14xf32>
    %v709 = stablehlo.add %v708, %v707 : tensor<256x256x14x14xf32>
    %v710 = stablehlo.reshape %v709 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v711 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v712 = stablehlo.maximum %v710, %v711 : tensor<256x50176xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v714 = stablehlo.convolution(%v713, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v715 = stablehlo.broadcast_in_dim %s3b3b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v716 = stablehlo.add %v714, %v715 : tensor<256x256x14x14xf32>
    %v717 = stablehlo.reshape %v716 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v720 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v721 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v722 = stablehlo.reduce(%v718 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v723 = stablehlo.broadcast_in_dim %v722, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v724 = stablehlo.divide %v723, %v720 : tensor<256x256x14x14xf32>
    %v725 = stablehlo.subtract %v718, %v724 : tensor<256x256x14x14xf32>
    %v726 = stablehlo.multiply %v725, %v725 : tensor<256x256x14x14xf32>
    %v727 = stablehlo.reduce(%v726 init: %v719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v728 = stablehlo.broadcast_in_dim %v727, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v729 = stablehlo.divide %v728, %v720 : tensor<256x256x14x14xf32>
    %v730 = stablehlo.add %v729, %v721 : tensor<256x256x14x14xf32>
    %v731 = stablehlo.rsqrt %v730 : tensor<256x256x14x14xf32>
    %v732 = stablehlo.multiply %v725, %v731 : tensor<256x256x14x14xf32>
    %v733 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v734 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v735 = stablehlo.multiply %v732, %v733 : tensor<256x256x14x14xf32>
    %v736 = stablehlo.add %v735, %v734 : tensor<256x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v738 = stablehlo.add %v737, %v685 : tensor<256x50176xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v740 = stablehlo.maximum %v738, %v739 : tensor<256x50176xf32>
    %v741 = stablehlo.reshape %v740 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v742 = stablehlo.convolution(%v741, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v743 = stablehlo.broadcast_in_dim %s3b4b1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v744 = stablehlo.add %v742, %v743 : tensor<256x256x14x14xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v749 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<256x256x14x14xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<256x256x14x14xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<256x256x14x14xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<256x256x14x14xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<256x256x14x14xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<256x256x14x14xf32>
    %v761 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<256x256x14x14xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<256x256x14x14xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v767 = stablehlo.maximum %v765, %v766 : tensor<256x50176xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v769 = stablehlo.convolution(%v768, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %s3b4b2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.add %v769, %v770 : tensor<256x256x14x14xf32>
    %v772 = stablehlo.reshape %v771 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v775 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v776 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v777 = stablehlo.reduce(%v773 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v778 = stablehlo.broadcast_in_dim %v777, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v779 = stablehlo.divide %v778, %v775 : tensor<256x256x14x14xf32>
    %v780 = stablehlo.subtract %v773, %v779 : tensor<256x256x14x14xf32>
    %v781 = stablehlo.multiply %v780, %v780 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.reduce(%v781 init: %v774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v783 = stablehlo.broadcast_in_dim %v782, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v784 = stablehlo.divide %v783, %v775 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v776 : tensor<256x256x14x14xf32>
    %v786 = stablehlo.rsqrt %v785 : tensor<256x256x14x14xf32>
    %v787 = stablehlo.multiply %v780, %v786 : tensor<256x256x14x14xf32>
    %v788 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v789 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v790 = stablehlo.multiply %v787, %v788 : tensor<256x256x14x14xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<256x256x14x14xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v793 = stablehlo.add %v792, %v740 : tensor<256x50176xf32>
    %v794 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v795 = stablehlo.maximum %v793, %v794 : tensor<256x50176xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v797 = stablehlo.convolution(%v796, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v798 = stablehlo.broadcast_in_dim %d4b1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v799 = stablehlo.add %v797, %v798 : tensor<256x512x7x7xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v801 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v803 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v804 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v805 = stablehlo.reduce(%v801 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v806 = stablehlo.broadcast_in_dim %v805, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v807 = stablehlo.divide %v806, %v803 : tensor<256x512x7x7xf32>
    %v808 = stablehlo.subtract %v801, %v807 : tensor<256x512x7x7xf32>
    %v809 = stablehlo.multiply %v808, %v808 : tensor<256x512x7x7xf32>
    %v810 = stablehlo.reduce(%v809 init: %v802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v811 = stablehlo.broadcast_in_dim %v810, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v812 = stablehlo.divide %v811, %v803 : tensor<256x512x7x7xf32>
    %v813 = stablehlo.add %v812, %v804 : tensor<256x512x7x7xf32>
    %v814 = stablehlo.rsqrt %v813 : tensor<256x512x7x7xf32>
    %v815 = stablehlo.multiply %v808, %v814 : tensor<256x512x7x7xf32>
    %v816 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v817 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v818 = stablehlo.multiply %v815, %v816 : tensor<256x512x7x7xf32>
    %v819 = stablehlo.add %v818, %v817 : tensor<256x512x7x7xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<256x25088xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v824 = stablehlo.convolution(%v823, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v825 = stablehlo.broadcast_in_dim %d4b2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v826 = stablehlo.add %v824, %v825 : tensor<256x512x7x7xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v830 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v831 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v832 = stablehlo.reduce(%v828 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v834 = stablehlo.divide %v833, %v830 : tensor<256x512x7x7xf32>
    %v835 = stablehlo.subtract %v828, %v834 : tensor<256x512x7x7xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<256x512x7x7xf32>
    %v837 = stablehlo.reduce(%v836 init: %v829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v838 = stablehlo.broadcast_in_dim %v837, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v839 = stablehlo.divide %v838, %v830 : tensor<256x512x7x7xf32>
    %v840 = stablehlo.add %v839, %v831 : tensor<256x512x7x7xf32>
    %v841 = stablehlo.rsqrt %v840 : tensor<256x512x7x7xf32>
    %v842 = stablehlo.multiply %v835, %v841 : tensor<256x512x7x7xf32>
    %v843 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v844 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v845 = stablehlo.multiply %v842, %v843 : tensor<256x512x7x7xf32>
    %v846 = stablehlo.add %v845, %v844 : tensor<256x512x7x7xf32>
    %v847 = stablehlo.reshape %v846 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v848 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v849 = stablehlo.convolution(%v848, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v850 = stablehlo.broadcast_in_dim %d4bp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v851 = stablehlo.add %v849, %v850 : tensor<256x512x7x7xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v855 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v856 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v857 = stablehlo.reduce(%v853 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v858 = stablehlo.broadcast_in_dim %v857, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v859 = stablehlo.divide %v858, %v855 : tensor<256x512x7x7xf32>
    %v860 = stablehlo.subtract %v853, %v859 : tensor<256x512x7x7xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<256x512x7x7xf32>
    %v862 = stablehlo.reduce(%v861 init: %v854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v863 = stablehlo.broadcast_in_dim %v862, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v864 = stablehlo.divide %v863, %v855 : tensor<256x512x7x7xf32>
    %v865 = stablehlo.add %v864, %v856 : tensor<256x512x7x7xf32>
    %v866 = stablehlo.rsqrt %v865 : tensor<256x512x7x7xf32>
    %v867 = stablehlo.multiply %v860, %v866 : tensor<256x512x7x7xf32>
    %v868 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v869 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v870 = stablehlo.multiply %v867, %v868 : tensor<256x512x7x7xf32>
    %v871 = stablehlo.add %v870, %v869 : tensor<256x512x7x7xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v873 = stablehlo.add %v847, %v872 : tensor<256x25088xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<256x25088xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v877 = stablehlo.convolution(%v876, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v878 = stablehlo.broadcast_in_dim %s4b0b1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<256x512x7x7xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v883 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v884 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v885 = stablehlo.reduce(%v881 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v887 = stablehlo.divide %v886, %v883 : tensor<256x512x7x7xf32>
    %v888 = stablehlo.subtract %v881, %v887 : tensor<256x512x7x7xf32>
    %v889 = stablehlo.multiply %v888, %v888 : tensor<256x512x7x7xf32>
    %v890 = stablehlo.reduce(%v889 init: %v882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v891 = stablehlo.broadcast_in_dim %v890, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v892 = stablehlo.divide %v891, %v883 : tensor<256x512x7x7xf32>
    %v893 = stablehlo.add %v892, %v884 : tensor<256x512x7x7xf32>
    %v894 = stablehlo.rsqrt %v893 : tensor<256x512x7x7xf32>
    %v895 = stablehlo.multiply %v888, %v894 : tensor<256x512x7x7xf32>
    %v896 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v897 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v898 = stablehlo.multiply %v895, %v896 : tensor<256x512x7x7xf32>
    %v899 = stablehlo.add %v898, %v897 : tensor<256x512x7x7xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v901 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v902 = stablehlo.maximum %v900, %v901 : tensor<256x25088xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v904 = stablehlo.convolution(%v903, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v905 = stablehlo.broadcast_in_dim %s4b0b2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<256x512x7x7xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v911 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v912 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v914 = stablehlo.divide %v913, %v910 : tensor<256x512x7x7xf32>
    %v915 = stablehlo.subtract %v908, %v914 : tensor<256x512x7x7xf32>
    %v916 = stablehlo.multiply %v915, %v915 : tensor<256x512x7x7xf32>
    %v917 = stablehlo.reduce(%v916 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v919 = stablehlo.divide %v918, %v910 : tensor<256x512x7x7xf32>
    %v920 = stablehlo.add %v919, %v911 : tensor<256x512x7x7xf32>
    %v921 = stablehlo.rsqrt %v920 : tensor<256x512x7x7xf32>
    %v922 = stablehlo.multiply %v915, %v921 : tensor<256x512x7x7xf32>
    %v923 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v924 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v925 = stablehlo.multiply %v922, %v923 : tensor<256x512x7x7xf32>
    %v926 = stablehlo.add %v925, %v924 : tensor<256x512x7x7xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v928 = stablehlo.add %v927, %v875 : tensor<256x25088xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v930 = stablehlo.maximum %v928, %v929 : tensor<256x25088xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v932 = stablehlo.convolution(%v931, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v933 = stablehlo.broadcast_in_dim %s4b1b1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v934 = stablehlo.add %v932, %v933 : tensor<256x512x7x7xf32>
    %v935 = stablehlo.reshape %v934 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v936 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v938 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v939 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v940 = stablehlo.reduce(%v936 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v942 = stablehlo.divide %v941, %v938 : tensor<256x512x7x7xf32>
    %v943 = stablehlo.subtract %v936, %v942 : tensor<256x512x7x7xf32>
    %v944 = stablehlo.multiply %v943, %v943 : tensor<256x512x7x7xf32>
    %v945 = stablehlo.reduce(%v944 init: %v937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v946 = stablehlo.broadcast_in_dim %v945, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v947 = stablehlo.divide %v946, %v938 : tensor<256x512x7x7xf32>
    %v948 = stablehlo.add %v947, %v939 : tensor<256x512x7x7xf32>
    %v949 = stablehlo.rsqrt %v948 : tensor<256x512x7x7xf32>
    %v950 = stablehlo.multiply %v943, %v949 : tensor<256x512x7x7xf32>
    %v951 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v952 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v953 = stablehlo.multiply %v950, %v951 : tensor<256x512x7x7xf32>
    %v954 = stablehlo.add %v953, %v952 : tensor<256x512x7x7xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v956 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v957 = stablehlo.maximum %v955, %v956 : tensor<256x25088xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v959 = stablehlo.convolution(%v958, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v960 = stablehlo.broadcast_in_dim %s4b1b2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v961 = stablehlo.add %v959, %v960 : tensor<256x512x7x7xf32>
    %v962 = stablehlo.reshape %v961 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v966 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v967 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v969 = stablehlo.divide %v968, %v965 : tensor<256x512x7x7xf32>
    %v970 = stablehlo.subtract %v963, %v969 : tensor<256x512x7x7xf32>
    %v971 = stablehlo.multiply %v970, %v970 : tensor<256x512x7x7xf32>
    %v972 = stablehlo.reduce(%v971 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v973 = stablehlo.broadcast_in_dim %v972, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v974 = stablehlo.divide %v973, %v965 : tensor<256x512x7x7xf32>
    %v975 = stablehlo.add %v974, %v966 : tensor<256x512x7x7xf32>
    %v976 = stablehlo.rsqrt %v975 : tensor<256x512x7x7xf32>
    %v977 = stablehlo.multiply %v970, %v976 : tensor<256x512x7x7xf32>
    %v978 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v979 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v980 = stablehlo.multiply %v977, %v978 : tensor<256x512x7x7xf32>
    %v981 = stablehlo.add %v980, %v979 : tensor<256x512x7x7xf32>
    %v982 = stablehlo.reshape %v981 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v983 = stablehlo.add %v982, %v930 : tensor<256x25088xf32>
    %v984 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v985 = stablehlo.maximum %v983, %v984 : tensor<256x25088xf32>
    %v986 = stablehlo.reshape %v985 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v988 = stablehlo.reduce(%v986 init: %v987) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512xf32>
    %v989 = stablehlo.constant dense<49.0> : tensor<256x512xf32>
    %v990 = stablehlo.divide %v988, %v989 : tensor<256x512xf32>
    %v991 = stablehlo.dot_general %v990, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<512x1000xf32>) -> tensor<256x1000xf32>
    %v992 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v993 = stablehlo.add %v991, %v992 : tensor<256x1000xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<f32>
    %v996 = stablehlo.exponential %v994 : tensor<256x1x1000xf32>
    %v997 = stablehlo.reduce(%v996 init: %v995) applies stablehlo.add across dimensions = [2] : (tensor<256x1x1000xf32>, tensor<f32>) -> tensor<256x1xf32>
    %v998 = stablehlo.broadcast_in_dim %v997, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x1x1000xf32>
    %v999 = stablehlo.divide %v996, %v998 : tensor<256x1x1000xf32>
    %v1000 = stablehlo.reshape %v999 : (tensor<256x1x1000xf32>) -> tensor<256x1000xf32>
    %v1001 = stablehlo.subtract %v1000, %onehot : tensor<256x1000xf32>
    %v1002 = stablehlo.constant dense<0.100000> : tensor<256x1000xf32>
    %v1003 = stablehlo.multiply %onehot, %v1002 : tensor<256x1000xf32>
    %v1004 = stablehlo.add %v1001, %v1003 : tensor<256x1000xf32>
    %v1005 = stablehlo.constant dense<-0.000100> : tensor<256x1000xf32>
    %v1006 = stablehlo.add %v1004, %v1005 : tensor<256x1000xf32>
    %v1007 = stablehlo.constant dense<256.0> : tensor<256x1000xf32>
    %v1008 = stablehlo.divide %v1006, %v1007 : tensor<256x1000xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1010 = stablehlo.dot_general %v1009, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1x1000xf32>, tensor<512x1000xf32>) -> tensor<256x1x512xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<256x1x512xf32>) -> tensor<256x512xf32>
    %v1012 = stablehlo.dot_general %v990, %v1008, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<256x1000xf32>) -> tensor<512x1000xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1014 = stablehlo.reduce(%v1008 init: %v1013) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1011, dims = [0, 1] : (tensor<256x512xf32>) -> tensor<256x512x7x7xf32>
    %v1016 = stablehlo.constant dense<49.0> : tensor<256x512x7x7xf32>
    %v1017 = stablehlo.divide %v1015, %v1016 : tensor<256x512x7x7xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1019 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1020 = stablehlo.compare GT, %v983, %v1019 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1021 = stablehlo.select %v1020, %v1018, %v1019 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1022 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1025 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<256x512x7x7xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<256x512x7x7xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<256x512x7x7xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<256x512x7x7xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<256x512x7x7xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<256x512x7x7xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<256x512x7x7xf32>
    %v1037 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1038 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1039 = stablehlo.multiply %v1037, %v1038 : tensor<256x512x7x7xf32>
    %v1040 = stablehlo.reduce(%v1039 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1042 = stablehlo.multiply %v1036, %v1039 : tensor<256x512x7x7xf32>
    %v1043 = stablehlo.reduce(%v1042 init: %v1023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1044 = stablehlo.broadcast_in_dim %v1043, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1045 = stablehlo.multiply %v1039, %v1024 : tensor<256x512x7x7xf32>
    %v1046 = stablehlo.subtract %v1045, %v1041 : tensor<256x512x7x7xf32>
    %v1047 = stablehlo.multiply %v1036, %v1044 : tensor<256x512x7x7xf32>
    %v1048 = stablehlo.subtract %v1046, %v1047 : tensor<256x512x7x7xf32>
    %v1049 = stablehlo.divide %v1035, %v1024 : tensor<256x512x7x7xf32>
    %v1050 = stablehlo.multiply %v1049, %v1048 : tensor<256x512x7x7xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1053 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1054 = stablehlo.transpose %v1053, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1055 = stablehlo.convolution(%v1052, %v1054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1056 = stablehlo.reshape %v1055 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1057 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1058 = stablehlo.compare GT, %v955, %v1057 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1059 = stablehlo.select %v1058, %v1056, %v1057 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1060 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1064 = stablehlo.reduce(%v1060 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<256x512x7x7xf32>
    %v1067 = stablehlo.subtract %v1060, %v1066 : tensor<256x512x7x7xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<256x512x7x7xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<256x512x7x7xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<256x512x7x7xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<256x512x7x7xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<256x512x7x7xf32>
    %v1075 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1076 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1077 = stablehlo.multiply %v1075, %v1076 : tensor<256x512x7x7xf32>
    %v1078 = stablehlo.reduce(%v1077 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1079 = stablehlo.broadcast_in_dim %v1078, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1080 = stablehlo.multiply %v1074, %v1077 : tensor<256x512x7x7xf32>
    %v1081 = stablehlo.reduce(%v1080 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1082 = stablehlo.broadcast_in_dim %v1081, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1083 = stablehlo.multiply %v1077, %v1062 : tensor<256x512x7x7xf32>
    %v1084 = stablehlo.subtract %v1083, %v1079 : tensor<256x512x7x7xf32>
    %v1085 = stablehlo.multiply %v1074, %v1082 : tensor<256x512x7x7xf32>
    %v1086 = stablehlo.subtract %v1084, %v1085 : tensor<256x512x7x7xf32>
    %v1087 = stablehlo.divide %v1073, %v1062 : tensor<256x512x7x7xf32>
    %v1088 = stablehlo.multiply %v1087, %v1086 : tensor<256x512x7x7xf32>
    %v1089 = stablehlo.reshape %v1088 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1091 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1092 = stablehlo.transpose %v1091, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1093 = stablehlo.convolution(%v1090, %v1092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1095 = stablehlo.add %v1094, %v1021 : tensor<256x25088xf32>
    %v1096 = stablehlo.reshape %v930 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1097 = stablehlo.reshape %v1089 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1098 = stablehlo.transpose %v1096, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1099 = stablehlo.transpose %v1097, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1100 = stablehlo.convolution(%v1098, %v1099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1101 = stablehlo.transpose %v1100, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1102 = stablehlo.reshape %v1089 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.reduce(%v1102 init: %v1103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1105 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1107 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1108 = stablehlo.reduce(%v1105 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1109 = stablehlo.broadcast_in_dim %v1108, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1110 = stablehlo.divide %v1109, %v1107 : tensor<256x512x7x7xf32>
    %v1111 = stablehlo.subtract %v1105, %v1110 : tensor<256x512x7x7xf32>
    %v1112 = stablehlo.multiply %v1111, %v1111 : tensor<256x512x7x7xf32>
    %v1113 = stablehlo.reduce(%v1112 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1114 = stablehlo.broadcast_in_dim %v1113, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1115 = stablehlo.divide %v1114, %v1107 : tensor<256x512x7x7xf32>
    %v1116 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<256x512x7x7xf32>
    %v1118 = stablehlo.rsqrt %v1117 : tensor<256x512x7x7xf32>
    %v1119 = stablehlo.multiply %v1111, %v1118 : tensor<256x512x7x7xf32>
    %v1120 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1121 = stablehlo.multiply %v1120, %v1119 : tensor<256x512x7x7xf32>
    %v1122 = stablehlo.reduce(%v1121 init: %v1106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1123 = stablehlo.reshape %v1059 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1125 = stablehlo.reduce(%v1123 init: %v1124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1126 = stablehlo.reshape %v957 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1127 = stablehlo.reshape %v1051 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1128 = stablehlo.transpose %v1126, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1129 = stablehlo.transpose %v1127, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1130 = stablehlo.convolution(%v1128, %v1129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1131 = stablehlo.transpose %v1130, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1132 = stablehlo.reshape %v1051 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1134 = stablehlo.reduce(%v1132 init: %v1133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1135 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1137 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1138 = stablehlo.reduce(%v1135 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1139 = stablehlo.broadcast_in_dim %v1138, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1140 = stablehlo.divide %v1139, %v1137 : tensor<256x512x7x7xf32>
    %v1141 = stablehlo.subtract %v1135, %v1140 : tensor<256x512x7x7xf32>
    %v1142 = stablehlo.multiply %v1141, %v1141 : tensor<256x512x7x7xf32>
    %v1143 = stablehlo.reduce(%v1142 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1144 = stablehlo.broadcast_in_dim %v1143, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1145 = stablehlo.divide %v1144, %v1137 : tensor<256x512x7x7xf32>
    %v1146 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<256x512x7x7xf32>
    %v1148 = stablehlo.rsqrt %v1147 : tensor<256x512x7x7xf32>
    %v1149 = stablehlo.multiply %v1141, %v1148 : tensor<256x512x7x7xf32>
    %v1150 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1151 = stablehlo.multiply %v1150, %v1149 : tensor<256x512x7x7xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1153 = stablehlo.reshape %v1021 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1155 = stablehlo.reduce(%v1153 init: %v1154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1157 = stablehlo.compare GT, %v928, %v1156 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1158 = stablehlo.select %v1157, %v1095, %v1156 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1159 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1161 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1162 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1163 = stablehlo.reduce(%v1159 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1164 = stablehlo.broadcast_in_dim %v1163, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1165 = stablehlo.divide %v1164, %v1161 : tensor<256x512x7x7xf32>
    %v1166 = stablehlo.subtract %v1159, %v1165 : tensor<256x512x7x7xf32>
    %v1167 = stablehlo.multiply %v1166, %v1166 : tensor<256x512x7x7xf32>
    %v1168 = stablehlo.reduce(%v1167 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1169 = stablehlo.broadcast_in_dim %v1168, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1170 = stablehlo.divide %v1169, %v1161 : tensor<256x512x7x7xf32>
    %v1171 = stablehlo.add %v1170, %v1162 : tensor<256x512x7x7xf32>
    %v1172 = stablehlo.rsqrt %v1171 : tensor<256x512x7x7xf32>
    %v1173 = stablehlo.multiply %v1166, %v1172 : tensor<256x512x7x7xf32>
    %v1174 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1175 = stablehlo.reshape %v1158 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1176 = stablehlo.multiply %v1174, %v1175 : tensor<256x512x7x7xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1178 = stablehlo.broadcast_in_dim %v1177, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1179 = stablehlo.multiply %v1173, %v1176 : tensor<256x512x7x7xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1182 = stablehlo.multiply %v1176, %v1161 : tensor<256x512x7x7xf32>
    %v1183 = stablehlo.subtract %v1182, %v1178 : tensor<256x512x7x7xf32>
    %v1184 = stablehlo.multiply %v1173, %v1181 : tensor<256x512x7x7xf32>
    %v1185 = stablehlo.subtract %v1183, %v1184 : tensor<256x512x7x7xf32>
    %v1186 = stablehlo.divide %v1172, %v1161 : tensor<256x512x7x7xf32>
    %v1187 = stablehlo.multiply %v1186, %v1185 : tensor<256x512x7x7xf32>
    %v1188 = stablehlo.reshape %v1187 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1189 = stablehlo.reshape %v1188 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1190 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1191 = stablehlo.transpose %v1190, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1192 = stablehlo.convolution(%v1189, %v1191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1193 = stablehlo.reshape %v1192 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1194 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1195 = stablehlo.compare GT, %v900, %v1194 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1196 = stablehlo.select %v1195, %v1193, %v1194 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1197 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1199 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1200 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1201 = stablehlo.reduce(%v1197 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1203 = stablehlo.divide %v1202, %v1199 : tensor<256x512x7x7xf32>
    %v1204 = stablehlo.subtract %v1197, %v1203 : tensor<256x512x7x7xf32>
    %v1205 = stablehlo.multiply %v1204, %v1204 : tensor<256x512x7x7xf32>
    %v1206 = stablehlo.reduce(%v1205 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1206, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1208 = stablehlo.divide %v1207, %v1199 : tensor<256x512x7x7xf32>
    %v1209 = stablehlo.add %v1208, %v1200 : tensor<256x512x7x7xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<256x512x7x7xf32>
    %v1211 = stablehlo.multiply %v1204, %v1210 : tensor<256x512x7x7xf32>
    %v1212 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1213 = stablehlo.reshape %v1196 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1214 = stablehlo.multiply %v1212, %v1213 : tensor<256x512x7x7xf32>
    %v1215 = stablehlo.reduce(%v1214 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1216 = stablehlo.broadcast_in_dim %v1215, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1217 = stablehlo.multiply %v1211, %v1214 : tensor<256x512x7x7xf32>
    %v1218 = stablehlo.reduce(%v1217 init: %v1198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1219 = stablehlo.broadcast_in_dim %v1218, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1220 = stablehlo.multiply %v1214, %v1199 : tensor<256x512x7x7xf32>
    %v1221 = stablehlo.subtract %v1220, %v1216 : tensor<256x512x7x7xf32>
    %v1222 = stablehlo.multiply %v1211, %v1219 : tensor<256x512x7x7xf32>
    %v1223 = stablehlo.subtract %v1221, %v1222 : tensor<256x512x7x7xf32>
    %v1224 = stablehlo.divide %v1210, %v1199 : tensor<256x512x7x7xf32>
    %v1225 = stablehlo.multiply %v1224, %v1223 : tensor<256x512x7x7xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1228 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1229 = stablehlo.transpose %v1228, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1230 = stablehlo.convolution(%v1227, %v1229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1232 = stablehlo.add %v1231, %v1158 : tensor<256x25088xf32>
    %v1233 = stablehlo.reshape %v875 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1234 = stablehlo.reshape %v1226 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1235 = stablehlo.transpose %v1233, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1236 = stablehlo.transpose %v1234, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1237 = stablehlo.convolution(%v1235, %v1236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1238 = stablehlo.transpose %v1237, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1239 = stablehlo.reshape %v1226 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1241 = stablehlo.reduce(%v1239 init: %v1240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1242 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1244 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1245 = stablehlo.reduce(%v1242 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1246 = stablehlo.broadcast_in_dim %v1245, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1247 = stablehlo.divide %v1246, %v1244 : tensor<256x512x7x7xf32>
    %v1248 = stablehlo.subtract %v1242, %v1247 : tensor<256x512x7x7xf32>
    %v1249 = stablehlo.multiply %v1248, %v1248 : tensor<256x512x7x7xf32>
    %v1250 = stablehlo.reduce(%v1249 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1251 = stablehlo.broadcast_in_dim %v1250, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1252 = stablehlo.divide %v1251, %v1244 : tensor<256x512x7x7xf32>
    %v1253 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<256x512x7x7xf32>
    %v1255 = stablehlo.rsqrt %v1254 : tensor<256x512x7x7xf32>
    %v1256 = stablehlo.multiply %v1248, %v1255 : tensor<256x512x7x7xf32>
    %v1257 = stablehlo.reshape %v1196 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<256x512x7x7xf32>
    %v1259 = stablehlo.reduce(%v1258 init: %v1243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1260 = stablehlo.reshape %v1196 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1263 = stablehlo.reshape %v902 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1264 = stablehlo.reshape %v1188 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1265 = stablehlo.transpose %v1263, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1266 = stablehlo.transpose %v1264, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1267 = stablehlo.convolution(%v1265, %v1266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1268 = stablehlo.transpose %v1267, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1269 = stablehlo.reshape %v1188 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1271 = stablehlo.reduce(%v1269 init: %v1270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1272 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1274 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1275 = stablehlo.reduce(%v1272 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1276 = stablehlo.broadcast_in_dim %v1275, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1277 = stablehlo.divide %v1276, %v1274 : tensor<256x512x7x7xf32>
    %v1278 = stablehlo.subtract %v1272, %v1277 : tensor<256x512x7x7xf32>
    %v1279 = stablehlo.multiply %v1278, %v1278 : tensor<256x512x7x7xf32>
    %v1280 = stablehlo.reduce(%v1279 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1281 = stablehlo.broadcast_in_dim %v1280, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1282 = stablehlo.divide %v1281, %v1274 : tensor<256x512x7x7xf32>
    %v1283 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1284 = stablehlo.add %v1282, %v1283 : tensor<256x512x7x7xf32>
    %v1285 = stablehlo.rsqrt %v1284 : tensor<256x512x7x7xf32>
    %v1286 = stablehlo.multiply %v1278, %v1285 : tensor<256x512x7x7xf32>
    %v1287 = stablehlo.reshape %v1158 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1288 = stablehlo.multiply %v1287, %v1286 : tensor<256x512x7x7xf32>
    %v1289 = stablehlo.reduce(%v1288 init: %v1273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1290 = stablehlo.reshape %v1158 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1292 = stablehlo.reduce(%v1290 init: %v1291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1294 = stablehlo.compare GT, %v873, %v1293 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1295 = stablehlo.select %v1294, %v1232, %v1293 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1296 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1299 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1300 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1302 = stablehlo.divide %v1301, %v1298 : tensor<256x512x7x7xf32>
    %v1303 = stablehlo.subtract %v1296, %v1302 : tensor<256x512x7x7xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<256x512x7x7xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1307 = stablehlo.divide %v1306, %v1298 : tensor<256x512x7x7xf32>
    %v1308 = stablehlo.add %v1307, %v1299 : tensor<256x512x7x7xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<256x512x7x7xf32>
    %v1310 = stablehlo.multiply %v1303, %v1309 : tensor<256x512x7x7xf32>
    %v1311 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1312 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1313 = stablehlo.multiply %v1311, %v1312 : tensor<256x512x7x7xf32>
    %v1314 = stablehlo.reduce(%v1313 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1315 = stablehlo.broadcast_in_dim %v1314, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1316 = stablehlo.multiply %v1310, %v1313 : tensor<256x512x7x7xf32>
    %v1317 = stablehlo.reduce(%v1316 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1318 = stablehlo.broadcast_in_dim %v1317, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1319 = stablehlo.multiply %v1313, %v1298 : tensor<256x512x7x7xf32>
    %v1320 = stablehlo.subtract %v1319, %v1315 : tensor<256x512x7x7xf32>
    %v1321 = stablehlo.multiply %v1310, %v1318 : tensor<256x512x7x7xf32>
    %v1322 = stablehlo.subtract %v1320, %v1321 : tensor<256x512x7x7xf32>
    %v1323 = stablehlo.divide %v1309, %v1298 : tensor<256x512x7x7xf32>
    %v1324 = stablehlo.multiply %v1323, %v1322 : tensor<256x512x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1326 = stablehlo.reshape %v1325 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1327 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1328 = stablehlo.transpose %v1327, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1329 = stablehlo.convolution(%v1326, %v1328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1331 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1332 = stablehlo.compare GT, %v820, %v1331 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1333 = stablehlo.select %v1332, %v1330, %v1331 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1334 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1336 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1337 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1338 = stablehlo.reduce(%v1334 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1339 = stablehlo.broadcast_in_dim %v1338, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1340 = stablehlo.divide %v1339, %v1336 : tensor<256x512x7x7xf32>
    %v1341 = stablehlo.subtract %v1334, %v1340 : tensor<256x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1341, %v1341 : tensor<256x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1342 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1345 = stablehlo.divide %v1344, %v1336 : tensor<256x512x7x7xf32>
    %v1346 = stablehlo.add %v1345, %v1337 : tensor<256x512x7x7xf32>
    %v1347 = stablehlo.rsqrt %v1346 : tensor<256x512x7x7xf32>
    %v1348 = stablehlo.multiply %v1341, %v1347 : tensor<256x512x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1350 = stablehlo.reshape %v1333 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1351 = stablehlo.multiply %v1349, %v1350 : tensor<256x512x7x7xf32>
    %v1352 = stablehlo.reduce(%v1351 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1353 = stablehlo.broadcast_in_dim %v1352, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1354 = stablehlo.multiply %v1348, %v1351 : tensor<256x512x7x7xf32>
    %v1355 = stablehlo.reduce(%v1354 init: %v1335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1356 = stablehlo.broadcast_in_dim %v1355, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1357 = stablehlo.multiply %v1351, %v1336 : tensor<256x512x7x7xf32>
    %v1358 = stablehlo.subtract %v1357, %v1353 : tensor<256x512x7x7xf32>
    %v1359 = stablehlo.multiply %v1348, %v1356 : tensor<256x512x7x7xf32>
    %v1360 = stablehlo.subtract %v1358, %v1359 : tensor<256x512x7x7xf32>
    %v1361 = stablehlo.divide %v1347, %v1336 : tensor<256x512x7x7xf32>
    %v1362 = stablehlo.multiply %v1361, %v1360 : tensor<256x512x7x7xf32>
    %v1363 = stablehlo.reshape %v1362 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.pad %v1364, %v1365, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1367 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1368 = stablehlo.transpose %v1367, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1369 = stablehlo.convolution(%v1366, %v1368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1370 = stablehlo.reshape %v1369 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1371 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1373 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1374 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1375 = stablehlo.reduce(%v1371 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1377 = stablehlo.divide %v1376, %v1373 : tensor<256x512x7x7xf32>
    %v1378 = stablehlo.subtract %v1371, %v1377 : tensor<256x512x7x7xf32>
    %v1379 = stablehlo.multiply %v1378, %v1378 : tensor<256x512x7x7xf32>
    %v1380 = stablehlo.reduce(%v1379 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1382 = stablehlo.divide %v1381, %v1373 : tensor<256x512x7x7xf32>
    %v1383 = stablehlo.add %v1382, %v1374 : tensor<256x512x7x7xf32>
    %v1384 = stablehlo.rsqrt %v1383 : tensor<256x512x7x7xf32>
    %v1385 = stablehlo.multiply %v1378, %v1384 : tensor<256x512x7x7xf32>
    %v1386 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1387 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1388 = stablehlo.multiply %v1386, %v1387 : tensor<256x512x7x7xf32>
    %v1389 = stablehlo.reduce(%v1388 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1390 = stablehlo.broadcast_in_dim %v1389, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1391 = stablehlo.multiply %v1385, %v1388 : tensor<256x512x7x7xf32>
    %v1392 = stablehlo.reduce(%v1391 init: %v1372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1393 = stablehlo.broadcast_in_dim %v1392, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1394 = stablehlo.multiply %v1388, %v1373 : tensor<256x512x7x7xf32>
    %v1395 = stablehlo.subtract %v1394, %v1390 : tensor<256x512x7x7xf32>
    %v1396 = stablehlo.multiply %v1385, %v1393 : tensor<256x512x7x7xf32>
    %v1397 = stablehlo.subtract %v1395, %v1396 : tensor<256x512x7x7xf32>
    %v1398 = stablehlo.divide %v1384, %v1373 : tensor<256x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1397 : tensor<256x512x7x7xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.pad %v1401, %v1402, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1404 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1405 = stablehlo.transpose %v1404, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1406 = stablehlo.convolution(%v1403, %v1405)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1408 = stablehlo.add %v1370, %v1407 : tensor<256x50176xf32>
    %v1409 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1410 = stablehlo.reshape %v1363 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.pad %v1410, %v1411, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1413 = stablehlo.transpose %v1409, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1414 = stablehlo.transpose %v1412, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v1415 = stablehlo.convolution(%v1413, %v1414)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1416 = stablehlo.transpose %v1415, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1417 = stablehlo.reshape %v1363 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1419 = stablehlo.reduce(%v1417 init: %v1418) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1420 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1423 = stablehlo.reduce(%v1420 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1424 = stablehlo.broadcast_in_dim %v1423, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1425 = stablehlo.divide %v1424, %v1422 : tensor<256x512x7x7xf32>
    %v1426 = stablehlo.subtract %v1420, %v1425 : tensor<256x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1426, %v1426 : tensor<256x512x7x7xf32>
    %v1428 = stablehlo.reduce(%v1427 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1429 = stablehlo.broadcast_in_dim %v1428, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1430 = stablehlo.divide %v1429, %v1422 : tensor<256x512x7x7xf32>
    %v1431 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1432 = stablehlo.add %v1430, %v1431 : tensor<256x512x7x7xf32>
    %v1433 = stablehlo.rsqrt %v1432 : tensor<256x512x7x7xf32>
    %v1434 = stablehlo.multiply %v1426, %v1433 : tensor<256x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1333 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1435, %v1434 : tensor<256x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.reshape %v1333 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1441 = stablehlo.reshape %v822 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1442 = stablehlo.reshape %v1325 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1443 = stablehlo.transpose %v1441, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1444 = stablehlo.transpose %v1442, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1445 = stablehlo.convolution(%v1443, %v1444)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1446 = stablehlo.transpose %v1445, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1447 = stablehlo.reshape %v1325 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1449 = stablehlo.reduce(%v1447 init: %v1448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1450 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1451 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1452 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1453 = stablehlo.reduce(%v1450 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1454 = stablehlo.broadcast_in_dim %v1453, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1455 = stablehlo.divide %v1454, %v1452 : tensor<256x512x7x7xf32>
    %v1456 = stablehlo.subtract %v1450, %v1455 : tensor<256x512x7x7xf32>
    %v1457 = stablehlo.multiply %v1456, %v1456 : tensor<256x512x7x7xf32>
    %v1458 = stablehlo.reduce(%v1457 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1459 = stablehlo.broadcast_in_dim %v1458, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1460 = stablehlo.divide %v1459, %v1452 : tensor<256x512x7x7xf32>
    %v1461 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1462 = stablehlo.add %v1460, %v1461 : tensor<256x512x7x7xf32>
    %v1463 = stablehlo.rsqrt %v1462 : tensor<256x512x7x7xf32>
    %v1464 = stablehlo.multiply %v1456, %v1463 : tensor<256x512x7x7xf32>
    %v1465 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1466 = stablehlo.multiply %v1465, %v1464 : tensor<256x512x7x7xf32>
    %v1467 = stablehlo.reduce(%v1466 init: %v1451) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1468 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1470 = stablehlo.reduce(%v1468 init: %v1469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1471 = stablehlo.reshape %v795 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1472 = stablehlo.reshape %v1400 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1474 = stablehlo.pad %v1472, %v1473, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1475 = stablehlo.transpose %v1471, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1476 = stablehlo.transpose %v1474, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v1477 = stablehlo.convolution(%v1475, %v1476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<256x512x3x3xf32>
    %v1478 = stablehlo.transpose %v1477, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1479 = stablehlo.reshape %v1400 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1482 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1484 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1485 = stablehlo.reduce(%v1482 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1486 = stablehlo.broadcast_in_dim %v1485, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1487 = stablehlo.divide %v1486, %v1484 : tensor<256x512x7x7xf32>
    %v1488 = stablehlo.subtract %v1482, %v1487 : tensor<256x512x7x7xf32>
    %v1489 = stablehlo.multiply %v1488, %v1488 : tensor<256x512x7x7xf32>
    %v1490 = stablehlo.reduce(%v1489 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1491 = stablehlo.broadcast_in_dim %v1490, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1492 = stablehlo.divide %v1491, %v1484 : tensor<256x512x7x7xf32>
    %v1493 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1494 = stablehlo.add %v1492, %v1493 : tensor<256x512x7x7xf32>
    %v1495 = stablehlo.rsqrt %v1494 : tensor<256x512x7x7xf32>
    %v1496 = stablehlo.multiply %v1488, %v1495 : tensor<256x512x7x7xf32>
    %v1497 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1498 = stablehlo.multiply %v1497, %v1496 : tensor<256x512x7x7xf32>
    %v1499 = stablehlo.reduce(%v1498 init: %v1483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1500 = stablehlo.reshape %v1295 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1502 = stablehlo.reduce(%v1500 init: %v1501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1504 = stablehlo.compare GT, %v793, %v1503 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1505 = stablehlo.select %v1504, %v1408, %v1503 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1506 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1508 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1509 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1510 = stablehlo.reduce(%v1506 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1511 = stablehlo.broadcast_in_dim %v1510, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1512 = stablehlo.divide %v1511, %v1508 : tensor<256x256x14x14xf32>
    %v1513 = stablehlo.subtract %v1506, %v1512 : tensor<256x256x14x14xf32>
    %v1514 = stablehlo.multiply %v1513, %v1513 : tensor<256x256x14x14xf32>
    %v1515 = stablehlo.reduce(%v1514 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1517 = stablehlo.divide %v1516, %v1508 : tensor<256x256x14x14xf32>
    %v1518 = stablehlo.add %v1517, %v1509 : tensor<256x256x14x14xf32>
    %v1519 = stablehlo.rsqrt %v1518 : tensor<256x256x14x14xf32>
    %v1520 = stablehlo.multiply %v1513, %v1519 : tensor<256x256x14x14xf32>
    %v1521 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1522 = stablehlo.reshape %v1505 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1523 = stablehlo.multiply %v1521, %v1522 : tensor<256x256x14x14xf32>
    %v1524 = stablehlo.reduce(%v1523 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1525 = stablehlo.broadcast_in_dim %v1524, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1526 = stablehlo.multiply %v1520, %v1523 : tensor<256x256x14x14xf32>
    %v1527 = stablehlo.reduce(%v1526 init: %v1507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1528 = stablehlo.broadcast_in_dim %v1527, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1529 = stablehlo.multiply %v1523, %v1508 : tensor<256x256x14x14xf32>
    %v1530 = stablehlo.subtract %v1529, %v1525 : tensor<256x256x14x14xf32>
    %v1531 = stablehlo.multiply %v1520, %v1528 : tensor<256x256x14x14xf32>
    %v1532 = stablehlo.subtract %v1530, %v1531 : tensor<256x256x14x14xf32>
    %v1533 = stablehlo.divide %v1519, %v1508 : tensor<256x256x14x14xf32>
    %v1534 = stablehlo.multiply %v1533, %v1532 : tensor<256x256x14x14xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1537 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1538 = stablehlo.transpose %v1537, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1539 = stablehlo.convolution(%v1536, %v1538)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1540 = stablehlo.reshape %v1539 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1541 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1542 = stablehlo.compare GT, %v765, %v1541 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1543 = stablehlo.select %v1542, %v1540, %v1541 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1544 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1546 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1547 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1548 = stablehlo.reduce(%v1544 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1550 = stablehlo.divide %v1549, %v1546 : tensor<256x256x14x14xf32>
    %v1551 = stablehlo.subtract %v1544, %v1550 : tensor<256x256x14x14xf32>
    %v1552 = stablehlo.multiply %v1551, %v1551 : tensor<256x256x14x14xf32>
    %v1553 = stablehlo.reduce(%v1552 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1555 = stablehlo.divide %v1554, %v1546 : tensor<256x256x14x14xf32>
    %v1556 = stablehlo.add %v1555, %v1547 : tensor<256x256x14x14xf32>
    %v1557 = stablehlo.rsqrt %v1556 : tensor<256x256x14x14xf32>
    %v1558 = stablehlo.multiply %v1551, %v1557 : tensor<256x256x14x14xf32>
    %v1559 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1560 = stablehlo.reshape %v1543 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1561 = stablehlo.multiply %v1559, %v1560 : tensor<256x256x14x14xf32>
    %v1562 = stablehlo.reduce(%v1561 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1563 = stablehlo.broadcast_in_dim %v1562, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1564 = stablehlo.multiply %v1558, %v1561 : tensor<256x256x14x14xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1567 = stablehlo.multiply %v1561, %v1546 : tensor<256x256x14x14xf32>
    %v1568 = stablehlo.subtract %v1567, %v1563 : tensor<256x256x14x14xf32>
    %v1569 = stablehlo.multiply %v1558, %v1566 : tensor<256x256x14x14xf32>
    %v1570 = stablehlo.subtract %v1568, %v1569 : tensor<256x256x14x14xf32>
    %v1571 = stablehlo.divide %v1557, %v1546 : tensor<256x256x14x14xf32>
    %v1572 = stablehlo.multiply %v1571, %v1570 : tensor<256x256x14x14xf32>
    %v1573 = stablehlo.reshape %v1572 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1575 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1576 = stablehlo.transpose %v1575, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1577 = stablehlo.convolution(%v1574, %v1576)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1579 = stablehlo.add %v1578, %v1505 : tensor<256x50176xf32>
    %v1580 = stablehlo.reshape %v740 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1581 = stablehlo.reshape %v1573 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1582 = stablehlo.transpose %v1580, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1583 = stablehlo.transpose %v1581, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1584 = stablehlo.convolution(%v1582, %v1583)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1585 = stablehlo.transpose %v1584, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1586 = stablehlo.reshape %v1573 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1588 = stablehlo.reduce(%v1586 init: %v1587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1589 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1592 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1592, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1594 = stablehlo.divide %v1593, %v1591 : tensor<256x256x14x14xf32>
    %v1595 = stablehlo.subtract %v1589, %v1594 : tensor<256x256x14x14xf32>
    %v1596 = stablehlo.multiply %v1595, %v1595 : tensor<256x256x14x14xf32>
    %v1597 = stablehlo.reduce(%v1596 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1597, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1599 = stablehlo.divide %v1598, %v1591 : tensor<256x256x14x14xf32>
    %v1600 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1601 = stablehlo.add %v1599, %v1600 : tensor<256x256x14x14xf32>
    %v1602 = stablehlo.rsqrt %v1601 : tensor<256x256x14x14xf32>
    %v1603 = stablehlo.multiply %v1595, %v1602 : tensor<256x256x14x14xf32>
    %v1604 = stablehlo.reshape %v1543 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1605 = stablehlo.multiply %v1604, %v1603 : tensor<256x256x14x14xf32>
    %v1606 = stablehlo.reduce(%v1605 init: %v1590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1607 = stablehlo.reshape %v1543 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1609 = stablehlo.reduce(%v1607 init: %v1608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1610 = stablehlo.reshape %v767 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1611 = stablehlo.reshape %v1535 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1612 = stablehlo.transpose %v1610, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1613 = stablehlo.transpose %v1611, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1614 = stablehlo.convolution(%v1612, %v1613)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1615 = stablehlo.transpose %v1614, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1616 = stablehlo.reshape %v1535 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1618 = stablehlo.reduce(%v1616 init: %v1617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1619 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1621 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1622 = stablehlo.reduce(%v1619 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1623 = stablehlo.broadcast_in_dim %v1622, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1624 = stablehlo.divide %v1623, %v1621 : tensor<256x256x14x14xf32>
    %v1625 = stablehlo.subtract %v1619, %v1624 : tensor<256x256x14x14xf32>
    %v1626 = stablehlo.multiply %v1625, %v1625 : tensor<256x256x14x14xf32>
    %v1627 = stablehlo.reduce(%v1626 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1629 = stablehlo.divide %v1628, %v1621 : tensor<256x256x14x14xf32>
    %v1630 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1631 = stablehlo.add %v1629, %v1630 : tensor<256x256x14x14xf32>
    %v1632 = stablehlo.rsqrt %v1631 : tensor<256x256x14x14xf32>
    %v1633 = stablehlo.multiply %v1625, %v1632 : tensor<256x256x14x14xf32>
    %v1634 = stablehlo.reshape %v1505 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1635 = stablehlo.multiply %v1634, %v1633 : tensor<256x256x14x14xf32>
    %v1636 = stablehlo.reduce(%v1635 init: %v1620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1637 = stablehlo.reshape %v1505 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1639 = stablehlo.reduce(%v1637 init: %v1638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1640 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1641 = stablehlo.compare GT, %v738, %v1640 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1642 = stablehlo.select %v1641, %v1579, %v1640 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1643 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1645 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1646 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1647 = stablehlo.reduce(%v1643 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1649 = stablehlo.divide %v1648, %v1645 : tensor<256x256x14x14xf32>
    %v1650 = stablehlo.subtract %v1643, %v1649 : tensor<256x256x14x14xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<256x256x14x14xf32>
    %v1652 = stablehlo.reduce(%v1651 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1653 = stablehlo.broadcast_in_dim %v1652, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1654 = stablehlo.divide %v1653, %v1645 : tensor<256x256x14x14xf32>
    %v1655 = stablehlo.add %v1654, %v1646 : tensor<256x256x14x14xf32>
    %v1656 = stablehlo.rsqrt %v1655 : tensor<256x256x14x14xf32>
    %v1657 = stablehlo.multiply %v1650, %v1656 : tensor<256x256x14x14xf32>
    %v1658 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1659 = stablehlo.reshape %v1642 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1660 = stablehlo.multiply %v1658, %v1659 : tensor<256x256x14x14xf32>
    %v1661 = stablehlo.reduce(%v1660 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1662 = stablehlo.broadcast_in_dim %v1661, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1663 = stablehlo.multiply %v1657, %v1660 : tensor<256x256x14x14xf32>
    %v1664 = stablehlo.reduce(%v1663 init: %v1644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1665 = stablehlo.broadcast_in_dim %v1664, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1666 = stablehlo.multiply %v1660, %v1645 : tensor<256x256x14x14xf32>
    %v1667 = stablehlo.subtract %v1666, %v1662 : tensor<256x256x14x14xf32>
    %v1668 = stablehlo.multiply %v1657, %v1665 : tensor<256x256x14x14xf32>
    %v1669 = stablehlo.subtract %v1667, %v1668 : tensor<256x256x14x14xf32>
    %v1670 = stablehlo.divide %v1656, %v1645 : tensor<256x256x14x14xf32>
    %v1671 = stablehlo.multiply %v1670, %v1669 : tensor<256x256x14x14xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1674 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1675 = stablehlo.transpose %v1674, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1676 = stablehlo.convolution(%v1673, %v1675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1678 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1679 = stablehlo.compare GT, %v710, %v1678 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1680 = stablehlo.select %v1679, %v1677, %v1678 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1681 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1683 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1684 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1681 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1687 = stablehlo.divide %v1686, %v1683 : tensor<256x256x14x14xf32>
    %v1688 = stablehlo.subtract %v1681, %v1687 : tensor<256x256x14x14xf32>
    %v1689 = stablehlo.multiply %v1688, %v1688 : tensor<256x256x14x14xf32>
    %v1690 = stablehlo.reduce(%v1689 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1691 = stablehlo.broadcast_in_dim %v1690, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1692 = stablehlo.divide %v1691, %v1683 : tensor<256x256x14x14xf32>
    %v1693 = stablehlo.add %v1692, %v1684 : tensor<256x256x14x14xf32>
    %v1694 = stablehlo.rsqrt %v1693 : tensor<256x256x14x14xf32>
    %v1695 = stablehlo.multiply %v1688, %v1694 : tensor<256x256x14x14xf32>
    %v1696 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1697 = stablehlo.reshape %v1680 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1698 = stablehlo.multiply %v1696, %v1697 : tensor<256x256x14x14xf32>
    %v1699 = stablehlo.reduce(%v1698 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1700 = stablehlo.broadcast_in_dim %v1699, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1701 = stablehlo.multiply %v1695, %v1698 : tensor<256x256x14x14xf32>
    %v1702 = stablehlo.reduce(%v1701 init: %v1682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1703 = stablehlo.broadcast_in_dim %v1702, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1704 = stablehlo.multiply %v1698, %v1683 : tensor<256x256x14x14xf32>
    %v1705 = stablehlo.subtract %v1704, %v1700 : tensor<256x256x14x14xf32>
    %v1706 = stablehlo.multiply %v1695, %v1703 : tensor<256x256x14x14xf32>
    %v1707 = stablehlo.subtract %v1705, %v1706 : tensor<256x256x14x14xf32>
    %v1708 = stablehlo.divide %v1694, %v1683 : tensor<256x256x14x14xf32>
    %v1709 = stablehlo.multiply %v1708, %v1707 : tensor<256x256x14x14xf32>
    %v1710 = stablehlo.reshape %v1709 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1712 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1713 = stablehlo.transpose %v1712, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1714 = stablehlo.convolution(%v1711, %v1713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1715 = stablehlo.reshape %v1714 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1716 = stablehlo.add %v1715, %v1642 : tensor<256x50176xf32>
    %v1717 = stablehlo.reshape %v685 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1718 = stablehlo.reshape %v1710 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1719 = stablehlo.transpose %v1717, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1720 = stablehlo.transpose %v1718, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1721 = stablehlo.convolution(%v1719, %v1720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1722 = stablehlo.transpose %v1721, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1723 = stablehlo.reshape %v1710 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1725 = stablehlo.reduce(%v1723 init: %v1724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1726 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1728 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1729 = stablehlo.reduce(%v1726 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1729, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1731 = stablehlo.divide %v1730, %v1728 : tensor<256x256x14x14xf32>
    %v1732 = stablehlo.subtract %v1726, %v1731 : tensor<256x256x14x14xf32>
    %v1733 = stablehlo.multiply %v1732, %v1732 : tensor<256x256x14x14xf32>
    %v1734 = stablehlo.reduce(%v1733 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1734, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1736 = stablehlo.divide %v1735, %v1728 : tensor<256x256x14x14xf32>
    %v1737 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1738 = stablehlo.add %v1736, %v1737 : tensor<256x256x14x14xf32>
    %v1739 = stablehlo.rsqrt %v1738 : tensor<256x256x14x14xf32>
    %v1740 = stablehlo.multiply %v1732, %v1739 : tensor<256x256x14x14xf32>
    %v1741 = stablehlo.reshape %v1680 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1742 = stablehlo.multiply %v1741, %v1740 : tensor<256x256x14x14xf32>
    %v1743 = stablehlo.reduce(%v1742 init: %v1727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1744 = stablehlo.reshape %v1680 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1745 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1746 = stablehlo.reduce(%v1744 init: %v1745) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1747 = stablehlo.reshape %v712 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1748 = stablehlo.reshape %v1672 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1749 = stablehlo.transpose %v1747, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1750 = stablehlo.transpose %v1748, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1751 = stablehlo.convolution(%v1749, %v1750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1752 = stablehlo.transpose %v1751, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1753 = stablehlo.reshape %v1672 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1755 = stablehlo.reduce(%v1753 init: %v1754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1756 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1758 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1759 = stablehlo.reduce(%v1756 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1760 = stablehlo.broadcast_in_dim %v1759, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1761 = stablehlo.divide %v1760, %v1758 : tensor<256x256x14x14xf32>
    %v1762 = stablehlo.subtract %v1756, %v1761 : tensor<256x256x14x14xf32>
    %v1763 = stablehlo.multiply %v1762, %v1762 : tensor<256x256x14x14xf32>
    %v1764 = stablehlo.reduce(%v1763 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1765 = stablehlo.broadcast_in_dim %v1764, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1766 = stablehlo.divide %v1765, %v1758 : tensor<256x256x14x14xf32>
    %v1767 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1768 = stablehlo.add %v1766, %v1767 : tensor<256x256x14x14xf32>
    %v1769 = stablehlo.rsqrt %v1768 : tensor<256x256x14x14xf32>
    %v1770 = stablehlo.multiply %v1762, %v1769 : tensor<256x256x14x14xf32>
    %v1771 = stablehlo.reshape %v1642 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1772 = stablehlo.multiply %v1771, %v1770 : tensor<256x256x14x14xf32>
    %v1773 = stablehlo.reduce(%v1772 init: %v1757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1774 = stablehlo.reshape %v1642 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.reduce(%v1774 init: %v1775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1777 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1778 = stablehlo.compare GT, %v683, %v1777 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1779 = stablehlo.select %v1778, %v1716, %v1777 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1780 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1782 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1783 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1784 = stablehlo.reduce(%v1780 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1785 = stablehlo.broadcast_in_dim %v1784, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1786 = stablehlo.divide %v1785, %v1782 : tensor<256x256x14x14xf32>
    %v1787 = stablehlo.subtract %v1780, %v1786 : tensor<256x256x14x14xf32>
    %v1788 = stablehlo.multiply %v1787, %v1787 : tensor<256x256x14x14xf32>
    %v1789 = stablehlo.reduce(%v1788 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1789, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1791 = stablehlo.divide %v1790, %v1782 : tensor<256x256x14x14xf32>
    %v1792 = stablehlo.add %v1791, %v1783 : tensor<256x256x14x14xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<256x256x14x14xf32>
    %v1794 = stablehlo.multiply %v1787, %v1793 : tensor<256x256x14x14xf32>
    %v1795 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1796 = stablehlo.reshape %v1779 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1797 = stablehlo.multiply %v1795, %v1796 : tensor<256x256x14x14xf32>
    %v1798 = stablehlo.reduce(%v1797 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1799 = stablehlo.broadcast_in_dim %v1798, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1800 = stablehlo.multiply %v1794, %v1797 : tensor<256x256x14x14xf32>
    %v1801 = stablehlo.reduce(%v1800 init: %v1781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1803 = stablehlo.multiply %v1797, %v1782 : tensor<256x256x14x14xf32>
    %v1804 = stablehlo.subtract %v1803, %v1799 : tensor<256x256x14x14xf32>
    %v1805 = stablehlo.multiply %v1794, %v1802 : tensor<256x256x14x14xf32>
    %v1806 = stablehlo.subtract %v1804, %v1805 : tensor<256x256x14x14xf32>
    %v1807 = stablehlo.divide %v1793, %v1782 : tensor<256x256x14x14xf32>
    %v1808 = stablehlo.multiply %v1807, %v1806 : tensor<256x256x14x14xf32>
    %v1809 = stablehlo.reshape %v1808 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1811 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1812 = stablehlo.transpose %v1811, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1813 = stablehlo.convolution(%v1810, %v1812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1815 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1816 = stablehlo.compare GT, %v655, %v1815 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1817 = stablehlo.select %v1816, %v1814, %v1815 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1818 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1819 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1820 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1821 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1822 = stablehlo.reduce(%v1818 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1823 = stablehlo.broadcast_in_dim %v1822, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1824 = stablehlo.divide %v1823, %v1820 : tensor<256x256x14x14xf32>
    %v1825 = stablehlo.subtract %v1818, %v1824 : tensor<256x256x14x14xf32>
    %v1826 = stablehlo.multiply %v1825, %v1825 : tensor<256x256x14x14xf32>
    %v1827 = stablehlo.reduce(%v1826 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1828 = stablehlo.broadcast_in_dim %v1827, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1829 = stablehlo.divide %v1828, %v1820 : tensor<256x256x14x14xf32>
    %v1830 = stablehlo.add %v1829, %v1821 : tensor<256x256x14x14xf32>
    %v1831 = stablehlo.rsqrt %v1830 : tensor<256x256x14x14xf32>
    %v1832 = stablehlo.multiply %v1825, %v1831 : tensor<256x256x14x14xf32>
    %v1833 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1834 = stablehlo.reshape %v1817 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1835 = stablehlo.multiply %v1833, %v1834 : tensor<256x256x14x14xf32>
    %v1836 = stablehlo.reduce(%v1835 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1836, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1838 = stablehlo.multiply %v1832, %v1835 : tensor<256x256x14x14xf32>
    %v1839 = stablehlo.reduce(%v1838 init: %v1819) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1840 = stablehlo.broadcast_in_dim %v1839, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1841 = stablehlo.multiply %v1835, %v1820 : tensor<256x256x14x14xf32>
    %v1842 = stablehlo.subtract %v1841, %v1837 : tensor<256x256x14x14xf32>
    %v1843 = stablehlo.multiply %v1832, %v1840 : tensor<256x256x14x14xf32>
    %v1844 = stablehlo.subtract %v1842, %v1843 : tensor<256x256x14x14xf32>
    %v1845 = stablehlo.divide %v1831, %v1820 : tensor<256x256x14x14xf32>
    %v1846 = stablehlo.multiply %v1845, %v1844 : tensor<256x256x14x14xf32>
    %v1847 = stablehlo.reshape %v1846 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1849 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1850 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1851 = stablehlo.convolution(%v1848, %v1850)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1852 = stablehlo.reshape %v1851 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1853 = stablehlo.add %v1852, %v1779 : tensor<256x50176xf32>
    %v1854 = stablehlo.reshape %v630 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1855 = stablehlo.reshape %v1847 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1856 = stablehlo.transpose %v1854, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1857 = stablehlo.transpose %v1855, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1858 = stablehlo.convolution(%v1856, %v1857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1859 = stablehlo.transpose %v1858, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1860 = stablehlo.reshape %v1847 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1862 = stablehlo.reduce(%v1860 init: %v1861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1863 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1865 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1866 = stablehlo.reduce(%v1863 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1867 = stablehlo.broadcast_in_dim %v1866, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1868 = stablehlo.divide %v1867, %v1865 : tensor<256x256x14x14xf32>
    %v1869 = stablehlo.subtract %v1863, %v1868 : tensor<256x256x14x14xf32>
    %v1870 = stablehlo.multiply %v1869, %v1869 : tensor<256x256x14x14xf32>
    %v1871 = stablehlo.reduce(%v1870 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1872 = stablehlo.broadcast_in_dim %v1871, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1873 = stablehlo.divide %v1872, %v1865 : tensor<256x256x14x14xf32>
    %v1874 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1875 = stablehlo.add %v1873, %v1874 : tensor<256x256x14x14xf32>
    %v1876 = stablehlo.rsqrt %v1875 : tensor<256x256x14x14xf32>
    %v1877 = stablehlo.multiply %v1869, %v1876 : tensor<256x256x14x14xf32>
    %v1878 = stablehlo.reshape %v1817 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1879 = stablehlo.multiply %v1878, %v1877 : tensor<256x256x14x14xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1881 = stablehlo.reshape %v1817 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1883 = stablehlo.reduce(%v1881 init: %v1882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1884 = stablehlo.reshape %v657 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1885 = stablehlo.reshape %v1809 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1886 = stablehlo.transpose %v1884, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1887 = stablehlo.transpose %v1885, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1888 = stablehlo.convolution(%v1886, %v1887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1889 = stablehlo.transpose %v1888, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1890 = stablehlo.reshape %v1809 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1892 = stablehlo.reduce(%v1890 init: %v1891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1893 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1895 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1896 = stablehlo.reduce(%v1893 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1898 = stablehlo.divide %v1897, %v1895 : tensor<256x256x14x14xf32>
    %v1899 = stablehlo.subtract %v1893, %v1898 : tensor<256x256x14x14xf32>
    %v1900 = stablehlo.multiply %v1899, %v1899 : tensor<256x256x14x14xf32>
    %v1901 = stablehlo.reduce(%v1900 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1901, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1903 = stablehlo.divide %v1902, %v1895 : tensor<256x256x14x14xf32>
    %v1904 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1905 = stablehlo.add %v1903, %v1904 : tensor<256x256x14x14xf32>
    %v1906 = stablehlo.rsqrt %v1905 : tensor<256x256x14x14xf32>
    %v1907 = stablehlo.multiply %v1899, %v1906 : tensor<256x256x14x14xf32>
    %v1908 = stablehlo.reshape %v1779 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1909 = stablehlo.multiply %v1908, %v1907 : tensor<256x256x14x14xf32>
    %v1910 = stablehlo.reduce(%v1909 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1911 = stablehlo.reshape %v1779 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1913 = stablehlo.reduce(%v1911 init: %v1912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1914 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1915 = stablehlo.compare GT, %v628, %v1914 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1916 = stablehlo.select %v1915, %v1853, %v1914 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1917 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1919 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1920 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1921 = stablehlo.reduce(%v1917 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1922 = stablehlo.broadcast_in_dim %v1921, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1923 = stablehlo.divide %v1922, %v1919 : tensor<256x256x14x14xf32>
    %v1924 = stablehlo.subtract %v1917, %v1923 : tensor<256x256x14x14xf32>
    %v1925 = stablehlo.multiply %v1924, %v1924 : tensor<256x256x14x14xf32>
    %v1926 = stablehlo.reduce(%v1925 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1927 = stablehlo.broadcast_in_dim %v1926, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1928 = stablehlo.divide %v1927, %v1919 : tensor<256x256x14x14xf32>
    %v1929 = stablehlo.add %v1928, %v1920 : tensor<256x256x14x14xf32>
    %v1930 = stablehlo.rsqrt %v1929 : tensor<256x256x14x14xf32>
    %v1931 = stablehlo.multiply %v1924, %v1930 : tensor<256x256x14x14xf32>
    %v1932 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1933 = stablehlo.reshape %v1916 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1934 = stablehlo.multiply %v1932, %v1933 : tensor<256x256x14x14xf32>
    %v1935 = stablehlo.reduce(%v1934 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1935, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1937 = stablehlo.multiply %v1931, %v1934 : tensor<256x256x14x14xf32>
    %v1938 = stablehlo.reduce(%v1937 init: %v1918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1940 = stablehlo.multiply %v1934, %v1919 : tensor<256x256x14x14xf32>
    %v1941 = stablehlo.subtract %v1940, %v1936 : tensor<256x256x14x14xf32>
    %v1942 = stablehlo.multiply %v1931, %v1939 : tensor<256x256x14x14xf32>
    %v1943 = stablehlo.subtract %v1941, %v1942 : tensor<256x256x14x14xf32>
    %v1944 = stablehlo.divide %v1930, %v1919 : tensor<256x256x14x14xf32>
    %v1945 = stablehlo.multiply %v1944, %v1943 : tensor<256x256x14x14xf32>
    %v1946 = stablehlo.reshape %v1945 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1948 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1949 = stablehlo.transpose %v1948, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1950 = stablehlo.convolution(%v1947, %v1949)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1952 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1953 = stablehlo.compare GT, %v600, %v1952 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v1954 = stablehlo.select %v1953, %v1951, %v1952 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v1955 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1957 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1958 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1959 = stablehlo.reduce(%v1955 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1961 = stablehlo.divide %v1960, %v1957 : tensor<256x256x14x14xf32>
    %v1962 = stablehlo.subtract %v1955, %v1961 : tensor<256x256x14x14xf32>
    %v1963 = stablehlo.multiply %v1962, %v1962 : tensor<256x256x14x14xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1966 = stablehlo.divide %v1965, %v1957 : tensor<256x256x14x14xf32>
    %v1967 = stablehlo.add %v1966, %v1958 : tensor<256x256x14x14xf32>
    %v1968 = stablehlo.rsqrt %v1967 : tensor<256x256x14x14xf32>
    %v1969 = stablehlo.multiply %v1962, %v1968 : tensor<256x256x14x14xf32>
    %v1970 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1971 = stablehlo.reshape %v1954 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1972 = stablehlo.multiply %v1970, %v1971 : tensor<256x256x14x14xf32>
    %v1973 = stablehlo.reduce(%v1972 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1974 = stablehlo.broadcast_in_dim %v1973, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1975 = stablehlo.multiply %v1969, %v1972 : tensor<256x256x14x14xf32>
    %v1976 = stablehlo.reduce(%v1975 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1977 = stablehlo.broadcast_in_dim %v1976, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1978 = stablehlo.multiply %v1972, %v1957 : tensor<256x256x14x14xf32>
    %v1979 = stablehlo.subtract %v1978, %v1974 : tensor<256x256x14x14xf32>
    %v1980 = stablehlo.multiply %v1969, %v1977 : tensor<256x256x14x14xf32>
    %v1981 = stablehlo.subtract %v1979, %v1980 : tensor<256x256x14x14xf32>
    %v1982 = stablehlo.divide %v1968, %v1957 : tensor<256x256x14x14xf32>
    %v1983 = stablehlo.multiply %v1982, %v1981 : tensor<256x256x14x14xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1985 = stablehlo.reshape %v1984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1986 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1987 = stablehlo.transpose %v1986, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1988 = stablehlo.convolution(%v1985, %v1987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1989 = stablehlo.reshape %v1988 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1990 = stablehlo.add %v1989, %v1916 : tensor<256x50176xf32>
    %v1991 = stablehlo.reshape %v575 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1992 = stablehlo.reshape %v1984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1993 = stablehlo.transpose %v1991, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1994 = stablehlo.transpose %v1992, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v1995 = stablehlo.convolution(%v1993, %v1994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v1996 = stablehlo.transpose %v1995, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1997 = stablehlo.reshape %v1984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.reduce(%v1997 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2000 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2002 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2003 = stablehlo.reduce(%v2000 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2004 = stablehlo.broadcast_in_dim %v2003, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2005 = stablehlo.divide %v2004, %v2002 : tensor<256x256x14x14xf32>
    %v2006 = stablehlo.subtract %v2000, %v2005 : tensor<256x256x14x14xf32>
    %v2007 = stablehlo.multiply %v2006, %v2006 : tensor<256x256x14x14xf32>
    %v2008 = stablehlo.reduce(%v2007 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2009 = stablehlo.broadcast_in_dim %v2008, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2010 = stablehlo.divide %v2009, %v2002 : tensor<256x256x14x14xf32>
    %v2011 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2012 = stablehlo.add %v2010, %v2011 : tensor<256x256x14x14xf32>
    %v2013 = stablehlo.rsqrt %v2012 : tensor<256x256x14x14xf32>
    %v2014 = stablehlo.multiply %v2006, %v2013 : tensor<256x256x14x14xf32>
    %v2015 = stablehlo.reshape %v1954 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2016 = stablehlo.multiply %v2015, %v2014 : tensor<256x256x14x14xf32>
    %v2017 = stablehlo.reduce(%v2016 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2018 = stablehlo.reshape %v1954 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2020 = stablehlo.reduce(%v2018 init: %v2019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2021 = stablehlo.reshape %v602 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2022 = stablehlo.reshape %v1946 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2023 = stablehlo.transpose %v2021, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2024 = stablehlo.transpose %v2022, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2025 = stablehlo.convolution(%v2023, %v2024)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2026 = stablehlo.transpose %v2025, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2027 = stablehlo.reshape %v1946 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2029 = stablehlo.reduce(%v2027 init: %v2028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2030 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2033 = stablehlo.reduce(%v2030 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2034 = stablehlo.broadcast_in_dim %v2033, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2035 = stablehlo.divide %v2034, %v2032 : tensor<256x256x14x14xf32>
    %v2036 = stablehlo.subtract %v2030, %v2035 : tensor<256x256x14x14xf32>
    %v2037 = stablehlo.multiply %v2036, %v2036 : tensor<256x256x14x14xf32>
    %v2038 = stablehlo.reduce(%v2037 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2039 = stablehlo.broadcast_in_dim %v2038, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2040 = stablehlo.divide %v2039, %v2032 : tensor<256x256x14x14xf32>
    %v2041 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2042 = stablehlo.add %v2040, %v2041 : tensor<256x256x14x14xf32>
    %v2043 = stablehlo.rsqrt %v2042 : tensor<256x256x14x14xf32>
    %v2044 = stablehlo.multiply %v2036, %v2043 : tensor<256x256x14x14xf32>
    %v2045 = stablehlo.reshape %v1916 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2046 = stablehlo.multiply %v2045, %v2044 : tensor<256x256x14x14xf32>
    %v2047 = stablehlo.reduce(%v2046 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2048 = stablehlo.reshape %v1916 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2050 = stablehlo.reduce(%v2048 init: %v2049) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2051 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2052 = stablehlo.compare GT, %v573, %v2051 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2053 = stablehlo.select %v2052, %v1990, %v2051 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2054 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2056 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2057 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2058 = stablehlo.reduce(%v2054 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2060 = stablehlo.divide %v2059, %v2056 : tensor<256x256x14x14xf32>
    %v2061 = stablehlo.subtract %v2054, %v2060 : tensor<256x256x14x14xf32>
    %v2062 = stablehlo.multiply %v2061, %v2061 : tensor<256x256x14x14xf32>
    %v2063 = stablehlo.reduce(%v2062 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2064 = stablehlo.broadcast_in_dim %v2063, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2065 = stablehlo.divide %v2064, %v2056 : tensor<256x256x14x14xf32>
    %v2066 = stablehlo.add %v2065, %v2057 : tensor<256x256x14x14xf32>
    %v2067 = stablehlo.rsqrt %v2066 : tensor<256x256x14x14xf32>
    %v2068 = stablehlo.multiply %v2061, %v2067 : tensor<256x256x14x14xf32>
    %v2069 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2070 = stablehlo.reshape %v2053 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2071 = stablehlo.multiply %v2069, %v2070 : tensor<256x256x14x14xf32>
    %v2072 = stablehlo.reduce(%v2071 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2073 = stablehlo.broadcast_in_dim %v2072, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2074 = stablehlo.multiply %v2068, %v2071 : tensor<256x256x14x14xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2077 = stablehlo.multiply %v2071, %v2056 : tensor<256x256x14x14xf32>
    %v2078 = stablehlo.subtract %v2077, %v2073 : tensor<256x256x14x14xf32>
    %v2079 = stablehlo.multiply %v2068, %v2076 : tensor<256x256x14x14xf32>
    %v2080 = stablehlo.subtract %v2078, %v2079 : tensor<256x256x14x14xf32>
    %v2081 = stablehlo.divide %v2067, %v2056 : tensor<256x256x14x14xf32>
    %v2082 = stablehlo.multiply %v2081, %v2080 : tensor<256x256x14x14xf32>
    %v2083 = stablehlo.reshape %v2082 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2085 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2086 = stablehlo.transpose %v2085, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2087 = stablehlo.convolution(%v2084, %v2086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2088 = stablehlo.reshape %v2087 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2089 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2090 = stablehlo.compare GT, %v545, %v2089 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2091 = stablehlo.select %v2090, %v2088, %v2089 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2092 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2094 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2095 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2096 = stablehlo.reduce(%v2092 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2097 = stablehlo.broadcast_in_dim %v2096, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2098 = stablehlo.divide %v2097, %v2094 : tensor<256x256x14x14xf32>
    %v2099 = stablehlo.subtract %v2092, %v2098 : tensor<256x256x14x14xf32>
    %v2100 = stablehlo.multiply %v2099, %v2099 : tensor<256x256x14x14xf32>
    %v2101 = stablehlo.reduce(%v2100 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2102 = stablehlo.broadcast_in_dim %v2101, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2103 = stablehlo.divide %v2102, %v2094 : tensor<256x256x14x14xf32>
    %v2104 = stablehlo.add %v2103, %v2095 : tensor<256x256x14x14xf32>
    %v2105 = stablehlo.rsqrt %v2104 : tensor<256x256x14x14xf32>
    %v2106 = stablehlo.multiply %v2099, %v2105 : tensor<256x256x14x14xf32>
    %v2107 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2108 = stablehlo.reshape %v2091 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2109 = stablehlo.multiply %v2107, %v2108 : tensor<256x256x14x14xf32>
    %v2110 = stablehlo.reduce(%v2109 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2111 = stablehlo.broadcast_in_dim %v2110, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2112 = stablehlo.multiply %v2106, %v2109 : tensor<256x256x14x14xf32>
    %v2113 = stablehlo.reduce(%v2112 init: %v2093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2114 = stablehlo.broadcast_in_dim %v2113, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2115 = stablehlo.multiply %v2109, %v2094 : tensor<256x256x14x14xf32>
    %v2116 = stablehlo.subtract %v2115, %v2111 : tensor<256x256x14x14xf32>
    %v2117 = stablehlo.multiply %v2106, %v2114 : tensor<256x256x14x14xf32>
    %v2118 = stablehlo.subtract %v2116, %v2117 : tensor<256x256x14x14xf32>
    %v2119 = stablehlo.divide %v2105, %v2094 : tensor<256x256x14x14xf32>
    %v2120 = stablehlo.multiply %v2119, %v2118 : tensor<256x256x14x14xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2123 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2124 = stablehlo.transpose %v2123, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2125 = stablehlo.convolution(%v2122, %v2124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2126 = stablehlo.reshape %v2125 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2127 = stablehlo.add %v2126, %v2053 : tensor<256x50176xf32>
    %v2128 = stablehlo.reshape %v520 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2129 = stablehlo.reshape %v2121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2130 = stablehlo.transpose %v2128, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2131 = stablehlo.transpose %v2129, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2132 = stablehlo.convolution(%v2130, %v2131)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2133 = stablehlo.transpose %v2132, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2134 = stablehlo.reshape %v2121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2136 = stablehlo.reduce(%v2134 init: %v2135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2137 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2140 = stablehlo.reduce(%v2137 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2141 = stablehlo.broadcast_in_dim %v2140, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2142 = stablehlo.divide %v2141, %v2139 : tensor<256x256x14x14xf32>
    %v2143 = stablehlo.subtract %v2137, %v2142 : tensor<256x256x14x14xf32>
    %v2144 = stablehlo.multiply %v2143, %v2143 : tensor<256x256x14x14xf32>
    %v2145 = stablehlo.reduce(%v2144 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2147 = stablehlo.divide %v2146, %v2139 : tensor<256x256x14x14xf32>
    %v2148 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2149 = stablehlo.add %v2147, %v2148 : tensor<256x256x14x14xf32>
    %v2150 = stablehlo.rsqrt %v2149 : tensor<256x256x14x14xf32>
    %v2151 = stablehlo.multiply %v2143, %v2150 : tensor<256x256x14x14xf32>
    %v2152 = stablehlo.reshape %v2091 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2153 = stablehlo.multiply %v2152, %v2151 : tensor<256x256x14x14xf32>
    %v2154 = stablehlo.reduce(%v2153 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2155 = stablehlo.reshape %v2091 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2157 = stablehlo.reduce(%v2155 init: %v2156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2158 = stablehlo.reshape %v547 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2159 = stablehlo.reshape %v2083 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2160 = stablehlo.transpose %v2158, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2161 = stablehlo.transpose %v2159, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2162 = stablehlo.convolution(%v2160, %v2161)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2163 = stablehlo.transpose %v2162, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2164 = stablehlo.reshape %v2083 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2166 = stablehlo.reduce(%v2164 init: %v2165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2167 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2170 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2172 = stablehlo.divide %v2171, %v2169 : tensor<256x256x14x14xf32>
    %v2173 = stablehlo.subtract %v2167, %v2172 : tensor<256x256x14x14xf32>
    %v2174 = stablehlo.multiply %v2173, %v2173 : tensor<256x256x14x14xf32>
    %v2175 = stablehlo.reduce(%v2174 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2176 = stablehlo.broadcast_in_dim %v2175, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2177 = stablehlo.divide %v2176, %v2169 : tensor<256x256x14x14xf32>
    %v2178 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2179 = stablehlo.add %v2177, %v2178 : tensor<256x256x14x14xf32>
    %v2180 = stablehlo.rsqrt %v2179 : tensor<256x256x14x14xf32>
    %v2181 = stablehlo.multiply %v2173, %v2180 : tensor<256x256x14x14xf32>
    %v2182 = stablehlo.reshape %v2053 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2183 = stablehlo.multiply %v2182, %v2181 : tensor<256x256x14x14xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2185 = stablehlo.reshape %v2053 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2187 = stablehlo.reduce(%v2185 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2188 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2189 = stablehlo.compare GT, %v518, %v2188 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2190 = stablehlo.select %v2189, %v2127, %v2188 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2191 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2193 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2194 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2195 = stablehlo.reduce(%v2191 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2196 = stablehlo.broadcast_in_dim %v2195, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2197 = stablehlo.divide %v2196, %v2193 : tensor<256x256x14x14xf32>
    %v2198 = stablehlo.subtract %v2191, %v2197 : tensor<256x256x14x14xf32>
    %v2199 = stablehlo.multiply %v2198, %v2198 : tensor<256x256x14x14xf32>
    %v2200 = stablehlo.reduce(%v2199 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2201 = stablehlo.broadcast_in_dim %v2200, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2202 = stablehlo.divide %v2201, %v2193 : tensor<256x256x14x14xf32>
    %v2203 = stablehlo.add %v2202, %v2194 : tensor<256x256x14x14xf32>
    %v2204 = stablehlo.rsqrt %v2203 : tensor<256x256x14x14xf32>
    %v2205 = stablehlo.multiply %v2198, %v2204 : tensor<256x256x14x14xf32>
    %v2206 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2207 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2208 = stablehlo.multiply %v2206, %v2207 : tensor<256x256x14x14xf32>
    %v2209 = stablehlo.reduce(%v2208 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2210 = stablehlo.broadcast_in_dim %v2209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2211 = stablehlo.multiply %v2205, %v2208 : tensor<256x256x14x14xf32>
    %v2212 = stablehlo.reduce(%v2211 init: %v2192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2212, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2214 = stablehlo.multiply %v2208, %v2193 : tensor<256x256x14x14xf32>
    %v2215 = stablehlo.subtract %v2214, %v2210 : tensor<256x256x14x14xf32>
    %v2216 = stablehlo.multiply %v2205, %v2213 : tensor<256x256x14x14xf32>
    %v2217 = stablehlo.subtract %v2215, %v2216 : tensor<256x256x14x14xf32>
    %v2218 = stablehlo.divide %v2204, %v2193 : tensor<256x256x14x14xf32>
    %v2219 = stablehlo.multiply %v2218, %v2217 : tensor<256x256x14x14xf32>
    %v2220 = stablehlo.reshape %v2219 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2221 = stablehlo.reshape %v2220 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2222 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2223 = stablehlo.transpose %v2222, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2224 = stablehlo.convolution(%v2221, %v2223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2226 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2227 = stablehlo.compare GT, %v465, %v2226 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2228 = stablehlo.select %v2227, %v2225, %v2226 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2229 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2231 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2232 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2233 = stablehlo.reduce(%v2229 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2235 = stablehlo.divide %v2234, %v2231 : tensor<256x256x14x14xf32>
    %v2236 = stablehlo.subtract %v2229, %v2235 : tensor<256x256x14x14xf32>
    %v2237 = stablehlo.multiply %v2236, %v2236 : tensor<256x256x14x14xf32>
    %v2238 = stablehlo.reduce(%v2237 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2239 = stablehlo.broadcast_in_dim %v2238, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2240 = stablehlo.divide %v2239, %v2231 : tensor<256x256x14x14xf32>
    %v2241 = stablehlo.add %v2240, %v2232 : tensor<256x256x14x14xf32>
    %v2242 = stablehlo.rsqrt %v2241 : tensor<256x256x14x14xf32>
    %v2243 = stablehlo.multiply %v2236, %v2242 : tensor<256x256x14x14xf32>
    %v2244 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2245 = stablehlo.reshape %v2228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2246 = stablehlo.multiply %v2244, %v2245 : tensor<256x256x14x14xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2249 = stablehlo.multiply %v2243, %v2246 : tensor<256x256x14x14xf32>
    %v2250 = stablehlo.reduce(%v2249 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2251 = stablehlo.broadcast_in_dim %v2250, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2252 = stablehlo.multiply %v2246, %v2231 : tensor<256x256x14x14xf32>
    %v2253 = stablehlo.subtract %v2252, %v2248 : tensor<256x256x14x14xf32>
    %v2254 = stablehlo.multiply %v2243, %v2251 : tensor<256x256x14x14xf32>
    %v2255 = stablehlo.subtract %v2253, %v2254 : tensor<256x256x14x14xf32>
    %v2256 = stablehlo.divide %v2242, %v2231 : tensor<256x256x14x14xf32>
    %v2257 = stablehlo.multiply %v2256, %v2255 : tensor<256x256x14x14xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2261 = stablehlo.pad %v2259, %v2260, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2262 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2263 = stablehlo.transpose %v2262, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2264 = stablehlo.convolution(%v2261, %v2263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2266 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2268 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2269 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2270 = stablehlo.reduce(%v2266 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2271 = stablehlo.broadcast_in_dim %v2270, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2272 = stablehlo.divide %v2271, %v2268 : tensor<256x256x14x14xf32>
    %v2273 = stablehlo.subtract %v2266, %v2272 : tensor<256x256x14x14xf32>
    %v2274 = stablehlo.multiply %v2273, %v2273 : tensor<256x256x14x14xf32>
    %v2275 = stablehlo.reduce(%v2274 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2276 = stablehlo.broadcast_in_dim %v2275, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2277 = stablehlo.divide %v2276, %v2268 : tensor<256x256x14x14xf32>
    %v2278 = stablehlo.add %v2277, %v2269 : tensor<256x256x14x14xf32>
    %v2279 = stablehlo.rsqrt %v2278 : tensor<256x256x14x14xf32>
    %v2280 = stablehlo.multiply %v2273, %v2279 : tensor<256x256x14x14xf32>
    %v2281 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2282 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2283 = stablehlo.multiply %v2281, %v2282 : tensor<256x256x14x14xf32>
    %v2284 = stablehlo.reduce(%v2283 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2285 = stablehlo.broadcast_in_dim %v2284, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2286 = stablehlo.multiply %v2280, %v2283 : tensor<256x256x14x14xf32>
    %v2287 = stablehlo.reduce(%v2286 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2288 = stablehlo.broadcast_in_dim %v2287, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2289 = stablehlo.multiply %v2283, %v2268 : tensor<256x256x14x14xf32>
    %v2290 = stablehlo.subtract %v2289, %v2285 : tensor<256x256x14x14xf32>
    %v2291 = stablehlo.multiply %v2280, %v2288 : tensor<256x256x14x14xf32>
    %v2292 = stablehlo.subtract %v2290, %v2291 : tensor<256x256x14x14xf32>
    %v2293 = stablehlo.divide %v2279, %v2268 : tensor<256x256x14x14xf32>
    %v2294 = stablehlo.multiply %v2293, %v2292 : tensor<256x256x14x14xf32>
    %v2295 = stablehlo.reshape %v2294 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2296 = stablehlo.reshape %v2295 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.pad %v2296, %v2297, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2299 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2300 = stablehlo.transpose %v2299, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2301 = stablehlo.convolution(%v2298, %v2300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2302 = stablehlo.reshape %v2301 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2303 = stablehlo.add %v2265, %v2302 : tensor<256x100352xf32>
    %v2304 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2305 = stablehlo.reshape %v2258 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2307 = stablehlo.pad %v2305, %v2306, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2308 = stablehlo.transpose %v2304, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2309 = stablehlo.transpose %v2307, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v2310 = stablehlo.convolution(%v2308, %v2309)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2311 = stablehlo.transpose %v2310, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2312 = stablehlo.reshape %v2258 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2314 = stablehlo.reduce(%v2312 init: %v2313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2315 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2317 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2318 = stablehlo.reduce(%v2315 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2319 = stablehlo.broadcast_in_dim %v2318, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2320 = stablehlo.divide %v2319, %v2317 : tensor<256x256x14x14xf32>
    %v2321 = stablehlo.subtract %v2315, %v2320 : tensor<256x256x14x14xf32>
    %v2322 = stablehlo.multiply %v2321, %v2321 : tensor<256x256x14x14xf32>
    %v2323 = stablehlo.reduce(%v2322 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2324 = stablehlo.broadcast_in_dim %v2323, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2325 = stablehlo.divide %v2324, %v2317 : tensor<256x256x14x14xf32>
    %v2326 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2327 = stablehlo.add %v2325, %v2326 : tensor<256x256x14x14xf32>
    %v2328 = stablehlo.rsqrt %v2327 : tensor<256x256x14x14xf32>
    %v2329 = stablehlo.multiply %v2321, %v2328 : tensor<256x256x14x14xf32>
    %v2330 = stablehlo.reshape %v2228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2331 = stablehlo.multiply %v2330, %v2329 : tensor<256x256x14x14xf32>
    %v2332 = stablehlo.reduce(%v2331 init: %v2316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2333 = stablehlo.reshape %v2228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2335 = stablehlo.reduce(%v2333 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2336 = stablehlo.reshape %v467 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2337 = stablehlo.reshape %v2220 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2338 = stablehlo.transpose %v2336, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2339 = stablehlo.transpose %v2337, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2340 = stablehlo.convolution(%v2338, %v2339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2341 = stablehlo.transpose %v2340, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2342 = stablehlo.reshape %v2220 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2344 = stablehlo.reduce(%v2342 init: %v2343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2345 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2347 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2348 = stablehlo.reduce(%v2345 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2349 = stablehlo.broadcast_in_dim %v2348, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2350 = stablehlo.divide %v2349, %v2347 : tensor<256x256x14x14xf32>
    %v2351 = stablehlo.subtract %v2345, %v2350 : tensor<256x256x14x14xf32>
    %v2352 = stablehlo.multiply %v2351, %v2351 : tensor<256x256x14x14xf32>
    %v2353 = stablehlo.reduce(%v2352 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2354 = stablehlo.broadcast_in_dim %v2353, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2355 = stablehlo.divide %v2354, %v2347 : tensor<256x256x14x14xf32>
    %v2356 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2357 = stablehlo.add %v2355, %v2356 : tensor<256x256x14x14xf32>
    %v2358 = stablehlo.rsqrt %v2357 : tensor<256x256x14x14xf32>
    %v2359 = stablehlo.multiply %v2351, %v2358 : tensor<256x256x14x14xf32>
    %v2360 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2361 = stablehlo.multiply %v2360, %v2359 : tensor<256x256x14x14xf32>
    %v2362 = stablehlo.reduce(%v2361 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2363 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2366 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2367 = stablehlo.reshape %v2295 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2369 = stablehlo.pad %v2367, %v2368, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v2370 = stablehlo.transpose %v2366, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2371 = stablehlo.transpose %v2369, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v2372 = stablehlo.convolution(%v2370, %v2371)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<128x256x3x3xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2374 = stablehlo.reshape %v2295 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2376 = stablehlo.reduce(%v2374 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2377 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2379 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2380 = stablehlo.reduce(%v2377 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2381 = stablehlo.broadcast_in_dim %v2380, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2382 = stablehlo.divide %v2381, %v2379 : tensor<256x256x14x14xf32>
    %v2383 = stablehlo.subtract %v2377, %v2382 : tensor<256x256x14x14xf32>
    %v2384 = stablehlo.multiply %v2383, %v2383 : tensor<256x256x14x14xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2387 = stablehlo.divide %v2386, %v2379 : tensor<256x256x14x14xf32>
    %v2388 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2389 = stablehlo.add %v2387, %v2388 : tensor<256x256x14x14xf32>
    %v2390 = stablehlo.rsqrt %v2389 : tensor<256x256x14x14xf32>
    %v2391 = stablehlo.multiply %v2383, %v2390 : tensor<256x256x14x14xf32>
    %v2392 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2393 = stablehlo.multiply %v2392, %v2391 : tensor<256x256x14x14xf32>
    %v2394 = stablehlo.reduce(%v2393 init: %v2378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2395 = stablehlo.reshape %v2190 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2397 = stablehlo.reduce(%v2395 init: %v2396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2399 = stablehlo.compare GT, %v438, %v2398 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2400 = stablehlo.select %v2399, %v2303, %v2398 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2401 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2403 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2404 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2405 = stablehlo.reduce(%v2401 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2406 = stablehlo.broadcast_in_dim %v2405, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2407 = stablehlo.divide %v2406, %v2403 : tensor<256x128x28x28xf32>
    %v2408 = stablehlo.subtract %v2401, %v2407 : tensor<256x128x28x28xf32>
    %v2409 = stablehlo.multiply %v2408, %v2408 : tensor<256x128x28x28xf32>
    %v2410 = stablehlo.reduce(%v2409 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2411 = stablehlo.broadcast_in_dim %v2410, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2412 = stablehlo.divide %v2411, %v2403 : tensor<256x128x28x28xf32>
    %v2413 = stablehlo.add %v2412, %v2404 : tensor<256x128x28x28xf32>
    %v2414 = stablehlo.rsqrt %v2413 : tensor<256x128x28x28xf32>
    %v2415 = stablehlo.multiply %v2408, %v2414 : tensor<256x128x28x28xf32>
    %v2416 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2417 = stablehlo.reshape %v2400 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2418 = stablehlo.multiply %v2416, %v2417 : tensor<256x128x28x28xf32>
    %v2419 = stablehlo.reduce(%v2418 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2420 = stablehlo.broadcast_in_dim %v2419, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2421 = stablehlo.multiply %v2415, %v2418 : tensor<256x128x28x28xf32>
    %v2422 = stablehlo.reduce(%v2421 init: %v2402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2423 = stablehlo.broadcast_in_dim %v2422, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2424 = stablehlo.multiply %v2418, %v2403 : tensor<256x128x28x28xf32>
    %v2425 = stablehlo.subtract %v2424, %v2420 : tensor<256x128x28x28xf32>
    %v2426 = stablehlo.multiply %v2415, %v2423 : tensor<256x128x28x28xf32>
    %v2427 = stablehlo.subtract %v2425, %v2426 : tensor<256x128x28x28xf32>
    %v2428 = stablehlo.divide %v2414, %v2403 : tensor<256x128x28x28xf32>
    %v2429 = stablehlo.multiply %v2428, %v2427 : tensor<256x128x28x28xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2431 = stablehlo.reshape %v2430 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2432 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2433 = stablehlo.transpose %v2432, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2434 = stablehlo.convolution(%v2431, %v2433)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2435 = stablehlo.reshape %v2434 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2436 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2437 = stablehlo.compare GT, %v410, %v2436 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2438 = stablehlo.select %v2437, %v2435, %v2436 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2439 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2440 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2441 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2442 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2443 = stablehlo.reduce(%v2439 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2444 = stablehlo.broadcast_in_dim %v2443, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2445 = stablehlo.divide %v2444, %v2441 : tensor<256x128x28x28xf32>
    %v2446 = stablehlo.subtract %v2439, %v2445 : tensor<256x128x28x28xf32>
    %v2447 = stablehlo.multiply %v2446, %v2446 : tensor<256x128x28x28xf32>
    %v2448 = stablehlo.reduce(%v2447 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2449 = stablehlo.broadcast_in_dim %v2448, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2450 = stablehlo.divide %v2449, %v2441 : tensor<256x128x28x28xf32>
    %v2451 = stablehlo.add %v2450, %v2442 : tensor<256x128x28x28xf32>
    %v2452 = stablehlo.rsqrt %v2451 : tensor<256x128x28x28xf32>
    %v2453 = stablehlo.multiply %v2446, %v2452 : tensor<256x128x28x28xf32>
    %v2454 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2455 = stablehlo.reshape %v2438 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2456 = stablehlo.multiply %v2454, %v2455 : tensor<256x128x28x28xf32>
    %v2457 = stablehlo.reduce(%v2456 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2458 = stablehlo.broadcast_in_dim %v2457, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2459 = stablehlo.multiply %v2453, %v2456 : tensor<256x128x28x28xf32>
    %v2460 = stablehlo.reduce(%v2459 init: %v2440) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2461 = stablehlo.broadcast_in_dim %v2460, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2462 = stablehlo.multiply %v2456, %v2441 : tensor<256x128x28x28xf32>
    %v2463 = stablehlo.subtract %v2462, %v2458 : tensor<256x128x28x28xf32>
    %v2464 = stablehlo.multiply %v2453, %v2461 : tensor<256x128x28x28xf32>
    %v2465 = stablehlo.subtract %v2463, %v2464 : tensor<256x128x28x28xf32>
    %v2466 = stablehlo.divide %v2452, %v2441 : tensor<256x128x28x28xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<256x128x28x28xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2469 = stablehlo.reshape %v2468 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2470 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2471 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2472 = stablehlo.convolution(%v2469, %v2471)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2473 = stablehlo.reshape %v2472 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2474 = stablehlo.add %v2473, %v2400 : tensor<256x100352xf32>
    %v2475 = stablehlo.reshape %v385 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2476 = stablehlo.reshape %v2468 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2477 = stablehlo.transpose %v2475, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2478 = stablehlo.transpose %v2476, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2479 = stablehlo.convolution(%v2477, %v2478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2480 = stablehlo.transpose %v2479, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2481 = stablehlo.reshape %v2468 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2483 = stablehlo.reduce(%v2481 init: %v2482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2484 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2486 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2487 = stablehlo.reduce(%v2484 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2488 = stablehlo.broadcast_in_dim %v2487, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2489 = stablehlo.divide %v2488, %v2486 : tensor<256x128x28x28xf32>
    %v2490 = stablehlo.subtract %v2484, %v2489 : tensor<256x128x28x28xf32>
    %v2491 = stablehlo.multiply %v2490, %v2490 : tensor<256x128x28x28xf32>
    %v2492 = stablehlo.reduce(%v2491 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2493 = stablehlo.broadcast_in_dim %v2492, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2494 = stablehlo.divide %v2493, %v2486 : tensor<256x128x28x28xf32>
    %v2495 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2496 = stablehlo.add %v2494, %v2495 : tensor<256x128x28x28xf32>
    %v2497 = stablehlo.rsqrt %v2496 : tensor<256x128x28x28xf32>
    %v2498 = stablehlo.multiply %v2490, %v2497 : tensor<256x128x28x28xf32>
    %v2499 = stablehlo.reshape %v2438 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2500 = stablehlo.multiply %v2499, %v2498 : tensor<256x128x28x28xf32>
    %v2501 = stablehlo.reduce(%v2500 init: %v2485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2502 = stablehlo.reshape %v2438 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2504 = stablehlo.reduce(%v2502 init: %v2503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2505 = stablehlo.reshape %v412 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2506 = stablehlo.reshape %v2430 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2507 = stablehlo.transpose %v2505, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2508 = stablehlo.transpose %v2506, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2509 = stablehlo.convolution(%v2507, %v2508)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2510 = stablehlo.transpose %v2509, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2511 = stablehlo.reshape %v2430 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2513 = stablehlo.reduce(%v2511 init: %v2512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2514 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2516 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2517 = stablehlo.reduce(%v2514 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2518 = stablehlo.broadcast_in_dim %v2517, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2519 = stablehlo.divide %v2518, %v2516 : tensor<256x128x28x28xf32>
    %v2520 = stablehlo.subtract %v2514, %v2519 : tensor<256x128x28x28xf32>
    %v2521 = stablehlo.multiply %v2520, %v2520 : tensor<256x128x28x28xf32>
    %v2522 = stablehlo.reduce(%v2521 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2523 = stablehlo.broadcast_in_dim %v2522, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2524 = stablehlo.divide %v2523, %v2516 : tensor<256x128x28x28xf32>
    %v2525 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2526 = stablehlo.add %v2524, %v2525 : tensor<256x128x28x28xf32>
    %v2527 = stablehlo.rsqrt %v2526 : tensor<256x128x28x28xf32>
    %v2528 = stablehlo.multiply %v2520, %v2527 : tensor<256x128x28x28xf32>
    %v2529 = stablehlo.reshape %v2400 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2530 = stablehlo.multiply %v2529, %v2528 : tensor<256x128x28x28xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2532 = stablehlo.reshape %v2400 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2534 = stablehlo.reduce(%v2532 init: %v2533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2536 = stablehlo.compare GT, %v383, %v2535 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2537 = stablehlo.select %v2536, %v2474, %v2535 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2538 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2540 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2541 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2542 = stablehlo.reduce(%v2538 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2543 = stablehlo.broadcast_in_dim %v2542, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2544 = stablehlo.divide %v2543, %v2540 : tensor<256x128x28x28xf32>
    %v2545 = stablehlo.subtract %v2538, %v2544 : tensor<256x128x28x28xf32>
    %v2546 = stablehlo.multiply %v2545, %v2545 : tensor<256x128x28x28xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2549 = stablehlo.divide %v2548, %v2540 : tensor<256x128x28x28xf32>
    %v2550 = stablehlo.add %v2549, %v2541 : tensor<256x128x28x28xf32>
    %v2551 = stablehlo.rsqrt %v2550 : tensor<256x128x28x28xf32>
    %v2552 = stablehlo.multiply %v2545, %v2551 : tensor<256x128x28x28xf32>
    %v2553 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2554 = stablehlo.reshape %v2537 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2555 = stablehlo.multiply %v2553, %v2554 : tensor<256x128x28x28xf32>
    %v2556 = stablehlo.reduce(%v2555 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2557 = stablehlo.broadcast_in_dim %v2556, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2558 = stablehlo.multiply %v2552, %v2555 : tensor<256x128x28x28xf32>
    %v2559 = stablehlo.reduce(%v2558 init: %v2539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2560 = stablehlo.broadcast_in_dim %v2559, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2561 = stablehlo.multiply %v2555, %v2540 : tensor<256x128x28x28xf32>
    %v2562 = stablehlo.subtract %v2561, %v2557 : tensor<256x128x28x28xf32>
    %v2563 = stablehlo.multiply %v2552, %v2560 : tensor<256x128x28x28xf32>
    %v2564 = stablehlo.subtract %v2562, %v2563 : tensor<256x128x28x28xf32>
    %v2565 = stablehlo.divide %v2551, %v2540 : tensor<256x128x28x28xf32>
    %v2566 = stablehlo.multiply %v2565, %v2564 : tensor<256x128x28x28xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2568 = stablehlo.reshape %v2567 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2569 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2570 = stablehlo.transpose %v2569, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2571 = stablehlo.convolution(%v2568, %v2570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2572 = stablehlo.reshape %v2571 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2573 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2574 = stablehlo.compare GT, %v355, %v2573 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2575 = stablehlo.select %v2574, %v2572, %v2573 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2576 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2578 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2579 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2580 = stablehlo.reduce(%v2576 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2581 = stablehlo.broadcast_in_dim %v2580, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2582 = stablehlo.divide %v2581, %v2578 : tensor<256x128x28x28xf32>
    %v2583 = stablehlo.subtract %v2576, %v2582 : tensor<256x128x28x28xf32>
    %v2584 = stablehlo.multiply %v2583, %v2583 : tensor<256x128x28x28xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2585, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2587 = stablehlo.divide %v2586, %v2578 : tensor<256x128x28x28xf32>
    %v2588 = stablehlo.add %v2587, %v2579 : tensor<256x128x28x28xf32>
    %v2589 = stablehlo.rsqrt %v2588 : tensor<256x128x28x28xf32>
    %v2590 = stablehlo.multiply %v2583, %v2589 : tensor<256x128x28x28xf32>
    %v2591 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2592 = stablehlo.reshape %v2575 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2593 = stablehlo.multiply %v2591, %v2592 : tensor<256x128x28x28xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2596 = stablehlo.multiply %v2590, %v2593 : tensor<256x128x28x28xf32>
    %v2597 = stablehlo.reduce(%v2596 init: %v2577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2598 = stablehlo.broadcast_in_dim %v2597, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2599 = stablehlo.multiply %v2593, %v2578 : tensor<256x128x28x28xf32>
    %v2600 = stablehlo.subtract %v2599, %v2595 : tensor<256x128x28x28xf32>
    %v2601 = stablehlo.multiply %v2590, %v2598 : tensor<256x128x28x28xf32>
    %v2602 = stablehlo.subtract %v2600, %v2601 : tensor<256x128x28x28xf32>
    %v2603 = stablehlo.divide %v2589, %v2578 : tensor<256x128x28x28xf32>
    %v2604 = stablehlo.multiply %v2603, %v2602 : tensor<256x128x28x28xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2607 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2608 = stablehlo.transpose %v2607, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2609 = stablehlo.convolution(%v2606, %v2608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2611 = stablehlo.add %v2610, %v2537 : tensor<256x100352xf32>
    %v2612 = stablehlo.reshape %v330 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2613 = stablehlo.reshape %v2605 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2614 = stablehlo.transpose %v2612, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2615 = stablehlo.transpose %v2613, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2616 = stablehlo.convolution(%v2614, %v2615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2617 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2618 = stablehlo.reshape %v2605 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2619 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2620 = stablehlo.reduce(%v2618 init: %v2619) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2621 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2623 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2624 = stablehlo.reduce(%v2621 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2626 = stablehlo.divide %v2625, %v2623 : tensor<256x128x28x28xf32>
    %v2627 = stablehlo.subtract %v2621, %v2626 : tensor<256x128x28x28xf32>
    %v2628 = stablehlo.multiply %v2627, %v2627 : tensor<256x128x28x28xf32>
    %v2629 = stablehlo.reduce(%v2628 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2630 = stablehlo.broadcast_in_dim %v2629, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2631 = stablehlo.divide %v2630, %v2623 : tensor<256x128x28x28xf32>
    %v2632 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2633 = stablehlo.add %v2631, %v2632 : tensor<256x128x28x28xf32>
    %v2634 = stablehlo.rsqrt %v2633 : tensor<256x128x28x28xf32>
    %v2635 = stablehlo.multiply %v2627, %v2634 : tensor<256x128x28x28xf32>
    %v2636 = stablehlo.reshape %v2575 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2637 = stablehlo.multiply %v2636, %v2635 : tensor<256x128x28x28xf32>
    %v2638 = stablehlo.reduce(%v2637 init: %v2622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2639 = stablehlo.reshape %v2575 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.reduce(%v2639 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2642 = stablehlo.reshape %v357 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2643 = stablehlo.reshape %v2567 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2644 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2645 = stablehlo.transpose %v2643, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2646 = stablehlo.convolution(%v2644, %v2645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2647 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2648 = stablehlo.reshape %v2567 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2650 = stablehlo.reduce(%v2648 init: %v2649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2651 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2653 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2654 = stablehlo.reduce(%v2651 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2654, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2656 = stablehlo.divide %v2655, %v2653 : tensor<256x128x28x28xf32>
    %v2657 = stablehlo.subtract %v2651, %v2656 : tensor<256x128x28x28xf32>
    %v2658 = stablehlo.multiply %v2657, %v2657 : tensor<256x128x28x28xf32>
    %v2659 = stablehlo.reduce(%v2658 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2661 = stablehlo.divide %v2660, %v2653 : tensor<256x128x28x28xf32>
    %v2662 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2663 = stablehlo.add %v2661, %v2662 : tensor<256x128x28x28xf32>
    %v2664 = stablehlo.rsqrt %v2663 : tensor<256x128x28x28xf32>
    %v2665 = stablehlo.multiply %v2657, %v2664 : tensor<256x128x28x28xf32>
    %v2666 = stablehlo.reshape %v2537 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2667 = stablehlo.multiply %v2666, %v2665 : tensor<256x128x28x28xf32>
    %v2668 = stablehlo.reduce(%v2667 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2669 = stablehlo.reshape %v2537 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2671 = stablehlo.reduce(%v2669 init: %v2670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2672 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2673 = stablehlo.compare GT, %v328, %v2672 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2674 = stablehlo.select %v2673, %v2611, %v2672 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2675 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2678 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2679 = stablehlo.reduce(%v2675 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2680 = stablehlo.broadcast_in_dim %v2679, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2681 = stablehlo.divide %v2680, %v2677 : tensor<256x128x28x28xf32>
    %v2682 = stablehlo.subtract %v2675, %v2681 : tensor<256x128x28x28xf32>
    %v2683 = stablehlo.multiply %v2682, %v2682 : tensor<256x128x28x28xf32>
    %v2684 = stablehlo.reduce(%v2683 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2685 = stablehlo.broadcast_in_dim %v2684, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2686 = stablehlo.divide %v2685, %v2677 : tensor<256x128x28x28xf32>
    %v2687 = stablehlo.add %v2686, %v2678 : tensor<256x128x28x28xf32>
    %v2688 = stablehlo.rsqrt %v2687 : tensor<256x128x28x28xf32>
    %v2689 = stablehlo.multiply %v2682, %v2688 : tensor<256x128x28x28xf32>
    %v2690 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2691 = stablehlo.reshape %v2674 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2692 = stablehlo.multiply %v2690, %v2691 : tensor<256x128x28x28xf32>
    %v2693 = stablehlo.reduce(%v2692 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2694 = stablehlo.broadcast_in_dim %v2693, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2695 = stablehlo.multiply %v2689, %v2692 : tensor<256x128x28x28xf32>
    %v2696 = stablehlo.reduce(%v2695 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2697 = stablehlo.broadcast_in_dim %v2696, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2698 = stablehlo.multiply %v2692, %v2677 : tensor<256x128x28x28xf32>
    %v2699 = stablehlo.subtract %v2698, %v2694 : tensor<256x128x28x28xf32>
    %v2700 = stablehlo.multiply %v2689, %v2697 : tensor<256x128x28x28xf32>
    %v2701 = stablehlo.subtract %v2699, %v2700 : tensor<256x128x28x28xf32>
    %v2702 = stablehlo.divide %v2688, %v2677 : tensor<256x128x28x28xf32>
    %v2703 = stablehlo.multiply %v2702, %v2701 : tensor<256x128x28x28xf32>
    %v2704 = stablehlo.reshape %v2703 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2706 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2707 = stablehlo.transpose %v2706, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2708 = stablehlo.convolution(%v2705, %v2707)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2709 = stablehlo.reshape %v2708 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2710 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2711 = stablehlo.compare GT, %v300, %v2710 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2712 = stablehlo.select %v2711, %v2709, %v2710 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2713 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2715 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2716 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2717 = stablehlo.reduce(%v2713 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2717, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2719 = stablehlo.divide %v2718, %v2715 : tensor<256x128x28x28xf32>
    %v2720 = stablehlo.subtract %v2713, %v2719 : tensor<256x128x28x28xf32>
    %v2721 = stablehlo.multiply %v2720, %v2720 : tensor<256x128x28x28xf32>
    %v2722 = stablehlo.reduce(%v2721 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2723 = stablehlo.broadcast_in_dim %v2722, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2724 = stablehlo.divide %v2723, %v2715 : tensor<256x128x28x28xf32>
    %v2725 = stablehlo.add %v2724, %v2716 : tensor<256x128x28x28xf32>
    %v2726 = stablehlo.rsqrt %v2725 : tensor<256x128x28x28xf32>
    %v2727 = stablehlo.multiply %v2720, %v2726 : tensor<256x128x28x28xf32>
    %v2728 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2729 = stablehlo.reshape %v2712 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2730 = stablehlo.multiply %v2728, %v2729 : tensor<256x128x28x28xf32>
    %v2731 = stablehlo.reduce(%v2730 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2732 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2733 = stablehlo.multiply %v2727, %v2730 : tensor<256x128x28x28xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2736 = stablehlo.multiply %v2730, %v2715 : tensor<256x128x28x28xf32>
    %v2737 = stablehlo.subtract %v2736, %v2732 : tensor<256x128x28x28xf32>
    %v2738 = stablehlo.multiply %v2727, %v2735 : tensor<256x128x28x28xf32>
    %v2739 = stablehlo.subtract %v2737, %v2738 : tensor<256x128x28x28xf32>
    %v2740 = stablehlo.divide %v2726, %v2715 : tensor<256x128x28x28xf32>
    %v2741 = stablehlo.multiply %v2740, %v2739 : tensor<256x128x28x28xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2744 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2745 = stablehlo.transpose %v2744, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2746 = stablehlo.convolution(%v2743, %v2745)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2748 = stablehlo.add %v2747, %v2674 : tensor<256x100352xf32>
    %v2749 = stablehlo.reshape %v275 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2750 = stablehlo.reshape %v2742 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2751 = stablehlo.transpose %v2749, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2752 = stablehlo.transpose %v2750, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2753 = stablehlo.convolution(%v2751, %v2752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2754 = stablehlo.transpose %v2753, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2755 = stablehlo.reshape %v2742 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2757 = stablehlo.reduce(%v2755 init: %v2756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2758 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2761 = stablehlo.reduce(%v2758 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2762 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2763 = stablehlo.divide %v2762, %v2760 : tensor<256x128x28x28xf32>
    %v2764 = stablehlo.subtract %v2758, %v2763 : tensor<256x128x28x28xf32>
    %v2765 = stablehlo.multiply %v2764, %v2764 : tensor<256x128x28x28xf32>
    %v2766 = stablehlo.reduce(%v2765 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2767 = stablehlo.broadcast_in_dim %v2766, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2768 = stablehlo.divide %v2767, %v2760 : tensor<256x128x28x28xf32>
    %v2769 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2770 = stablehlo.add %v2768, %v2769 : tensor<256x128x28x28xf32>
    %v2771 = stablehlo.rsqrt %v2770 : tensor<256x128x28x28xf32>
    %v2772 = stablehlo.multiply %v2764, %v2771 : tensor<256x128x28x28xf32>
    %v2773 = stablehlo.reshape %v2712 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2774 = stablehlo.multiply %v2773, %v2772 : tensor<256x128x28x28xf32>
    %v2775 = stablehlo.reduce(%v2774 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2776 = stablehlo.reshape %v2712 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2777 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2778 = stablehlo.reduce(%v2776 init: %v2777) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2779 = stablehlo.reshape %v302 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2780 = stablehlo.reshape %v2704 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2781 = stablehlo.transpose %v2779, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2782 = stablehlo.transpose %v2780, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2783 = stablehlo.convolution(%v2781, %v2782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2784 = stablehlo.transpose %v2783, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2785 = stablehlo.reshape %v2704 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2787 = stablehlo.reduce(%v2785 init: %v2786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2788 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2790 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2791 = stablehlo.reduce(%v2788 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2791, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2793 = stablehlo.divide %v2792, %v2790 : tensor<256x128x28x28xf32>
    %v2794 = stablehlo.subtract %v2788, %v2793 : tensor<256x128x28x28xf32>
    %v2795 = stablehlo.multiply %v2794, %v2794 : tensor<256x128x28x28xf32>
    %v2796 = stablehlo.reduce(%v2795 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2797 = stablehlo.broadcast_in_dim %v2796, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2798 = stablehlo.divide %v2797, %v2790 : tensor<256x128x28x28xf32>
    %v2799 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2800 = stablehlo.add %v2798, %v2799 : tensor<256x128x28x28xf32>
    %v2801 = stablehlo.rsqrt %v2800 : tensor<256x128x28x28xf32>
    %v2802 = stablehlo.multiply %v2794, %v2801 : tensor<256x128x28x28xf32>
    %v2803 = stablehlo.reshape %v2674 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2804 = stablehlo.multiply %v2803, %v2802 : tensor<256x128x28x28xf32>
    %v2805 = stablehlo.reduce(%v2804 init: %v2789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2806 = stablehlo.reshape %v2674 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2808 = stablehlo.reduce(%v2806 init: %v2807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2809 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2810 = stablehlo.compare GT, %v273, %v2809 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2811 = stablehlo.select %v2810, %v2748, %v2809 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2812 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2814 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2815 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2816 = stablehlo.reduce(%v2812 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2817 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2818 = stablehlo.divide %v2817, %v2814 : tensor<256x128x28x28xf32>
    %v2819 = stablehlo.subtract %v2812, %v2818 : tensor<256x128x28x28xf32>
    %v2820 = stablehlo.multiply %v2819, %v2819 : tensor<256x128x28x28xf32>
    %v2821 = stablehlo.reduce(%v2820 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2822 = stablehlo.broadcast_in_dim %v2821, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2823 = stablehlo.divide %v2822, %v2814 : tensor<256x128x28x28xf32>
    %v2824 = stablehlo.add %v2823, %v2815 : tensor<256x128x28x28xf32>
    %v2825 = stablehlo.rsqrt %v2824 : tensor<256x128x28x28xf32>
    %v2826 = stablehlo.multiply %v2819, %v2825 : tensor<256x128x28x28xf32>
    %v2827 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2828 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2829 = stablehlo.multiply %v2827, %v2828 : tensor<256x128x28x28xf32>
    %v2830 = stablehlo.reduce(%v2829 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2831 = stablehlo.broadcast_in_dim %v2830, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2832 = stablehlo.multiply %v2826, %v2829 : tensor<256x128x28x28xf32>
    %v2833 = stablehlo.reduce(%v2832 init: %v2813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2834 = stablehlo.broadcast_in_dim %v2833, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2835 = stablehlo.multiply %v2829, %v2814 : tensor<256x128x28x28xf32>
    %v2836 = stablehlo.subtract %v2835, %v2831 : tensor<256x128x28x28xf32>
    %v2837 = stablehlo.multiply %v2826, %v2834 : tensor<256x128x28x28xf32>
    %v2838 = stablehlo.subtract %v2836, %v2837 : tensor<256x128x28x28xf32>
    %v2839 = stablehlo.divide %v2825, %v2814 : tensor<256x128x28x28xf32>
    %v2840 = stablehlo.multiply %v2839, %v2838 : tensor<256x128x28x28xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2843 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2844 = stablehlo.transpose %v2843, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2845 = stablehlo.convolution(%v2842, %v2844)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v2846 = stablehlo.reshape %v2845 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2847 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v2848 = stablehlo.compare GT, %v220, %v2847 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v2849 = stablehlo.select %v2848, %v2846, %v2847 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v2850 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2852 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2853 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2854 = stablehlo.reduce(%v2850 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2855 = stablehlo.broadcast_in_dim %v2854, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2856 = stablehlo.divide %v2855, %v2852 : tensor<256x128x28x28xf32>
    %v2857 = stablehlo.subtract %v2850, %v2856 : tensor<256x128x28x28xf32>
    %v2858 = stablehlo.multiply %v2857, %v2857 : tensor<256x128x28x28xf32>
    %v2859 = stablehlo.reduce(%v2858 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2860 = stablehlo.broadcast_in_dim %v2859, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2861 = stablehlo.divide %v2860, %v2852 : tensor<256x128x28x28xf32>
    %v2862 = stablehlo.add %v2861, %v2853 : tensor<256x128x28x28xf32>
    %v2863 = stablehlo.rsqrt %v2862 : tensor<256x128x28x28xf32>
    %v2864 = stablehlo.multiply %v2857, %v2863 : tensor<256x128x28x28xf32>
    %v2865 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2866 = stablehlo.reshape %v2849 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2867 = stablehlo.multiply %v2865, %v2866 : tensor<256x128x28x28xf32>
    %v2868 = stablehlo.reduce(%v2867 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2869 = stablehlo.broadcast_in_dim %v2868, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2870 = stablehlo.multiply %v2864, %v2867 : tensor<256x128x28x28xf32>
    %v2871 = stablehlo.reduce(%v2870 init: %v2851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2872 = stablehlo.broadcast_in_dim %v2871, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2873 = stablehlo.multiply %v2867, %v2852 : tensor<256x128x28x28xf32>
    %v2874 = stablehlo.subtract %v2873, %v2869 : tensor<256x128x28x28xf32>
    %v2875 = stablehlo.multiply %v2864, %v2872 : tensor<256x128x28x28xf32>
    %v2876 = stablehlo.subtract %v2874, %v2875 : tensor<256x128x28x28xf32>
    %v2877 = stablehlo.divide %v2863, %v2852 : tensor<256x128x28x28xf32>
    %v2878 = stablehlo.multiply %v2877, %v2876 : tensor<256x128x28x28xf32>
    %v2879 = stablehlo.reshape %v2878 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2880 = stablehlo.reshape %v2879 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2882 = stablehlo.pad %v2880, %v2881, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2883 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2884 = stablehlo.transpose %v2883, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2885 = stablehlo.convolution(%v2882, %v2884)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v2886 = stablehlo.reshape %v2885 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2887 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2889 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2890 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2891 = stablehlo.reduce(%v2887 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2892 = stablehlo.broadcast_in_dim %v2891, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2893 = stablehlo.divide %v2892, %v2889 : tensor<256x128x28x28xf32>
    %v2894 = stablehlo.subtract %v2887, %v2893 : tensor<256x128x28x28xf32>
    %v2895 = stablehlo.multiply %v2894, %v2894 : tensor<256x128x28x28xf32>
    %v2896 = stablehlo.reduce(%v2895 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2898 = stablehlo.divide %v2897, %v2889 : tensor<256x128x28x28xf32>
    %v2899 = stablehlo.add %v2898, %v2890 : tensor<256x128x28x28xf32>
    %v2900 = stablehlo.rsqrt %v2899 : tensor<256x128x28x28xf32>
    %v2901 = stablehlo.multiply %v2894, %v2900 : tensor<256x128x28x28xf32>
    %v2902 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2903 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2904 = stablehlo.multiply %v2902, %v2903 : tensor<256x128x28x28xf32>
    %v2905 = stablehlo.reduce(%v2904 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2905, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2907 = stablehlo.multiply %v2901, %v2904 : tensor<256x128x28x28xf32>
    %v2908 = stablehlo.reduce(%v2907 init: %v2888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2910 = stablehlo.multiply %v2904, %v2889 : tensor<256x128x28x28xf32>
    %v2911 = stablehlo.subtract %v2910, %v2906 : tensor<256x128x28x28xf32>
    %v2912 = stablehlo.multiply %v2901, %v2909 : tensor<256x128x28x28xf32>
    %v2913 = stablehlo.subtract %v2911, %v2912 : tensor<256x128x28x28xf32>
    %v2914 = stablehlo.divide %v2900, %v2889 : tensor<256x128x28x28xf32>
    %v2915 = stablehlo.multiply %v2914, %v2913 : tensor<256x128x28x28xf32>
    %v2916 = stablehlo.reshape %v2915 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v2917 = stablehlo.reshape %v2916 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.pad %v2917, %v2918, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2920 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v2921 = stablehlo.transpose %v2920, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v2922 = stablehlo.convolution(%v2919, %v2921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v2923 = stablehlo.reshape %v2922 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v2924 = stablehlo.add %v2886, %v2923 : tensor<256x200704xf32>
    %v2925 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2926 = stablehlo.reshape %v2879 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2928 = stablehlo.pad %v2926, %v2927, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2929 = stablehlo.transpose %v2925, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v2930 = stablehlo.transpose %v2928, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v2931 = stablehlo.convolution(%v2929, %v2930)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2932 = stablehlo.transpose %v2931, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2933 = stablehlo.reshape %v2879 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2935 = stablehlo.reduce(%v2933 init: %v2934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2936 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2939 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2941 = stablehlo.divide %v2940, %v2938 : tensor<256x128x28x28xf32>
    %v2942 = stablehlo.subtract %v2936, %v2941 : tensor<256x128x28x28xf32>
    %v2943 = stablehlo.multiply %v2942, %v2942 : tensor<256x128x28x28xf32>
    %v2944 = stablehlo.reduce(%v2943 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2946 = stablehlo.divide %v2945, %v2938 : tensor<256x128x28x28xf32>
    %v2947 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2948 = stablehlo.add %v2946, %v2947 : tensor<256x128x28x28xf32>
    %v2949 = stablehlo.rsqrt %v2948 : tensor<256x128x28x28xf32>
    %v2950 = stablehlo.multiply %v2942, %v2949 : tensor<256x128x28x28xf32>
    %v2951 = stablehlo.reshape %v2849 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2952 = stablehlo.multiply %v2951, %v2950 : tensor<256x128x28x28xf32>
    %v2953 = stablehlo.reduce(%v2952 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2954 = stablehlo.reshape %v2849 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2956 = stablehlo.reduce(%v2954 init: %v2955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2957 = stablehlo.reshape %v222 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2958 = stablehlo.reshape %v2841 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2959 = stablehlo.transpose %v2957, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2960 = stablehlo.transpose %v2958, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v2961 = stablehlo.convolution(%v2959, %v2960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v2962 = stablehlo.transpose %v2961, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2963 = stablehlo.reshape %v2841 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2965 = stablehlo.reduce(%v2963 init: %v2964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2966 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2968 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v2969 = stablehlo.reduce(%v2966 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2971 = stablehlo.divide %v2970, %v2968 : tensor<256x128x28x28xf32>
    %v2972 = stablehlo.subtract %v2966, %v2971 : tensor<256x128x28x28xf32>
    %v2973 = stablehlo.multiply %v2972, %v2972 : tensor<256x128x28x28xf32>
    %v2974 = stablehlo.reduce(%v2973 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2975 = stablehlo.broadcast_in_dim %v2974, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v2976 = stablehlo.divide %v2975, %v2968 : tensor<256x128x28x28xf32>
    %v2977 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v2978 = stablehlo.add %v2976, %v2977 : tensor<256x128x28x28xf32>
    %v2979 = stablehlo.rsqrt %v2978 : tensor<256x128x28x28xf32>
    %v2980 = stablehlo.multiply %v2972, %v2979 : tensor<256x128x28x28xf32>
    %v2981 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2982 = stablehlo.multiply %v2981, %v2980 : tensor<256x128x28x28xf32>
    %v2983 = stablehlo.reduce(%v2982 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2984 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2986 = stablehlo.reduce(%v2984 init: %v2985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2987 = stablehlo.reshape %v195 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v2988 = stablehlo.reshape %v2916 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2990 = stablehlo.pad %v2988, %v2989, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v2991 = stablehlo.transpose %v2987, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v2992 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v2993 = stablehlo.convolution(%v2991, %v2992)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<64x128x3x3xf32>
    %v2994 = stablehlo.transpose %v2993, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v2995 = stablehlo.reshape %v2916 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2997 = stablehlo.reduce(%v2995 init: %v2996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2998 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3000 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3001 = stablehlo.reduce(%v2998 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3003 = stablehlo.divide %v3002, %v3000 : tensor<256x128x28x28xf32>
    %v3004 = stablehlo.subtract %v2998, %v3003 : tensor<256x128x28x28xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<256x128x28x28xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3008 = stablehlo.divide %v3007, %v3000 : tensor<256x128x28x28xf32>
    %v3009 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3010 = stablehlo.add %v3008, %v3009 : tensor<256x128x28x28xf32>
    %v3011 = stablehlo.rsqrt %v3010 : tensor<256x128x28x28xf32>
    %v3012 = stablehlo.multiply %v3004, %v3011 : tensor<256x128x28x28xf32>
    %v3013 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3014 = stablehlo.multiply %v3013, %v3012 : tensor<256x128x28x28xf32>
    %v3015 = stablehlo.reduce(%v3014 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3016 = stablehlo.reshape %v2811 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3018 = stablehlo.reduce(%v3016 init: %v3017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3019 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3020 = stablehlo.compare GT, %v193, %v3019 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3021 = stablehlo.select %v3020, %v2924, %v3019 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3022 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3024 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3025 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3026 = stablehlo.reduce(%v3022 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3027 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3028 = stablehlo.divide %v3027, %v3024 : tensor<256x64x56x56xf32>
    %v3029 = stablehlo.subtract %v3022, %v3028 : tensor<256x64x56x56xf32>
    %v3030 = stablehlo.multiply %v3029, %v3029 : tensor<256x64x56x56xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3031, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3033 = stablehlo.divide %v3032, %v3024 : tensor<256x64x56x56xf32>
    %v3034 = stablehlo.add %v3033, %v3025 : tensor<256x64x56x56xf32>
    %v3035 = stablehlo.rsqrt %v3034 : tensor<256x64x56x56xf32>
    %v3036 = stablehlo.multiply %v3029, %v3035 : tensor<256x64x56x56xf32>
    %v3037 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3038 = stablehlo.reshape %v3021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3039 = stablehlo.multiply %v3037, %v3038 : tensor<256x64x56x56xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3042 = stablehlo.multiply %v3036, %v3039 : tensor<256x64x56x56xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3045 = stablehlo.multiply %v3039, %v3024 : tensor<256x64x56x56xf32>
    %v3046 = stablehlo.subtract %v3045, %v3041 : tensor<256x64x56x56xf32>
    %v3047 = stablehlo.multiply %v3036, %v3044 : tensor<256x64x56x56xf32>
    %v3048 = stablehlo.subtract %v3046, %v3047 : tensor<256x64x56x56xf32>
    %v3049 = stablehlo.divide %v3035, %v3024 : tensor<256x64x56x56xf32>
    %v3050 = stablehlo.multiply %v3049, %v3048 : tensor<256x64x56x56xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3053 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3054 = stablehlo.transpose %v3053, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3055 = stablehlo.convolution(%v3052, %v3054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3057 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3058 = stablehlo.compare GT, %v165, %v3057 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3059 = stablehlo.select %v3058, %v3056, %v3057 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3060 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3063 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3064 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3065 = stablehlo.broadcast_in_dim %v3064, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3066 = stablehlo.divide %v3065, %v3062 : tensor<256x64x56x56xf32>
    %v3067 = stablehlo.subtract %v3060, %v3066 : tensor<256x64x56x56xf32>
    %v3068 = stablehlo.multiply %v3067, %v3067 : tensor<256x64x56x56xf32>
    %v3069 = stablehlo.reduce(%v3068 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3070 = stablehlo.broadcast_in_dim %v3069, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3071 = stablehlo.divide %v3070, %v3062 : tensor<256x64x56x56xf32>
    %v3072 = stablehlo.add %v3071, %v3063 : tensor<256x64x56x56xf32>
    %v3073 = stablehlo.rsqrt %v3072 : tensor<256x64x56x56xf32>
    %v3074 = stablehlo.multiply %v3067, %v3073 : tensor<256x64x56x56xf32>
    %v3075 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3076 = stablehlo.reshape %v3059 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3077 = stablehlo.multiply %v3075, %v3076 : tensor<256x64x56x56xf32>
    %v3078 = stablehlo.reduce(%v3077 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3079 = stablehlo.broadcast_in_dim %v3078, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3080 = stablehlo.multiply %v3074, %v3077 : tensor<256x64x56x56xf32>
    %v3081 = stablehlo.reduce(%v3080 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3082 = stablehlo.broadcast_in_dim %v3081, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3083 = stablehlo.multiply %v3077, %v3062 : tensor<256x64x56x56xf32>
    %v3084 = stablehlo.subtract %v3083, %v3079 : tensor<256x64x56x56xf32>
    %v3085 = stablehlo.multiply %v3074, %v3082 : tensor<256x64x56x56xf32>
    %v3086 = stablehlo.subtract %v3084, %v3085 : tensor<256x64x56x56xf32>
    %v3087 = stablehlo.divide %v3073, %v3062 : tensor<256x64x56x56xf32>
    %v3088 = stablehlo.multiply %v3087, %v3086 : tensor<256x64x56x56xf32>
    %v3089 = stablehlo.reshape %v3088 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3090 = stablehlo.reshape %v3089 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3091 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3092 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3093 = stablehlo.convolution(%v3090, %v3092)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3095 = stablehlo.add %v3094, %v3021 : tensor<256x200704xf32>
    %v3096 = stablehlo.reshape %v140 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3097 = stablehlo.reshape %v3089 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3098 = stablehlo.transpose %v3096, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3099 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3100 = stablehlo.convolution(%v3098, %v3099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3101 = stablehlo.transpose %v3100, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3102 = stablehlo.reshape %v3089 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3104 = stablehlo.reduce(%v3102 init: %v3103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3105 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3107 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3108 = stablehlo.reduce(%v3105 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3110 = stablehlo.divide %v3109, %v3107 : tensor<256x64x56x56xf32>
    %v3111 = stablehlo.subtract %v3105, %v3110 : tensor<256x64x56x56xf32>
    %v3112 = stablehlo.multiply %v3111, %v3111 : tensor<256x64x56x56xf32>
    %v3113 = stablehlo.reduce(%v3112 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3114 = stablehlo.broadcast_in_dim %v3113, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3115 = stablehlo.divide %v3114, %v3107 : tensor<256x64x56x56xf32>
    %v3116 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3117 = stablehlo.add %v3115, %v3116 : tensor<256x64x56x56xf32>
    %v3118 = stablehlo.rsqrt %v3117 : tensor<256x64x56x56xf32>
    %v3119 = stablehlo.multiply %v3111, %v3118 : tensor<256x64x56x56xf32>
    %v3120 = stablehlo.reshape %v3059 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3121 = stablehlo.multiply %v3120, %v3119 : tensor<256x64x56x56xf32>
    %v3122 = stablehlo.reduce(%v3121 init: %v3106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3123 = stablehlo.reshape %v3059 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3125 = stablehlo.reduce(%v3123 init: %v3124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3126 = stablehlo.reshape %v167 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3127 = stablehlo.reshape %v3051 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3128 = stablehlo.transpose %v3126, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3129 = stablehlo.transpose %v3127, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3130 = stablehlo.convolution(%v3128, %v3129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3131 = stablehlo.transpose %v3130, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3132 = stablehlo.reshape %v3051 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3135 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3137 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3138 = stablehlo.reduce(%v3135 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3139 = stablehlo.broadcast_in_dim %v3138, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3140 = stablehlo.divide %v3139, %v3137 : tensor<256x64x56x56xf32>
    %v3141 = stablehlo.subtract %v3135, %v3140 : tensor<256x64x56x56xf32>
    %v3142 = stablehlo.multiply %v3141, %v3141 : tensor<256x64x56x56xf32>
    %v3143 = stablehlo.reduce(%v3142 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3144 = stablehlo.broadcast_in_dim %v3143, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3145 = stablehlo.divide %v3144, %v3137 : tensor<256x64x56x56xf32>
    %v3146 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3147 = stablehlo.add %v3145, %v3146 : tensor<256x64x56x56xf32>
    %v3148 = stablehlo.rsqrt %v3147 : tensor<256x64x56x56xf32>
    %v3149 = stablehlo.multiply %v3141, %v3148 : tensor<256x64x56x56xf32>
    %v3150 = stablehlo.reshape %v3021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3151 = stablehlo.multiply %v3150, %v3149 : tensor<256x64x56x56xf32>
    %v3152 = stablehlo.reduce(%v3151 init: %v3136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3153 = stablehlo.reshape %v3021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3155 = stablehlo.reduce(%v3153 init: %v3154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3156 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3157 = stablehlo.compare GT, %v138, %v3156 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3158 = stablehlo.select %v3157, %v3095, %v3156 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3159 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3161 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3162 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3163 = stablehlo.reduce(%v3159 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3164 = stablehlo.broadcast_in_dim %v3163, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3165 = stablehlo.divide %v3164, %v3161 : tensor<256x64x56x56xf32>
    %v3166 = stablehlo.subtract %v3159, %v3165 : tensor<256x64x56x56xf32>
    %v3167 = stablehlo.multiply %v3166, %v3166 : tensor<256x64x56x56xf32>
    %v3168 = stablehlo.reduce(%v3167 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3169 = stablehlo.broadcast_in_dim %v3168, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3170 = stablehlo.divide %v3169, %v3161 : tensor<256x64x56x56xf32>
    %v3171 = stablehlo.add %v3170, %v3162 : tensor<256x64x56x56xf32>
    %v3172 = stablehlo.rsqrt %v3171 : tensor<256x64x56x56xf32>
    %v3173 = stablehlo.multiply %v3166, %v3172 : tensor<256x64x56x56xf32>
    %v3174 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3175 = stablehlo.reshape %v3158 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3176 = stablehlo.multiply %v3174, %v3175 : tensor<256x64x56x56xf32>
    %v3177 = stablehlo.reduce(%v3176 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3178 = stablehlo.broadcast_in_dim %v3177, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3179 = stablehlo.multiply %v3173, %v3176 : tensor<256x64x56x56xf32>
    %v3180 = stablehlo.reduce(%v3179 init: %v3160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3181 = stablehlo.broadcast_in_dim %v3180, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3182 = stablehlo.multiply %v3176, %v3161 : tensor<256x64x56x56xf32>
    %v3183 = stablehlo.subtract %v3182, %v3178 : tensor<256x64x56x56xf32>
    %v3184 = stablehlo.multiply %v3173, %v3181 : tensor<256x64x56x56xf32>
    %v3185 = stablehlo.subtract %v3183, %v3184 : tensor<256x64x56x56xf32>
    %v3186 = stablehlo.divide %v3172, %v3161 : tensor<256x64x56x56xf32>
    %v3187 = stablehlo.multiply %v3186, %v3185 : tensor<256x64x56x56xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3190 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3191 = stablehlo.transpose %v3190, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3192 = stablehlo.convolution(%v3189, %v3191)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3193 = stablehlo.reshape %v3192 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3195 = stablehlo.compare GT, %v110, %v3194 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3196 = stablehlo.select %v3195, %v3193, %v3194 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3197 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3199 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3200 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3201 = stablehlo.reduce(%v3197 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3203 = stablehlo.divide %v3202, %v3199 : tensor<256x64x56x56xf32>
    %v3204 = stablehlo.subtract %v3197, %v3203 : tensor<256x64x56x56xf32>
    %v3205 = stablehlo.multiply %v3204, %v3204 : tensor<256x64x56x56xf32>
    %v3206 = stablehlo.reduce(%v3205 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3207 = stablehlo.broadcast_in_dim %v3206, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3208 = stablehlo.divide %v3207, %v3199 : tensor<256x64x56x56xf32>
    %v3209 = stablehlo.add %v3208, %v3200 : tensor<256x64x56x56xf32>
    %v3210 = stablehlo.rsqrt %v3209 : tensor<256x64x56x56xf32>
    %v3211 = stablehlo.multiply %v3204, %v3210 : tensor<256x64x56x56xf32>
    %v3212 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3213 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3214 = stablehlo.multiply %v3212, %v3213 : tensor<256x64x56x56xf32>
    %v3215 = stablehlo.reduce(%v3214 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3216 = stablehlo.broadcast_in_dim %v3215, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3217 = stablehlo.multiply %v3211, %v3214 : tensor<256x64x56x56xf32>
    %v3218 = stablehlo.reduce(%v3217 init: %v3198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3219 = stablehlo.broadcast_in_dim %v3218, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3220 = stablehlo.multiply %v3214, %v3199 : tensor<256x64x56x56xf32>
    %v3221 = stablehlo.subtract %v3220, %v3216 : tensor<256x64x56x56xf32>
    %v3222 = stablehlo.multiply %v3211, %v3219 : tensor<256x64x56x56xf32>
    %v3223 = stablehlo.subtract %v3221, %v3222 : tensor<256x64x56x56xf32>
    %v3224 = stablehlo.divide %v3210, %v3199 : tensor<256x64x56x56xf32>
    %v3225 = stablehlo.multiply %v3224, %v3223 : tensor<256x64x56x56xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3227 = stablehlo.reshape %v3226 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3228 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3229 = stablehlo.transpose %v3228, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3230 = stablehlo.convolution(%v3227, %v3229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3232 = stablehlo.add %v3231, %v3158 : tensor<256x200704xf32>
    %v3233 = stablehlo.reshape %v85 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3234 = stablehlo.reshape %v3226 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3235 = stablehlo.transpose %v3233, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3236 = stablehlo.transpose %v3234, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3237 = stablehlo.convolution(%v3235, %v3236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3238 = stablehlo.transpose %v3237, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3239 = stablehlo.reshape %v3226 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3241 = stablehlo.reduce(%v3239 init: %v3240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3242 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3243 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3244 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3245 = stablehlo.reduce(%v3242 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3247 = stablehlo.divide %v3246, %v3244 : tensor<256x64x56x56xf32>
    %v3248 = stablehlo.subtract %v3242, %v3247 : tensor<256x64x56x56xf32>
    %v3249 = stablehlo.multiply %v3248, %v3248 : tensor<256x64x56x56xf32>
    %v3250 = stablehlo.reduce(%v3249 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3250, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3252 = stablehlo.divide %v3251, %v3244 : tensor<256x64x56x56xf32>
    %v3253 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3254 = stablehlo.add %v3252, %v3253 : tensor<256x64x56x56xf32>
    %v3255 = stablehlo.rsqrt %v3254 : tensor<256x64x56x56xf32>
    %v3256 = stablehlo.multiply %v3248, %v3255 : tensor<256x64x56x56xf32>
    %v3257 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3258 = stablehlo.multiply %v3257, %v3256 : tensor<256x64x56x56xf32>
    %v3259 = stablehlo.reduce(%v3258 init: %v3243) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3260 = stablehlo.reshape %v3196 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.reduce(%v3260 init: %v3261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3263 = stablehlo.reshape %v112 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3264 = stablehlo.reshape %v3188 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3265 = stablehlo.transpose %v3263, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3266 = stablehlo.transpose %v3264, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3267 = stablehlo.convolution(%v3265, %v3266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3268 = stablehlo.transpose %v3267, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3269 = stablehlo.reshape %v3188 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3270 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3271 = stablehlo.reduce(%v3269 init: %v3270) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3272 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3274 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3275 = stablehlo.reduce(%v3272 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3276 = stablehlo.broadcast_in_dim %v3275, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3277 = stablehlo.divide %v3276, %v3274 : tensor<256x64x56x56xf32>
    %v3278 = stablehlo.subtract %v3272, %v3277 : tensor<256x64x56x56xf32>
    %v3279 = stablehlo.multiply %v3278, %v3278 : tensor<256x64x56x56xf32>
    %v3280 = stablehlo.reduce(%v3279 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3281 = stablehlo.broadcast_in_dim %v3280, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3282 = stablehlo.divide %v3281, %v3274 : tensor<256x64x56x56xf32>
    %v3283 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3284 = stablehlo.add %v3282, %v3283 : tensor<256x64x56x56xf32>
    %v3285 = stablehlo.rsqrt %v3284 : tensor<256x64x56x56xf32>
    %v3286 = stablehlo.multiply %v3278, %v3285 : tensor<256x64x56x56xf32>
    %v3287 = stablehlo.reshape %v3158 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3288 = stablehlo.multiply %v3287, %v3286 : tensor<256x64x56x56xf32>
    %v3289 = stablehlo.reduce(%v3288 init: %v3273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3290 = stablehlo.reshape %v3158 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3291 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3292 = stablehlo.reduce(%v3290 init: %v3291) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3294 = stablehlo.compare GT, %v83, %v3293 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3295 = stablehlo.select %v3294, %v3232, %v3293 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3296 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3298 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3299 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3300 = stablehlo.reduce(%v3296 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3300, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3302 = stablehlo.divide %v3301, %v3298 : tensor<256x64x56x56xf32>
    %v3303 = stablehlo.subtract %v3296, %v3302 : tensor<256x64x56x56xf32>
    %v3304 = stablehlo.multiply %v3303, %v3303 : tensor<256x64x56x56xf32>
    %v3305 = stablehlo.reduce(%v3304 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3306 = stablehlo.broadcast_in_dim %v3305, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3307 = stablehlo.divide %v3306, %v3298 : tensor<256x64x56x56xf32>
    %v3308 = stablehlo.add %v3307, %v3299 : tensor<256x64x56x56xf32>
    %v3309 = stablehlo.rsqrt %v3308 : tensor<256x64x56x56xf32>
    %v3310 = stablehlo.multiply %v3303, %v3309 : tensor<256x64x56x56xf32>
    %v3311 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3312 = stablehlo.reshape %v3295 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3313 = stablehlo.multiply %v3311, %v3312 : tensor<256x64x56x56xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3316 = stablehlo.multiply %v3310, %v3313 : tensor<256x64x56x56xf32>
    %v3317 = stablehlo.reduce(%v3316 init: %v3297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3318 = stablehlo.broadcast_in_dim %v3317, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3319 = stablehlo.multiply %v3313, %v3298 : tensor<256x64x56x56xf32>
    %v3320 = stablehlo.subtract %v3319, %v3315 : tensor<256x64x56x56xf32>
    %v3321 = stablehlo.multiply %v3310, %v3318 : tensor<256x64x56x56xf32>
    %v3322 = stablehlo.subtract %v3320, %v3321 : tensor<256x64x56x56xf32>
    %v3323 = stablehlo.divide %v3309, %v3298 : tensor<256x64x56x56xf32>
    %v3324 = stablehlo.multiply %v3323, %v3322 : tensor<256x64x56x56xf32>
    %v3325 = stablehlo.reshape %v3324 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3326 = stablehlo.reshape %v3325 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3327 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3328 = stablehlo.transpose %v3327, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3329 = stablehlo.convolution(%v3326, %v3328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3330 = stablehlo.reshape %v3329 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3331 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3332 = stablehlo.compare GT, %v55, %v3331 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3333 = stablehlo.select %v3332, %v3330, %v3331 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3334 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3336 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3337 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3338 = stablehlo.reduce(%v3334 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3340 = stablehlo.divide %v3339, %v3336 : tensor<256x64x56x56xf32>
    %v3341 = stablehlo.subtract %v3334, %v3340 : tensor<256x64x56x56xf32>
    %v3342 = stablehlo.multiply %v3341, %v3341 : tensor<256x64x56x56xf32>
    %v3343 = stablehlo.reduce(%v3342 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3345 = stablehlo.divide %v3344, %v3336 : tensor<256x64x56x56xf32>
    %v3346 = stablehlo.add %v3345, %v3337 : tensor<256x64x56x56xf32>
    %v3347 = stablehlo.rsqrt %v3346 : tensor<256x64x56x56xf32>
    %v3348 = stablehlo.multiply %v3341, %v3347 : tensor<256x64x56x56xf32>
    %v3349 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3350 = stablehlo.reshape %v3333 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3351 = stablehlo.multiply %v3349, %v3350 : tensor<256x64x56x56xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3353 = stablehlo.broadcast_in_dim %v3352, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3354 = stablehlo.multiply %v3348, %v3351 : tensor<256x64x56x56xf32>
    %v3355 = stablehlo.reduce(%v3354 init: %v3335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3356 = stablehlo.broadcast_in_dim %v3355, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3357 = stablehlo.multiply %v3351, %v3336 : tensor<256x64x56x56xf32>
    %v3358 = stablehlo.subtract %v3357, %v3353 : tensor<256x64x56x56xf32>
    %v3359 = stablehlo.multiply %v3348, %v3356 : tensor<256x64x56x56xf32>
    %v3360 = stablehlo.subtract %v3358, %v3359 : tensor<256x64x56x56xf32>
    %v3361 = stablehlo.divide %v3347, %v3336 : tensor<256x64x56x56xf32>
    %v3362 = stablehlo.multiply %v3361, %v3360 : tensor<256x64x56x56xf32>
    %v3363 = stablehlo.reshape %v3362 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3365 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3366 = stablehlo.transpose %v3365, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3367 = stablehlo.convolution(%v3364, %v3366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v3368 = stablehlo.reshape %v3367 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v3369 = stablehlo.add %v3368, %v3295 : tensor<256x200704xf32>
    %v3370 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3371 = stablehlo.reshape %v3363 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3372 = stablehlo.transpose %v3370, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3373 = stablehlo.transpose %v3371, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3374 = stablehlo.convolution(%v3372, %v3373)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3375 = stablehlo.transpose %v3374, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3376 = stablehlo.reshape %v3363 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3378 = stablehlo.reduce(%v3376 init: %v3377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3379 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3381 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3382 = stablehlo.reduce(%v3379 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3383 = stablehlo.broadcast_in_dim %v3382, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3384 = stablehlo.divide %v3383, %v3381 : tensor<256x64x56x56xf32>
    %v3385 = stablehlo.subtract %v3379, %v3384 : tensor<256x64x56x56xf32>
    %v3386 = stablehlo.multiply %v3385, %v3385 : tensor<256x64x56x56xf32>
    %v3387 = stablehlo.reduce(%v3386 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3388 = stablehlo.broadcast_in_dim %v3387, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3389 = stablehlo.divide %v3388, %v3381 : tensor<256x64x56x56xf32>
    %v3390 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3391 = stablehlo.add %v3389, %v3390 : tensor<256x64x56x56xf32>
    %v3392 = stablehlo.rsqrt %v3391 : tensor<256x64x56x56xf32>
    %v3393 = stablehlo.multiply %v3385, %v3392 : tensor<256x64x56x56xf32>
    %v3394 = stablehlo.reshape %v3333 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3395 = stablehlo.multiply %v3394, %v3393 : tensor<256x64x56x56xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3397 = stablehlo.reshape %v3333 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3400 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3401 = stablehlo.reshape %v3325 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3402 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3403 = stablehlo.transpose %v3401, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v3404 = stablehlo.convolution(%v3402, %v3403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v3405 = stablehlo.transpose %v3404, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3406 = stablehlo.reshape %v3325 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3408 = stablehlo.reduce(%v3406 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3409 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3411 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3412 = stablehlo.reduce(%v3409 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3414 = stablehlo.divide %v3413, %v3411 : tensor<256x64x56x56xf32>
    %v3415 = stablehlo.subtract %v3409, %v3414 : tensor<256x64x56x56xf32>
    %v3416 = stablehlo.multiply %v3415, %v3415 : tensor<256x64x56x56xf32>
    %v3417 = stablehlo.reduce(%v3416 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3418 = stablehlo.broadcast_in_dim %v3417, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3419 = stablehlo.divide %v3418, %v3411 : tensor<256x64x56x56xf32>
    %v3420 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v3421 = stablehlo.add %v3419, %v3420 : tensor<256x64x56x56xf32>
    %v3422 = stablehlo.rsqrt %v3421 : tensor<256x64x56x56xf32>
    %v3423 = stablehlo.multiply %v3415, %v3422 : tensor<256x64x56x56xf32>
    %v3424 = stablehlo.reshape %v3295 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3425 = stablehlo.multiply %v3424, %v3423 : tensor<256x64x56x56xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3427 = stablehlo.reshape %v3295 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3429 = stablehlo.reduce(%v3427 init: %v3428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3430 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3431 = stablehlo.reshape %v3369 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3433 = "stablehlo.select_and_scatter"(%v3430, %v3431, %v3432) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<256x64x112x112xf32>, tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x112x112xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v3436 = stablehlo.compare GT, %v24, %v3435 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v3437 = stablehlo.select %v3436, %v3434, %v3435 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v3438 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3440 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3441 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v3442 = stablehlo.reduce(%v3438 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3444 = stablehlo.divide %v3443, %v3440 : tensor<256x64x112x112xf32>
    %v3445 = stablehlo.subtract %v3438, %v3444 : tensor<256x64x112x112xf32>
    %v3446 = stablehlo.multiply %v3445, %v3445 : tensor<256x64x112x112xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3449 = stablehlo.divide %v3448, %v3440 : tensor<256x64x112x112xf32>
    %v3450 = stablehlo.add %v3449, %v3441 : tensor<256x64x112x112xf32>
    %v3451 = stablehlo.rsqrt %v3450 : tensor<256x64x112x112xf32>
    %v3452 = stablehlo.multiply %v3445, %v3451 : tensor<256x64x112x112xf32>
    %v3453 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3454 = stablehlo.reshape %v3437 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3455 = stablehlo.multiply %v3453, %v3454 : tensor<256x64x112x112xf32>
    %v3456 = stablehlo.reduce(%v3455 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3457 = stablehlo.broadcast_in_dim %v3456, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3458 = stablehlo.multiply %v3452, %v3455 : tensor<256x64x112x112xf32>
    %v3459 = stablehlo.reduce(%v3458 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3460 = stablehlo.broadcast_in_dim %v3459, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3461 = stablehlo.multiply %v3455, %v3440 : tensor<256x64x112x112xf32>
    %v3462 = stablehlo.subtract %v3461, %v3457 : tensor<256x64x112x112xf32>
    %v3463 = stablehlo.multiply %v3452, %v3460 : tensor<256x64x112x112xf32>
    %v3464 = stablehlo.subtract %v3462, %v3463 : tensor<256x64x112x112xf32>
    %v3465 = stablehlo.divide %v3451, %v3440 : tensor<256x64x112x112xf32>
    %v3466 = stablehlo.multiply %v3465, %v3464 : tensor<256x64x112x112xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v3468 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v3469 = stablehlo.reshape %v3467 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3471 = stablehlo.pad %v3469, %v3470, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x224x224xf32>
    %v3472 = stablehlo.transpose %v3468, dims = [1, 0, 2, 3] : (tensor<256x3x224x224xf32>) -> tensor<3x256x224x224xf32>
    %v3473 = stablehlo.transpose %v3471, dims = [1, 0, 2, 3] : (tensor<256x64x224x224xf32>) -> tensor<64x256x224x224xf32>
    %v3474 = stablehlo.convolution(%v3472, %v3473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x224x224xf32>, tensor<64x256x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v3475 = stablehlo.transpose %v3474, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3476 = stablehlo.reshape %v3467 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3478 = stablehlo.reduce(%v3476 init: %v3477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3479 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3482 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3482, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3484 = stablehlo.divide %v3483, %v3481 : tensor<256x64x112x112xf32>
    %v3485 = stablehlo.subtract %v3479, %v3484 : tensor<256x64x112x112xf32>
    %v3486 = stablehlo.multiply %v3485, %v3485 : tensor<256x64x112x112xf32>
    %v3487 = stablehlo.reduce(%v3486 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3489 = stablehlo.divide %v3488, %v3481 : tensor<256x64x112x112xf32>
    %v3490 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v3491 = stablehlo.add %v3489, %v3490 : tensor<256x64x112x112xf32>
    %v3492 = stablehlo.rsqrt %v3491 : tensor<256x64x112x112xf32>
    %v3493 = stablehlo.multiply %v3485, %v3492 : tensor<256x64x112x112xf32>
    %v3494 = stablehlo.reshape %v3437 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3495 = stablehlo.multiply %v3494, %v3493 : tensor<256x64x112x112xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.reshape %v3437 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3499 = stablehlo.reduce(%v3497 init: %v3498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3502 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v3503 = stablehlo.reduce(%v3500 init: %v3501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3504 = stablehlo.divide %v3503, %v3502 : tensor<64xf32>
    %v3505 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v3506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3507 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v3508 = stablehlo.reduce(%v3505 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3509 = stablehlo.broadcast_in_dim %v3508, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3510 = stablehlo.divide %v3509, %v3507 : tensor<256x64x112x112xf32>
    %v3511 = stablehlo.subtract %v3505, %v3510 : tensor<256x64x112x112xf32>
    %v3512 = stablehlo.multiply %v3511, %v3511 : tensor<256x64x112x112xf32>
    %v3513 = stablehlo.reduce(%v3512 init: %v3506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3514 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v3515 = stablehlo.divide %v3513, %v3514 : tensor<64xf32>
    %v3516 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3518 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3519 = stablehlo.reduce(%v3516 init: %v3517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3520 = stablehlo.divide %v3519, %v3518 : tensor<64xf32>
    %v3521 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3523 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3524 = stablehlo.reduce(%v3521 init: %v3522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3526 = stablehlo.divide %v3525, %v3523 : tensor<256x64x56x56xf32>
    %v3527 = stablehlo.subtract %v3521, %v3526 : tensor<256x64x56x56xf32>
    %v3528 = stablehlo.multiply %v3527, %v3527 : tensor<256x64x56x56xf32>
    %v3529 = stablehlo.reduce(%v3528 init: %v3522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3530 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3531 = stablehlo.divide %v3529, %v3530 : tensor<64xf32>
    %v3532 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3534 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3535 = stablehlo.reduce(%v3532 init: %v3533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3536 = stablehlo.divide %v3535, %v3534 : tensor<64xf32>
    %v3537 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3539 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3540 = stablehlo.reduce(%v3537 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3541 = stablehlo.broadcast_in_dim %v3540, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3542 = stablehlo.divide %v3541, %v3539 : tensor<256x64x56x56xf32>
    %v3543 = stablehlo.subtract %v3537, %v3542 : tensor<256x64x56x56xf32>
    %v3544 = stablehlo.multiply %v3543, %v3543 : tensor<256x64x56x56xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3546 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3547 = stablehlo.divide %v3545, %v3546 : tensor<64xf32>
    %v3548 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3550 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3551 = stablehlo.reduce(%v3548 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3552 = stablehlo.divide %v3551, %v3550 : tensor<64xf32>
    %v3553 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3555 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3556 = stablehlo.reduce(%v3553 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3557 = stablehlo.broadcast_in_dim %v3556, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3558 = stablehlo.divide %v3557, %v3555 : tensor<256x64x56x56xf32>
    %v3559 = stablehlo.subtract %v3553, %v3558 : tensor<256x64x56x56xf32>
    %v3560 = stablehlo.multiply %v3559, %v3559 : tensor<256x64x56x56xf32>
    %v3561 = stablehlo.reduce(%v3560 init: %v3554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3562 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3563 = stablehlo.divide %v3561, %v3562 : tensor<64xf32>
    %v3564 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3566 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3567 = stablehlo.reduce(%v3564 init: %v3565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3568 = stablehlo.divide %v3567, %v3566 : tensor<64xf32>
    %v3569 = stablehlo.reshape %v117 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3571 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3572 = stablehlo.reduce(%v3569 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3573 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3574 = stablehlo.divide %v3573, %v3571 : tensor<256x64x56x56xf32>
    %v3575 = stablehlo.subtract %v3569, %v3574 : tensor<256x64x56x56xf32>
    %v3576 = stablehlo.multiply %v3575, %v3575 : tensor<256x64x56x56xf32>
    %v3577 = stablehlo.reduce(%v3576 init: %v3570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3578 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3579 = stablehlo.divide %v3577, %v3578 : tensor<64xf32>
    %v3580 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3582 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3583 = stablehlo.reduce(%v3580 init: %v3581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3584 = stablehlo.divide %v3583, %v3582 : tensor<64xf32>
    %v3585 = stablehlo.reshape %v145 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3588 = stablehlo.reduce(%v3585 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3589 = stablehlo.broadcast_in_dim %v3588, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3590 = stablehlo.divide %v3589, %v3587 : tensor<256x64x56x56xf32>
    %v3591 = stablehlo.subtract %v3585, %v3590 : tensor<256x64x56x56xf32>
    %v3592 = stablehlo.multiply %v3591, %v3591 : tensor<256x64x56x56xf32>
    %v3593 = stablehlo.reduce(%v3592 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3594 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3595 = stablehlo.divide %v3593, %v3594 : tensor<64xf32>
    %v3596 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3598 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3599 = stablehlo.reduce(%v3596 init: %v3597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3600 = stablehlo.divide %v3599, %v3598 : tensor<64xf32>
    %v3601 = stablehlo.reshape %v172 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v3602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3603 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v3604 = stablehlo.reduce(%v3601 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3605 = stablehlo.broadcast_in_dim %v3604, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v3606 = stablehlo.divide %v3605, %v3603 : tensor<256x64x56x56xf32>
    %v3607 = stablehlo.subtract %v3601, %v3606 : tensor<256x64x56x56xf32>
    %v3608 = stablehlo.multiply %v3607, %v3607 : tensor<256x64x56x56xf32>
    %v3609 = stablehlo.reduce(%v3608 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3610 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3611 = stablehlo.divide %v3609, %v3610 : tensor<64xf32>
    %v3612 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3614 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3615 = stablehlo.reduce(%v3612 init: %v3613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3616 = stablehlo.divide %v3615, %v3614 : tensor<128xf32>
    %v3617 = stablehlo.reshape %v200 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3619 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3620 = stablehlo.reduce(%v3617 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3621 = stablehlo.broadcast_in_dim %v3620, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3622 = stablehlo.divide %v3621, %v3619 : tensor<256x128x28x28xf32>
    %v3623 = stablehlo.subtract %v3617, %v3622 : tensor<256x128x28x28xf32>
    %v3624 = stablehlo.multiply %v3623, %v3623 : tensor<256x128x28x28xf32>
    %v3625 = stablehlo.reduce(%v3624 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3626 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3627 = stablehlo.divide %v3625, %v3626 : tensor<128xf32>
    %v3628 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3631 = stablehlo.reduce(%v3628 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3632 = stablehlo.divide %v3631, %v3630 : tensor<128xf32>
    %v3633 = stablehlo.reshape %v227 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3635 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3636 = stablehlo.reduce(%v3633 init: %v3634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3637 = stablehlo.broadcast_in_dim %v3636, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3638 = stablehlo.divide %v3637, %v3635 : tensor<256x128x28x28xf32>
    %v3639 = stablehlo.subtract %v3633, %v3638 : tensor<256x128x28x28xf32>
    %v3640 = stablehlo.multiply %v3639, %v3639 : tensor<256x128x28x28xf32>
    %v3641 = stablehlo.reduce(%v3640 init: %v3634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3642 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3643 = stablehlo.divide %v3641, %v3642 : tensor<128xf32>
    %v3644 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3647 = stablehlo.reduce(%v3644 init: %v3645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3648 = stablehlo.divide %v3647, %v3646 : tensor<128xf32>
    %v3649 = stablehlo.reshape %v252 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3651 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3652 = stablehlo.reduce(%v3649 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3653 = stablehlo.broadcast_in_dim %v3652, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3654 = stablehlo.divide %v3653, %v3651 : tensor<256x128x28x28xf32>
    %v3655 = stablehlo.subtract %v3649, %v3654 : tensor<256x128x28x28xf32>
    %v3656 = stablehlo.multiply %v3655, %v3655 : tensor<256x128x28x28xf32>
    %v3657 = stablehlo.reduce(%v3656 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3658 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3659 = stablehlo.divide %v3657, %v3658 : tensor<128xf32>
    %v3660 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3662 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3663 = stablehlo.reduce(%v3660 init: %v3661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3664 = stablehlo.divide %v3663, %v3662 : tensor<128xf32>
    %v3665 = stablehlo.reshape %v280 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3667 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3668 = stablehlo.reduce(%v3665 init: %v3666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3670 = stablehlo.divide %v3669, %v3667 : tensor<256x128x28x28xf32>
    %v3671 = stablehlo.subtract %v3665, %v3670 : tensor<256x128x28x28xf32>
    %v3672 = stablehlo.multiply %v3671, %v3671 : tensor<256x128x28x28xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3674 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3675 = stablehlo.divide %v3673, %v3674 : tensor<128xf32>
    %v3676 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3678 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3679 = stablehlo.reduce(%v3676 init: %v3677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3680 = stablehlo.divide %v3679, %v3678 : tensor<128xf32>
    %v3681 = stablehlo.reshape %v307 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3683 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3684 = stablehlo.reduce(%v3681 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3685 = stablehlo.broadcast_in_dim %v3684, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3686 = stablehlo.divide %v3685, %v3683 : tensor<256x128x28x28xf32>
    %v3687 = stablehlo.subtract %v3681, %v3686 : tensor<256x128x28x28xf32>
    %v3688 = stablehlo.multiply %v3687, %v3687 : tensor<256x128x28x28xf32>
    %v3689 = stablehlo.reduce(%v3688 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3690 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3691 = stablehlo.divide %v3689, %v3690 : tensor<128xf32>
    %v3692 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3696 = stablehlo.divide %v3695, %v3694 : tensor<128xf32>
    %v3697 = stablehlo.reshape %v335 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3699 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3700 = stablehlo.reduce(%v3697 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3702 = stablehlo.divide %v3701, %v3699 : tensor<256x128x28x28xf32>
    %v3703 = stablehlo.subtract %v3697, %v3702 : tensor<256x128x28x28xf32>
    %v3704 = stablehlo.multiply %v3703, %v3703 : tensor<256x128x28x28xf32>
    %v3705 = stablehlo.reduce(%v3704 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3706 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3707 = stablehlo.divide %v3705, %v3706 : tensor<128xf32>
    %v3708 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3710 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3711 = stablehlo.reduce(%v3708 init: %v3709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3712 = stablehlo.divide %v3711, %v3710 : tensor<128xf32>
    %v3713 = stablehlo.reshape %v362 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3716 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3717 = stablehlo.broadcast_in_dim %v3716, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3718 = stablehlo.divide %v3717, %v3715 : tensor<256x128x28x28xf32>
    %v3719 = stablehlo.subtract %v3713, %v3718 : tensor<256x128x28x28xf32>
    %v3720 = stablehlo.multiply %v3719, %v3719 : tensor<256x128x28x28xf32>
    %v3721 = stablehlo.reduce(%v3720 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3722 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3723 = stablehlo.divide %v3721, %v3722 : tensor<128xf32>
    %v3724 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3726 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3727 = stablehlo.reduce(%v3724 init: %v3725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3728 = stablehlo.divide %v3727, %v3726 : tensor<128xf32>
    %v3729 = stablehlo.reshape %v390 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3734 = stablehlo.divide %v3733, %v3731 : tensor<256x128x28x28xf32>
    %v3735 = stablehlo.subtract %v3729, %v3734 : tensor<256x128x28x28xf32>
    %v3736 = stablehlo.multiply %v3735, %v3735 : tensor<256x128x28x28xf32>
    %v3737 = stablehlo.reduce(%v3736 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3738 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3739 = stablehlo.divide %v3737, %v3738 : tensor<128xf32>
    %v3740 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3742 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3743 = stablehlo.reduce(%v3740 init: %v3741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3744 = stablehlo.divide %v3743, %v3742 : tensor<128xf32>
    %v3745 = stablehlo.reshape %v417 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3748 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3749 = stablehlo.broadcast_in_dim %v3748, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3750 = stablehlo.divide %v3749, %v3747 : tensor<256x128x28x28xf32>
    %v3751 = stablehlo.subtract %v3745, %v3750 : tensor<256x128x28x28xf32>
    %v3752 = stablehlo.multiply %v3751, %v3751 : tensor<256x128x28x28xf32>
    %v3753 = stablehlo.reduce(%v3752 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3754 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v3755 = stablehlo.divide %v3753, %v3754 : tensor<128xf32>
    %v3756 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3758 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3759 = stablehlo.reduce(%v3756 init: %v3757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3760 = stablehlo.divide %v3759, %v3758 : tensor<256xf32>
    %v3761 = stablehlo.reshape %v445 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3764 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3765 = stablehlo.broadcast_in_dim %v3764, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3766 = stablehlo.divide %v3765, %v3763 : tensor<256x256x14x14xf32>
    %v3767 = stablehlo.subtract %v3761, %v3766 : tensor<256x256x14x14xf32>
    %v3768 = stablehlo.multiply %v3767, %v3767 : tensor<256x256x14x14xf32>
    %v3769 = stablehlo.reduce(%v3768 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3770 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3771 = stablehlo.divide %v3769, %v3770 : tensor<256xf32>
    %v3772 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3774 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3775 = stablehlo.reduce(%v3772 init: %v3773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3776 = stablehlo.divide %v3775, %v3774 : tensor<256xf32>
    %v3777 = stablehlo.reshape %v472 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3781 = stablehlo.broadcast_in_dim %v3780, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3782 = stablehlo.divide %v3781, %v3779 : tensor<256x256x14x14xf32>
    %v3783 = stablehlo.subtract %v3777, %v3782 : tensor<256x256x14x14xf32>
    %v3784 = stablehlo.multiply %v3783, %v3783 : tensor<256x256x14x14xf32>
    %v3785 = stablehlo.reduce(%v3784 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3786 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3787 = stablehlo.divide %v3785, %v3786 : tensor<256xf32>
    %v3788 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3790 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3791 = stablehlo.reduce(%v3788 init: %v3789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3792 = stablehlo.divide %v3791, %v3790 : tensor<256xf32>
    %v3793 = stablehlo.reshape %v497 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3796 = stablehlo.reduce(%v3793 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3797 = stablehlo.broadcast_in_dim %v3796, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3798 = stablehlo.divide %v3797, %v3795 : tensor<256x256x14x14xf32>
    %v3799 = stablehlo.subtract %v3793, %v3798 : tensor<256x256x14x14xf32>
    %v3800 = stablehlo.multiply %v3799, %v3799 : tensor<256x256x14x14xf32>
    %v3801 = stablehlo.reduce(%v3800 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3802 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3803 = stablehlo.divide %v3801, %v3802 : tensor<256xf32>
    %v3804 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3806 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3807 = stablehlo.reduce(%v3804 init: %v3805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3808 = stablehlo.divide %v3807, %v3806 : tensor<256xf32>
    %v3809 = stablehlo.reshape %v525 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3814 = stablehlo.divide %v3813, %v3811 : tensor<256x256x14x14xf32>
    %v3815 = stablehlo.subtract %v3809, %v3814 : tensor<256x256x14x14xf32>
    %v3816 = stablehlo.multiply %v3815, %v3815 : tensor<256x256x14x14xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3818 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3819 = stablehlo.divide %v3817, %v3818 : tensor<256xf32>
    %v3820 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3822 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3823 = stablehlo.reduce(%v3820 init: %v3821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3824 = stablehlo.divide %v3823, %v3822 : tensor<256xf32>
    %v3825 = stablehlo.reshape %v552 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3828 = stablehlo.reduce(%v3825 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3829 = stablehlo.broadcast_in_dim %v3828, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3830 = stablehlo.divide %v3829, %v3827 : tensor<256x256x14x14xf32>
    %v3831 = stablehlo.subtract %v3825, %v3830 : tensor<256x256x14x14xf32>
    %v3832 = stablehlo.multiply %v3831, %v3831 : tensor<256x256x14x14xf32>
    %v3833 = stablehlo.reduce(%v3832 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3834 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3835 = stablehlo.divide %v3833, %v3834 : tensor<256xf32>
    %v3836 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3838 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3839 = stablehlo.reduce(%v3836 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3840 = stablehlo.divide %v3839, %v3838 : tensor<256xf32>
    %v3841 = stablehlo.reshape %v580 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3843 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3844 = stablehlo.reduce(%v3841 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3845 = stablehlo.broadcast_in_dim %v3844, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3846 = stablehlo.divide %v3845, %v3843 : tensor<256x256x14x14xf32>
    %v3847 = stablehlo.subtract %v3841, %v3846 : tensor<256x256x14x14xf32>
    %v3848 = stablehlo.multiply %v3847, %v3847 : tensor<256x256x14x14xf32>
    %v3849 = stablehlo.reduce(%v3848 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3850 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3851 = stablehlo.divide %v3849, %v3850 : tensor<256xf32>
    %v3852 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3854 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3855 = stablehlo.reduce(%v3852 init: %v3853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3856 = stablehlo.divide %v3855, %v3854 : tensor<256xf32>
    %v3857 = stablehlo.reshape %v607 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3860 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3861 = stablehlo.broadcast_in_dim %v3860, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3862 = stablehlo.divide %v3861, %v3859 : tensor<256x256x14x14xf32>
    %v3863 = stablehlo.subtract %v3857, %v3862 : tensor<256x256x14x14xf32>
    %v3864 = stablehlo.multiply %v3863, %v3863 : tensor<256x256x14x14xf32>
    %v3865 = stablehlo.reduce(%v3864 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3866 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3867 = stablehlo.divide %v3865, %v3866 : tensor<256xf32>
    %v3868 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3870 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3871 = stablehlo.reduce(%v3868 init: %v3869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3872 = stablehlo.divide %v3871, %v3870 : tensor<256xf32>
    %v3873 = stablehlo.reshape %v635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3875 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3876 = stablehlo.reduce(%v3873 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3877 = stablehlo.broadcast_in_dim %v3876, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3878 = stablehlo.divide %v3877, %v3875 : tensor<256x256x14x14xf32>
    %v3879 = stablehlo.subtract %v3873, %v3878 : tensor<256x256x14x14xf32>
    %v3880 = stablehlo.multiply %v3879, %v3879 : tensor<256x256x14x14xf32>
    %v3881 = stablehlo.reduce(%v3880 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3882 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3883 = stablehlo.divide %v3881, %v3882 : tensor<256xf32>
    %v3884 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3887 = stablehlo.reduce(%v3884 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3888 = stablehlo.divide %v3887, %v3886 : tensor<256xf32>
    %v3889 = stablehlo.reshape %v662 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3892 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3893 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3894 = stablehlo.divide %v3893, %v3891 : tensor<256x256x14x14xf32>
    %v3895 = stablehlo.subtract %v3889, %v3894 : tensor<256x256x14x14xf32>
    %v3896 = stablehlo.multiply %v3895, %v3895 : tensor<256x256x14x14xf32>
    %v3897 = stablehlo.reduce(%v3896 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3898 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3899 = stablehlo.divide %v3897, %v3898 : tensor<256xf32>
    %v3900 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3902 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3903 = stablehlo.reduce(%v3900 init: %v3901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3904 = stablehlo.divide %v3903, %v3902 : tensor<256xf32>
    %v3905 = stablehlo.reshape %v690 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3908 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3909 = stablehlo.broadcast_in_dim %v3908, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3910 = stablehlo.divide %v3909, %v3907 : tensor<256x256x14x14xf32>
    %v3911 = stablehlo.subtract %v3905, %v3910 : tensor<256x256x14x14xf32>
    %v3912 = stablehlo.multiply %v3911, %v3911 : tensor<256x256x14x14xf32>
    %v3913 = stablehlo.reduce(%v3912 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3914 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3915 = stablehlo.divide %v3913, %v3914 : tensor<256xf32>
    %v3916 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3918 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3919 = stablehlo.reduce(%v3916 init: %v3917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3920 = stablehlo.divide %v3919, %v3918 : tensor<256xf32>
    %v3921 = stablehlo.reshape %v717 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3924 = stablehlo.reduce(%v3921 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3925 = stablehlo.broadcast_in_dim %v3924, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3926 = stablehlo.divide %v3925, %v3923 : tensor<256x256x14x14xf32>
    %v3927 = stablehlo.subtract %v3921, %v3926 : tensor<256x256x14x14xf32>
    %v3928 = stablehlo.multiply %v3927, %v3927 : tensor<256x256x14x14xf32>
    %v3929 = stablehlo.reduce(%v3928 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3930 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3931 = stablehlo.divide %v3929, %v3930 : tensor<256xf32>
    %v3932 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3935 = stablehlo.reduce(%v3932 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3936 = stablehlo.divide %v3935, %v3934 : tensor<256xf32>
    %v3937 = stablehlo.reshape %v745 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3942 = stablehlo.divide %v3941, %v3939 : tensor<256x256x14x14xf32>
    %v3943 = stablehlo.subtract %v3937, %v3942 : tensor<256x256x14x14xf32>
    %v3944 = stablehlo.multiply %v3943, %v3943 : tensor<256x256x14x14xf32>
    %v3945 = stablehlo.reduce(%v3944 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3946 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3947 = stablehlo.divide %v3945, %v3946 : tensor<256xf32>
    %v3948 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3950 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3951 = stablehlo.reduce(%v3948 init: %v3949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3952 = stablehlo.divide %v3951, %v3950 : tensor<256xf32>
    %v3953 = stablehlo.reshape %v772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3957 = stablehlo.broadcast_in_dim %v3956, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3958 = stablehlo.divide %v3957, %v3955 : tensor<256x256x14x14xf32>
    %v3959 = stablehlo.subtract %v3953, %v3958 : tensor<256x256x14x14xf32>
    %v3960 = stablehlo.multiply %v3959, %v3959 : tensor<256x256x14x14xf32>
    %v3961 = stablehlo.reduce(%v3960 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3962 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v3963 = stablehlo.divide %v3961, %v3962 : tensor<256xf32>
    %v3964 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3966 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3967 = stablehlo.reduce(%v3964 init: %v3965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3968 = stablehlo.divide %v3967, %v3966 : tensor<512xf32>
    %v3969 = stablehlo.reshape %v800 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3973 = stablehlo.broadcast_in_dim %v3972, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3974 = stablehlo.divide %v3973, %v3971 : tensor<256x512x7x7xf32>
    %v3975 = stablehlo.subtract %v3969, %v3974 : tensor<256x512x7x7xf32>
    %v3976 = stablehlo.multiply %v3975, %v3975 : tensor<256x512x7x7xf32>
    %v3977 = stablehlo.reduce(%v3976 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3978 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3979 = stablehlo.divide %v3977, %v3978 : tensor<512xf32>
    %v3980 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3982 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3983 = stablehlo.reduce(%v3980 init: %v3981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3984 = stablehlo.divide %v3983, %v3982 : tensor<512xf32>
    %v3985 = stablehlo.reshape %v827 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v3988 = stablehlo.reduce(%v3985 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3989 = stablehlo.broadcast_in_dim %v3988, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v3990 = stablehlo.divide %v3989, %v3987 : tensor<256x512x7x7xf32>
    %v3991 = stablehlo.subtract %v3985, %v3990 : tensor<256x512x7x7xf32>
    %v3992 = stablehlo.multiply %v3991, %v3991 : tensor<256x512x7x7xf32>
    %v3993 = stablehlo.reduce(%v3992 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3994 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3995 = stablehlo.divide %v3993, %v3994 : tensor<512xf32>
    %v3996 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v3997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3998 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3999 = stablehlo.reduce(%v3996 init: %v3997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4000 = stablehlo.divide %v3999, %v3998 : tensor<512xf32>
    %v4001 = stablehlo.reshape %v852 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v4004 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4005 = stablehlo.broadcast_in_dim %v4004, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v4006 = stablehlo.divide %v4005, %v4003 : tensor<256x512x7x7xf32>
    %v4007 = stablehlo.subtract %v4001, %v4006 : tensor<256x512x7x7xf32>
    %v4008 = stablehlo.multiply %v4007, %v4007 : tensor<256x512x7x7xf32>
    %v4009 = stablehlo.reduce(%v4008 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4010 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4011 = stablehlo.divide %v4009, %v4010 : tensor<512xf32>
    %v4012 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4015 = stablehlo.reduce(%v4012 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4016 = stablehlo.divide %v4015, %v4014 : tensor<512xf32>
    %v4017 = stablehlo.reshape %v880 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4021 = stablehlo.broadcast_in_dim %v4020, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v4022 = stablehlo.divide %v4021, %v4019 : tensor<256x512x7x7xf32>
    %v4023 = stablehlo.subtract %v4017, %v4022 : tensor<256x512x7x7xf32>
    %v4024 = stablehlo.multiply %v4023, %v4023 : tensor<256x512x7x7xf32>
    %v4025 = stablehlo.reduce(%v4024 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4026 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4027 = stablehlo.divide %v4025, %v4026 : tensor<512xf32>
    %v4028 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4030 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4031 = stablehlo.reduce(%v4028 init: %v4029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4032 = stablehlo.divide %v4031, %v4030 : tensor<512xf32>
    %v4033 = stablehlo.reshape %v907 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4035 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v4036 = stablehlo.reduce(%v4033 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4037 = stablehlo.broadcast_in_dim %v4036, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v4038 = stablehlo.divide %v4037, %v4035 : tensor<256x512x7x7xf32>
    %v4039 = stablehlo.subtract %v4033, %v4038 : tensor<256x512x7x7xf32>
    %v4040 = stablehlo.multiply %v4039, %v4039 : tensor<256x512x7x7xf32>
    %v4041 = stablehlo.reduce(%v4040 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4042 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4043 = stablehlo.divide %v4041, %v4042 : tensor<512xf32>
    %v4044 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4046 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4047 = stablehlo.reduce(%v4044 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4048 = stablehlo.divide %v4047, %v4046 : tensor<512xf32>
    %v4049 = stablehlo.reshape %v935 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v4052 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4052, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v4054 = stablehlo.divide %v4053, %v4051 : tensor<256x512x7x7xf32>
    %v4055 = stablehlo.subtract %v4049, %v4054 : tensor<256x512x7x7xf32>
    %v4056 = stablehlo.multiply %v4055, %v4055 : tensor<256x512x7x7xf32>
    %v4057 = stablehlo.reduce(%v4056 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4058 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4059 = stablehlo.divide %v4057, %v4058 : tensor<512xf32>
    %v4060 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4062 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4063 = stablehlo.reduce(%v4060 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4064 = stablehlo.divide %v4063, %v4062 : tensor<512xf32>
    %v4065 = stablehlo.reshape %v962 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v4068 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4069 = stablehlo.broadcast_in_dim %v4068, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v4070 = stablehlo.divide %v4069, %v4067 : tensor<256x512x7x7xf32>
    %v4071 = stablehlo.subtract %v4065, %v4070 : tensor<256x512x7x7xf32>
    %v4072 = stablehlo.multiply %v4071, %v4071 : tensor<256x512x7x7xf32>
    %v4073 = stablehlo.reduce(%v4072 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4074 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v4075 = stablehlo.divide %v4073, %v4074 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v4076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4077 = stablehlo.multiply %v4076, %sW : tensor<64x3x7x7xf32>
    %v4078 = stablehlo.add %v4077, %v3475 : tensor<64x3x7x7xf32>
    %v4079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4080 = stablehlo.multiply %v4079, %sWv : tensor<64x3x7x7xf32>
    %v4081 = stablehlo.add %v4080, %v4078 : tensor<64x3x7x7xf32>
    %v4082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4083 = stablehlo.multiply %v4082, %v4081 : tensor<64x3x7x7xf32>
    %v4084 = stablehlo.subtract %sW, %v4083 : tensor<64x3x7x7xf32>
    %v4085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4086 = stablehlo.multiply %v4085, %sbi : tensor<64xf32>
    %v4087 = stablehlo.add %v4086, %v3478 : tensor<64xf32>
    %v4088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4089 = stablehlo.multiply %v4088, %sbiv : tensor<64xf32>
    %v4090 = stablehlo.add %v4089, %v4087 : tensor<64xf32>
    %v4091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4092 = stablehlo.multiply %v4091, %v4090 : tensor<64xf32>
    %v4093 = stablehlo.subtract %sbi, %v4092 : tensor<64xf32>
    %v4094 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4095 = stablehlo.multiply %v4094, %sg : tensor<64xf32>
    %v4096 = stablehlo.add %v4095, %v3496 : tensor<64xf32>
    %v4097 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4098 = stablehlo.multiply %v4097, %sgv : tensor<64xf32>
    %v4099 = stablehlo.add %v4098, %v4096 : tensor<64xf32>
    %v4100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4101 = stablehlo.multiply %v4100, %v4099 : tensor<64xf32>
    %v4102 = stablehlo.subtract %sg, %v4101 : tensor<64xf32>
    %v4103 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4104 = stablehlo.multiply %v4103, %sbt : tensor<64xf32>
    %v4105 = stablehlo.add %v4104, %v3499 : tensor<64xf32>
    %v4106 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4107 = stablehlo.multiply %v4106, %sbtv : tensor<64xf32>
    %v4108 = stablehlo.add %v4107, %v4105 : tensor<64xf32>
    %v4109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4110 = stablehlo.multiply %v4109, %v4108 : tensor<64xf32>
    %v4111 = stablehlo.subtract %sbt, %v4110 : tensor<64xf32>
    %v4112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4113 = stablehlo.multiply %v4112, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4114 = stablehlo.add %v4113, %v3375 : tensor<64x64x3x3xf32>
    %v4115 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4116 = stablehlo.multiply %v4115, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4117 = stablehlo.add %v4116, %v4114 : tensor<64x64x3x3xf32>
    %v4118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4119 = stablehlo.multiply %v4118, %v4117 : tensor<64x64x3x3xf32>
    %v4120 = stablehlo.subtract %s1b0W1, %v4119 : tensor<64x64x3x3xf32>
    %v4121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4122 = stablehlo.multiply %v4121, %s1b0b1 : tensor<64xf32>
    %v4123 = stablehlo.add %v4122, %v3378 : tensor<64xf32>
    %v4124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4125 = stablehlo.multiply %v4124, %s1b0b1v : tensor<64xf32>
    %v4126 = stablehlo.add %v4125, %v4123 : tensor<64xf32>
    %v4127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4128 = stablehlo.multiply %v4127, %v4126 : tensor<64xf32>
    %v4129 = stablehlo.subtract %s1b0b1, %v4128 : tensor<64xf32>
    %v4130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4131 = stablehlo.multiply %v4130, %s1b0g1 : tensor<64xf32>
    %v4132 = stablehlo.add %v4131, %v3396 : tensor<64xf32>
    %v4133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4134 = stablehlo.multiply %v4133, %s1b0g1v : tensor<64xf32>
    %v4135 = stablehlo.add %v4134, %v4132 : tensor<64xf32>
    %v4136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4137 = stablehlo.multiply %v4136, %v4135 : tensor<64xf32>
    %v4138 = stablehlo.subtract %s1b0g1, %v4137 : tensor<64xf32>
    %v4139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4140 = stablehlo.multiply %v4139, %s1b0bt1 : tensor<64xf32>
    %v4141 = stablehlo.add %v4140, %v3399 : tensor<64xf32>
    %v4142 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4143 = stablehlo.multiply %v4142, %s1b0bt1v : tensor<64xf32>
    %v4144 = stablehlo.add %v4143, %v4141 : tensor<64xf32>
    %v4145 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4146 = stablehlo.multiply %v4145, %v4144 : tensor<64xf32>
    %v4147 = stablehlo.subtract %s1b0bt1, %v4146 : tensor<64xf32>
    %v4148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4149 = stablehlo.multiply %v4148, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4150 = stablehlo.add %v4149, %v3405 : tensor<64x64x3x3xf32>
    %v4151 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4152 = stablehlo.multiply %v4151, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4153 = stablehlo.add %v4152, %v4150 : tensor<64x64x3x3xf32>
    %v4154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4155 = stablehlo.multiply %v4154, %v4153 : tensor<64x64x3x3xf32>
    %v4156 = stablehlo.subtract %s1b0W2, %v4155 : tensor<64x64x3x3xf32>
    %v4157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4158 = stablehlo.multiply %v4157, %s1b0b2 : tensor<64xf32>
    %v4159 = stablehlo.add %v4158, %v3408 : tensor<64xf32>
    %v4160 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4161 = stablehlo.multiply %v4160, %s1b0b2v : tensor<64xf32>
    %v4162 = stablehlo.add %v4161, %v4159 : tensor<64xf32>
    %v4163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4164 = stablehlo.multiply %v4163, %v4162 : tensor<64xf32>
    %v4165 = stablehlo.subtract %s1b0b2, %v4164 : tensor<64xf32>
    %v4166 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4167 = stablehlo.multiply %v4166, %s1b0g2 : tensor<64xf32>
    %v4168 = stablehlo.add %v4167, %v3426 : tensor<64xf32>
    %v4169 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4170 = stablehlo.multiply %v4169, %s1b0g2v : tensor<64xf32>
    %v4171 = stablehlo.add %v4170, %v4168 : tensor<64xf32>
    %v4172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4173 = stablehlo.multiply %v4172, %v4171 : tensor<64xf32>
    %v4174 = stablehlo.subtract %s1b0g2, %v4173 : tensor<64xf32>
    %v4175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4176 = stablehlo.multiply %v4175, %s1b0bt2 : tensor<64xf32>
    %v4177 = stablehlo.add %v4176, %v3429 : tensor<64xf32>
    %v4178 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4179 = stablehlo.multiply %v4178, %s1b0bt2v : tensor<64xf32>
    %v4180 = stablehlo.add %v4179, %v4177 : tensor<64xf32>
    %v4181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4182 = stablehlo.multiply %v4181, %v4180 : tensor<64xf32>
    %v4183 = stablehlo.subtract %s1b0bt2, %v4182 : tensor<64xf32>
    %v4184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4185 = stablehlo.multiply %v4184, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4186 = stablehlo.add %v4185, %v3238 : tensor<64x64x3x3xf32>
    %v4187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4188 = stablehlo.multiply %v4187, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4189 = stablehlo.add %v4188, %v4186 : tensor<64x64x3x3xf32>
    %v4190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4191 = stablehlo.multiply %v4190, %v4189 : tensor<64x64x3x3xf32>
    %v4192 = stablehlo.subtract %s1b1W1, %v4191 : tensor<64x64x3x3xf32>
    %v4193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4194 = stablehlo.multiply %v4193, %s1b1b1 : tensor<64xf32>
    %v4195 = stablehlo.add %v4194, %v3241 : tensor<64xf32>
    %v4196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4197 = stablehlo.multiply %v4196, %s1b1b1v : tensor<64xf32>
    %v4198 = stablehlo.add %v4197, %v4195 : tensor<64xf32>
    %v4199 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4200 = stablehlo.multiply %v4199, %v4198 : tensor<64xf32>
    %v4201 = stablehlo.subtract %s1b1b1, %v4200 : tensor<64xf32>
    %v4202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4203 = stablehlo.multiply %v4202, %s1b1g1 : tensor<64xf32>
    %v4204 = stablehlo.add %v4203, %v3259 : tensor<64xf32>
    %v4205 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4206 = stablehlo.multiply %v4205, %s1b1g1v : tensor<64xf32>
    %v4207 = stablehlo.add %v4206, %v4204 : tensor<64xf32>
    %v4208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4209 = stablehlo.multiply %v4208, %v4207 : tensor<64xf32>
    %v4210 = stablehlo.subtract %s1b1g1, %v4209 : tensor<64xf32>
    %v4211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4212 = stablehlo.multiply %v4211, %s1b1bt1 : tensor<64xf32>
    %v4213 = stablehlo.add %v4212, %v3262 : tensor<64xf32>
    %v4214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4215 = stablehlo.multiply %v4214, %s1b1bt1v : tensor<64xf32>
    %v4216 = stablehlo.add %v4215, %v4213 : tensor<64xf32>
    %v4217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4218 = stablehlo.multiply %v4217, %v4216 : tensor<64xf32>
    %v4219 = stablehlo.subtract %s1b1bt1, %v4218 : tensor<64xf32>
    %v4220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4221 = stablehlo.multiply %v4220, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4222 = stablehlo.add %v4221, %v3268 : tensor<64x64x3x3xf32>
    %v4223 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4224 = stablehlo.multiply %v4223, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4225 = stablehlo.add %v4224, %v4222 : tensor<64x64x3x3xf32>
    %v4226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4227 = stablehlo.multiply %v4226, %v4225 : tensor<64x64x3x3xf32>
    %v4228 = stablehlo.subtract %s1b1W2, %v4227 : tensor<64x64x3x3xf32>
    %v4229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4230 = stablehlo.multiply %v4229, %s1b1b2 : tensor<64xf32>
    %v4231 = stablehlo.add %v4230, %v3271 : tensor<64xf32>
    %v4232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4233 = stablehlo.multiply %v4232, %s1b1b2v : tensor<64xf32>
    %v4234 = stablehlo.add %v4233, %v4231 : tensor<64xf32>
    %v4235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4236 = stablehlo.multiply %v4235, %v4234 : tensor<64xf32>
    %v4237 = stablehlo.subtract %s1b1b2, %v4236 : tensor<64xf32>
    %v4238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4239 = stablehlo.multiply %v4238, %s1b1g2 : tensor<64xf32>
    %v4240 = stablehlo.add %v4239, %v3289 : tensor<64xf32>
    %v4241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4242 = stablehlo.multiply %v4241, %s1b1g2v : tensor<64xf32>
    %v4243 = stablehlo.add %v4242, %v4240 : tensor<64xf32>
    %v4244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4245 = stablehlo.multiply %v4244, %v4243 : tensor<64xf32>
    %v4246 = stablehlo.subtract %s1b1g2, %v4245 : tensor<64xf32>
    %v4247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4248 = stablehlo.multiply %v4247, %s1b1bt2 : tensor<64xf32>
    %v4249 = stablehlo.add %v4248, %v3292 : tensor<64xf32>
    %v4250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4251 = stablehlo.multiply %v4250, %s1b1bt2v : tensor<64xf32>
    %v4252 = stablehlo.add %v4251, %v4249 : tensor<64xf32>
    %v4253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4254 = stablehlo.multiply %v4253, %v4252 : tensor<64xf32>
    %v4255 = stablehlo.subtract %s1b1bt2, %v4254 : tensor<64xf32>
    %v4256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4257 = stablehlo.multiply %v4256, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4258 = stablehlo.add %v4257, %v3101 : tensor<64x64x3x3xf32>
    %v4259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4260 = stablehlo.multiply %v4259, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4261 = stablehlo.add %v4260, %v4258 : tensor<64x64x3x3xf32>
    %v4262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4263 = stablehlo.multiply %v4262, %v4261 : tensor<64x64x3x3xf32>
    %v4264 = stablehlo.subtract %s1b2W1, %v4263 : tensor<64x64x3x3xf32>
    %v4265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4266 = stablehlo.multiply %v4265, %s1b2b1 : tensor<64xf32>
    %v4267 = stablehlo.add %v4266, %v3104 : tensor<64xf32>
    %v4268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4269 = stablehlo.multiply %v4268, %s1b2b1v : tensor<64xf32>
    %v4270 = stablehlo.add %v4269, %v4267 : tensor<64xf32>
    %v4271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4272 = stablehlo.multiply %v4271, %v4270 : tensor<64xf32>
    %v4273 = stablehlo.subtract %s1b2b1, %v4272 : tensor<64xf32>
    %v4274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4275 = stablehlo.multiply %v4274, %s1b2g1 : tensor<64xf32>
    %v4276 = stablehlo.add %v4275, %v3122 : tensor<64xf32>
    %v4277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4278 = stablehlo.multiply %v4277, %s1b2g1v : tensor<64xf32>
    %v4279 = stablehlo.add %v4278, %v4276 : tensor<64xf32>
    %v4280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4281 = stablehlo.multiply %v4280, %v4279 : tensor<64xf32>
    %v4282 = stablehlo.subtract %s1b2g1, %v4281 : tensor<64xf32>
    %v4283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4284 = stablehlo.multiply %v4283, %s1b2bt1 : tensor<64xf32>
    %v4285 = stablehlo.add %v4284, %v3125 : tensor<64xf32>
    %v4286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4287 = stablehlo.multiply %v4286, %s1b2bt1v : tensor<64xf32>
    %v4288 = stablehlo.add %v4287, %v4285 : tensor<64xf32>
    %v4289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4290 = stablehlo.multiply %v4289, %v4288 : tensor<64xf32>
    %v4291 = stablehlo.subtract %s1b2bt1, %v4290 : tensor<64xf32>
    %v4292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4293 = stablehlo.multiply %v4292, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4294 = stablehlo.add %v4293, %v3131 : tensor<64x64x3x3xf32>
    %v4295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4296 = stablehlo.multiply %v4295, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4297 = stablehlo.add %v4296, %v4294 : tensor<64x64x3x3xf32>
    %v4298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4299 = stablehlo.multiply %v4298, %v4297 : tensor<64x64x3x3xf32>
    %v4300 = stablehlo.subtract %s1b2W2, %v4299 : tensor<64x64x3x3xf32>
    %v4301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4302 = stablehlo.multiply %v4301, %s1b2b2 : tensor<64xf32>
    %v4303 = stablehlo.add %v4302, %v3134 : tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4304, %s1b2b2v : tensor<64xf32>
    %v4306 = stablehlo.add %v4305, %v4303 : tensor<64xf32>
    %v4307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4308 = stablehlo.multiply %v4307, %v4306 : tensor<64xf32>
    %v4309 = stablehlo.subtract %s1b2b2, %v4308 : tensor<64xf32>
    %v4310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4311 = stablehlo.multiply %v4310, %s1b2g2 : tensor<64xf32>
    %v4312 = stablehlo.add %v4311, %v3152 : tensor<64xf32>
    %v4313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4314 = stablehlo.multiply %v4313, %s1b2g2v : tensor<64xf32>
    %v4315 = stablehlo.add %v4314, %v4312 : tensor<64xf32>
    %v4316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4317 = stablehlo.multiply %v4316, %v4315 : tensor<64xf32>
    %v4318 = stablehlo.subtract %s1b2g2, %v4317 : tensor<64xf32>
    %v4319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4320 = stablehlo.multiply %v4319, %s1b2bt2 : tensor<64xf32>
    %v4321 = stablehlo.add %v4320, %v3155 : tensor<64xf32>
    %v4322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4323 = stablehlo.multiply %v4322, %s1b2bt2v : tensor<64xf32>
    %v4324 = stablehlo.add %v4323, %v4321 : tensor<64xf32>
    %v4325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.multiply %v4325, %v4324 : tensor<64xf32>
    %v4327 = stablehlo.subtract %s1b2bt2, %v4326 : tensor<64xf32>
    %v4328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4329 = stablehlo.multiply %v4328, %d2W1 : tensor<128x64x3x3xf32>
    %v4330 = stablehlo.add %v4329, %v2932 : tensor<128x64x3x3xf32>
    %v4331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4332 = stablehlo.multiply %v4331, %d2W1v : tensor<128x64x3x3xf32>
    %v4333 = stablehlo.add %v4332, %v4330 : tensor<128x64x3x3xf32>
    %v4334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4335 = stablehlo.multiply %v4334, %v4333 : tensor<128x64x3x3xf32>
    %v4336 = stablehlo.subtract %d2W1, %v4335 : tensor<128x64x3x3xf32>
    %v4337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4338 = stablehlo.multiply %v4337, %d2b1 : tensor<128xf32>
    %v4339 = stablehlo.add %v4338, %v2935 : tensor<128xf32>
    %v4340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4341 = stablehlo.multiply %v4340, %d2b1v : tensor<128xf32>
    %v4342 = stablehlo.add %v4341, %v4339 : tensor<128xf32>
    %v4343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4344 = stablehlo.multiply %v4343, %v4342 : tensor<128xf32>
    %v4345 = stablehlo.subtract %d2b1, %v4344 : tensor<128xf32>
    %v4346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4347 = stablehlo.multiply %v4346, %d2g1 : tensor<128xf32>
    %v4348 = stablehlo.add %v4347, %v2953 : tensor<128xf32>
    %v4349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4350 = stablehlo.multiply %v4349, %d2g1v : tensor<128xf32>
    %v4351 = stablehlo.add %v4350, %v4348 : tensor<128xf32>
    %v4352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4353 = stablehlo.multiply %v4352, %v4351 : tensor<128xf32>
    %v4354 = stablehlo.subtract %d2g1, %v4353 : tensor<128xf32>
    %v4355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4356 = stablehlo.multiply %v4355, %d2bt1 : tensor<128xf32>
    %v4357 = stablehlo.add %v4356, %v2956 : tensor<128xf32>
    %v4358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4359 = stablehlo.multiply %v4358, %d2bt1v : tensor<128xf32>
    %v4360 = stablehlo.add %v4359, %v4357 : tensor<128xf32>
    %v4361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4362 = stablehlo.multiply %v4361, %v4360 : tensor<128xf32>
    %v4363 = stablehlo.subtract %d2bt1, %v4362 : tensor<128xf32>
    %v4364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4365 = stablehlo.multiply %v4364, %d2W2 : tensor<128x128x3x3xf32>
    %v4366 = stablehlo.add %v4365, %v2962 : tensor<128x128x3x3xf32>
    %v4367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4368 = stablehlo.multiply %v4367, %d2W2v : tensor<128x128x3x3xf32>
    %v4369 = stablehlo.add %v4368, %v4366 : tensor<128x128x3x3xf32>
    %v4370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4371 = stablehlo.multiply %v4370, %v4369 : tensor<128x128x3x3xf32>
    %v4372 = stablehlo.subtract %d2W2, %v4371 : tensor<128x128x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4374 = stablehlo.multiply %v4373, %d2b2 : tensor<128xf32>
    %v4375 = stablehlo.add %v4374, %v2965 : tensor<128xf32>
    %v4376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4377 = stablehlo.multiply %v4376, %d2b2v : tensor<128xf32>
    %v4378 = stablehlo.add %v4377, %v4375 : tensor<128xf32>
    %v4379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4380 = stablehlo.multiply %v4379, %v4378 : tensor<128xf32>
    %v4381 = stablehlo.subtract %d2b2, %v4380 : tensor<128xf32>
    %v4382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4383 = stablehlo.multiply %v4382, %d2g2 : tensor<128xf32>
    %v4384 = stablehlo.add %v4383, %v2983 : tensor<128xf32>
    %v4385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4386 = stablehlo.multiply %v4385, %d2g2v : tensor<128xf32>
    %v4387 = stablehlo.add %v4386, %v4384 : tensor<128xf32>
    %v4388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4389 = stablehlo.multiply %v4388, %v4387 : tensor<128xf32>
    %v4390 = stablehlo.subtract %d2g2, %v4389 : tensor<128xf32>
    %v4391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4392 = stablehlo.multiply %v4391, %d2bt2 : tensor<128xf32>
    %v4393 = stablehlo.add %v4392, %v2986 : tensor<128xf32>
    %v4394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4395 = stablehlo.multiply %v4394, %d2bt2v : tensor<128xf32>
    %v4396 = stablehlo.add %v4395, %v4393 : tensor<128xf32>
    %v4397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4398 = stablehlo.multiply %v4397, %v4396 : tensor<128xf32>
    %v4399 = stablehlo.subtract %d2bt2, %v4398 : tensor<128xf32>
    %v4400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4401 = stablehlo.multiply %v4400, %d2Wp : tensor<128x64x3x3xf32>
    %v4402 = stablehlo.add %v4401, %v2994 : tensor<128x64x3x3xf32>
    %v4403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4404 = stablehlo.multiply %v4403, %d2Wpv : tensor<128x64x3x3xf32>
    %v4405 = stablehlo.add %v4404, %v4402 : tensor<128x64x3x3xf32>
    %v4406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4407 = stablehlo.multiply %v4406, %v4405 : tensor<128x64x3x3xf32>
    %v4408 = stablehlo.subtract %d2Wp, %v4407 : tensor<128x64x3x3xf32>
    %v4409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4410 = stablehlo.multiply %v4409, %d2bp : tensor<128xf32>
    %v4411 = stablehlo.add %v4410, %v2997 : tensor<128xf32>
    %v4412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4413 = stablehlo.multiply %v4412, %d2bpv : tensor<128xf32>
    %v4414 = stablehlo.add %v4413, %v4411 : tensor<128xf32>
    %v4415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4416 = stablehlo.multiply %v4415, %v4414 : tensor<128xf32>
    %v4417 = stablehlo.subtract %d2bp, %v4416 : tensor<128xf32>
    %v4418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4419 = stablehlo.multiply %v4418, %d2gp : tensor<128xf32>
    %v4420 = stablehlo.add %v4419, %v3015 : tensor<128xf32>
    %v4421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4422 = stablehlo.multiply %v4421, %d2gpv : tensor<128xf32>
    %v4423 = stablehlo.add %v4422, %v4420 : tensor<128xf32>
    %v4424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4425 = stablehlo.multiply %v4424, %v4423 : tensor<128xf32>
    %v4426 = stablehlo.subtract %d2gp, %v4425 : tensor<128xf32>
    %v4427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4428 = stablehlo.multiply %v4427, %d2btp : tensor<128xf32>
    %v4429 = stablehlo.add %v4428, %v3018 : tensor<128xf32>
    %v4430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4431 = stablehlo.multiply %v4430, %d2btpv : tensor<128xf32>
    %v4432 = stablehlo.add %v4431, %v4429 : tensor<128xf32>
    %v4433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4434 = stablehlo.multiply %v4433, %v4432 : tensor<128xf32>
    %v4435 = stablehlo.subtract %d2btp, %v4434 : tensor<128xf32>
    %v4436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4437 = stablehlo.multiply %v4436, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4438 = stablehlo.add %v4437, %v2754 : tensor<128x128x3x3xf32>
    %v4439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4440 = stablehlo.multiply %v4439, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4441 = stablehlo.add %v4440, %v4438 : tensor<128x128x3x3xf32>
    %v4442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4443 = stablehlo.multiply %v4442, %v4441 : tensor<128x128x3x3xf32>
    %v4444 = stablehlo.subtract %s2b0W1, %v4443 : tensor<128x128x3x3xf32>
    %v4445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4446 = stablehlo.multiply %v4445, %s2b0b1 : tensor<128xf32>
    %v4447 = stablehlo.add %v4446, %v2757 : tensor<128xf32>
    %v4448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4449 = stablehlo.multiply %v4448, %s2b0b1v : tensor<128xf32>
    %v4450 = stablehlo.add %v4449, %v4447 : tensor<128xf32>
    %v4451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4452 = stablehlo.multiply %v4451, %v4450 : tensor<128xf32>
    %v4453 = stablehlo.subtract %s2b0b1, %v4452 : tensor<128xf32>
    %v4454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4455 = stablehlo.multiply %v4454, %s2b0g1 : tensor<128xf32>
    %v4456 = stablehlo.add %v4455, %v2775 : tensor<128xf32>
    %v4457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4458 = stablehlo.multiply %v4457, %s2b0g1v : tensor<128xf32>
    %v4459 = stablehlo.add %v4458, %v4456 : tensor<128xf32>
    %v4460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4461 = stablehlo.multiply %v4460, %v4459 : tensor<128xf32>
    %v4462 = stablehlo.subtract %s2b0g1, %v4461 : tensor<128xf32>
    %v4463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4464 = stablehlo.multiply %v4463, %s2b0bt1 : tensor<128xf32>
    %v4465 = stablehlo.add %v4464, %v2778 : tensor<128xf32>
    %v4466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4467 = stablehlo.multiply %v4466, %s2b0bt1v : tensor<128xf32>
    %v4468 = stablehlo.add %v4467, %v4465 : tensor<128xf32>
    %v4469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4470 = stablehlo.multiply %v4469, %v4468 : tensor<128xf32>
    %v4471 = stablehlo.subtract %s2b0bt1, %v4470 : tensor<128xf32>
    %v4472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4473 = stablehlo.multiply %v4472, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4474 = stablehlo.add %v4473, %v2784 : tensor<128x128x3x3xf32>
    %v4475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4476 = stablehlo.multiply %v4475, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4477 = stablehlo.add %v4476, %v4474 : tensor<128x128x3x3xf32>
    %v4478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4479 = stablehlo.multiply %v4478, %v4477 : tensor<128x128x3x3xf32>
    %v4480 = stablehlo.subtract %s2b0W2, %v4479 : tensor<128x128x3x3xf32>
    %v4481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4482 = stablehlo.multiply %v4481, %s2b0b2 : tensor<128xf32>
    %v4483 = stablehlo.add %v4482, %v2787 : tensor<128xf32>
    %v4484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4485 = stablehlo.multiply %v4484, %s2b0b2v : tensor<128xf32>
    %v4486 = stablehlo.add %v4485, %v4483 : tensor<128xf32>
    %v4487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4488 = stablehlo.multiply %v4487, %v4486 : tensor<128xf32>
    %v4489 = stablehlo.subtract %s2b0b2, %v4488 : tensor<128xf32>
    %v4490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4491 = stablehlo.multiply %v4490, %s2b0g2 : tensor<128xf32>
    %v4492 = stablehlo.add %v4491, %v2805 : tensor<128xf32>
    %v4493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4494 = stablehlo.multiply %v4493, %s2b0g2v : tensor<128xf32>
    %v4495 = stablehlo.add %v4494, %v4492 : tensor<128xf32>
    %v4496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4497 = stablehlo.multiply %v4496, %v4495 : tensor<128xf32>
    %v4498 = stablehlo.subtract %s2b0g2, %v4497 : tensor<128xf32>
    %v4499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4500 = stablehlo.multiply %v4499, %s2b0bt2 : tensor<128xf32>
    %v4501 = stablehlo.add %v4500, %v2808 : tensor<128xf32>
    %v4502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4503 = stablehlo.multiply %v4502, %s2b0bt2v : tensor<128xf32>
    %v4504 = stablehlo.add %v4503, %v4501 : tensor<128xf32>
    %v4505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4506 = stablehlo.multiply %v4505, %v4504 : tensor<128xf32>
    %v4507 = stablehlo.subtract %s2b0bt2, %v4506 : tensor<128xf32>
    %v4508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4509 = stablehlo.multiply %v4508, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4510 = stablehlo.add %v4509, %v2617 : tensor<128x128x3x3xf32>
    %v4511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4512 = stablehlo.multiply %v4511, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4513 = stablehlo.add %v4512, %v4510 : tensor<128x128x3x3xf32>
    %v4514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4515 = stablehlo.multiply %v4514, %v4513 : tensor<128x128x3x3xf32>
    %v4516 = stablehlo.subtract %s2b1W1, %v4515 : tensor<128x128x3x3xf32>
    %v4517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4518 = stablehlo.multiply %v4517, %s2b1b1 : tensor<128xf32>
    %v4519 = stablehlo.add %v4518, %v2620 : tensor<128xf32>
    %v4520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4521 = stablehlo.multiply %v4520, %s2b1b1v : tensor<128xf32>
    %v4522 = stablehlo.add %v4521, %v4519 : tensor<128xf32>
    %v4523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4524 = stablehlo.multiply %v4523, %v4522 : tensor<128xf32>
    %v4525 = stablehlo.subtract %s2b1b1, %v4524 : tensor<128xf32>
    %v4526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4527 = stablehlo.multiply %v4526, %s2b1g1 : tensor<128xf32>
    %v4528 = stablehlo.add %v4527, %v2638 : tensor<128xf32>
    %v4529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4530 = stablehlo.multiply %v4529, %s2b1g1v : tensor<128xf32>
    %v4531 = stablehlo.add %v4530, %v4528 : tensor<128xf32>
    %v4532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4533 = stablehlo.multiply %v4532, %v4531 : tensor<128xf32>
    %v4534 = stablehlo.subtract %s2b1g1, %v4533 : tensor<128xf32>
    %v4535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4536 = stablehlo.multiply %v4535, %s2b1bt1 : tensor<128xf32>
    %v4537 = stablehlo.add %v4536, %v2641 : tensor<128xf32>
    %v4538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4539 = stablehlo.multiply %v4538, %s2b1bt1v : tensor<128xf32>
    %v4540 = stablehlo.add %v4539, %v4537 : tensor<128xf32>
    %v4541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4542 = stablehlo.multiply %v4541, %v4540 : tensor<128xf32>
    %v4543 = stablehlo.subtract %s2b1bt1, %v4542 : tensor<128xf32>
    %v4544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4545 = stablehlo.multiply %v4544, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4546 = stablehlo.add %v4545, %v2647 : tensor<128x128x3x3xf32>
    %v4547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4548 = stablehlo.multiply %v4547, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4549 = stablehlo.add %v4548, %v4546 : tensor<128x128x3x3xf32>
    %v4550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4551 = stablehlo.multiply %v4550, %v4549 : tensor<128x128x3x3xf32>
    %v4552 = stablehlo.subtract %s2b1W2, %v4551 : tensor<128x128x3x3xf32>
    %v4553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4554 = stablehlo.multiply %v4553, %s2b1b2 : tensor<128xf32>
    %v4555 = stablehlo.add %v4554, %v2650 : tensor<128xf32>
    %v4556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4557 = stablehlo.multiply %v4556, %s2b1b2v : tensor<128xf32>
    %v4558 = stablehlo.add %v4557, %v4555 : tensor<128xf32>
    %v4559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4560 = stablehlo.multiply %v4559, %v4558 : tensor<128xf32>
    %v4561 = stablehlo.subtract %s2b1b2, %v4560 : tensor<128xf32>
    %v4562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4563 = stablehlo.multiply %v4562, %s2b1g2 : tensor<128xf32>
    %v4564 = stablehlo.add %v4563, %v2668 : tensor<128xf32>
    %v4565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4566 = stablehlo.multiply %v4565, %s2b1g2v : tensor<128xf32>
    %v4567 = stablehlo.add %v4566, %v4564 : tensor<128xf32>
    %v4568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4569 = stablehlo.multiply %v4568, %v4567 : tensor<128xf32>
    %v4570 = stablehlo.subtract %s2b1g2, %v4569 : tensor<128xf32>
    %v4571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4572 = stablehlo.multiply %v4571, %s2b1bt2 : tensor<128xf32>
    %v4573 = stablehlo.add %v4572, %v2671 : tensor<128xf32>
    %v4574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4575 = stablehlo.multiply %v4574, %s2b1bt2v : tensor<128xf32>
    %v4576 = stablehlo.add %v4575, %v4573 : tensor<128xf32>
    %v4577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4578 = stablehlo.multiply %v4577, %v4576 : tensor<128xf32>
    %v4579 = stablehlo.subtract %s2b1bt2, %v4578 : tensor<128xf32>
    %v4580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4581 = stablehlo.multiply %v4580, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4582 = stablehlo.add %v4581, %v2480 : tensor<128x128x3x3xf32>
    %v4583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4584 = stablehlo.multiply %v4583, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4585 = stablehlo.add %v4584, %v4582 : tensor<128x128x3x3xf32>
    %v4586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4587 = stablehlo.multiply %v4586, %v4585 : tensor<128x128x3x3xf32>
    %v4588 = stablehlo.subtract %s2b2W1, %v4587 : tensor<128x128x3x3xf32>
    %v4589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4590 = stablehlo.multiply %v4589, %s2b2b1 : tensor<128xf32>
    %v4591 = stablehlo.add %v4590, %v2483 : tensor<128xf32>
    %v4592 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4593 = stablehlo.multiply %v4592, %s2b2b1v : tensor<128xf32>
    %v4594 = stablehlo.add %v4593, %v4591 : tensor<128xf32>
    %v4595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4596 = stablehlo.multiply %v4595, %v4594 : tensor<128xf32>
    %v4597 = stablehlo.subtract %s2b2b1, %v4596 : tensor<128xf32>
    %v4598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4599 = stablehlo.multiply %v4598, %s2b2g1 : tensor<128xf32>
    %v4600 = stablehlo.add %v4599, %v2501 : tensor<128xf32>
    %v4601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4602 = stablehlo.multiply %v4601, %s2b2g1v : tensor<128xf32>
    %v4603 = stablehlo.add %v4602, %v4600 : tensor<128xf32>
    %v4604 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4605 = stablehlo.multiply %v4604, %v4603 : tensor<128xf32>
    %v4606 = stablehlo.subtract %s2b2g1, %v4605 : tensor<128xf32>
    %v4607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4608 = stablehlo.multiply %v4607, %s2b2bt1 : tensor<128xf32>
    %v4609 = stablehlo.add %v4608, %v2504 : tensor<128xf32>
    %v4610 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4611 = stablehlo.multiply %v4610, %s2b2bt1v : tensor<128xf32>
    %v4612 = stablehlo.add %v4611, %v4609 : tensor<128xf32>
    %v4613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4614 = stablehlo.multiply %v4613, %v4612 : tensor<128xf32>
    %v4615 = stablehlo.subtract %s2b2bt1, %v4614 : tensor<128xf32>
    %v4616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4617 = stablehlo.multiply %v4616, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4618 = stablehlo.add %v4617, %v2510 : tensor<128x128x3x3xf32>
    %v4619 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4620 = stablehlo.multiply %v4619, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4621 = stablehlo.add %v4620, %v4618 : tensor<128x128x3x3xf32>
    %v4622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4623 = stablehlo.multiply %v4622, %v4621 : tensor<128x128x3x3xf32>
    %v4624 = stablehlo.subtract %s2b2W2, %v4623 : tensor<128x128x3x3xf32>
    %v4625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4626 = stablehlo.multiply %v4625, %s2b2b2 : tensor<128xf32>
    %v4627 = stablehlo.add %v4626, %v2513 : tensor<128xf32>
    %v4628 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4629 = stablehlo.multiply %v4628, %s2b2b2v : tensor<128xf32>
    %v4630 = stablehlo.add %v4629, %v4627 : tensor<128xf32>
    %v4631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4632 = stablehlo.multiply %v4631, %v4630 : tensor<128xf32>
    %v4633 = stablehlo.subtract %s2b2b2, %v4632 : tensor<128xf32>
    %v4634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4635 = stablehlo.multiply %v4634, %s2b2g2 : tensor<128xf32>
    %v4636 = stablehlo.add %v4635, %v2531 : tensor<128xf32>
    %v4637 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4638 = stablehlo.multiply %v4637, %s2b2g2v : tensor<128xf32>
    %v4639 = stablehlo.add %v4638, %v4636 : tensor<128xf32>
    %v4640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4641 = stablehlo.multiply %v4640, %v4639 : tensor<128xf32>
    %v4642 = stablehlo.subtract %s2b2g2, %v4641 : tensor<128xf32>
    %v4643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4644 = stablehlo.multiply %v4643, %s2b2bt2 : tensor<128xf32>
    %v4645 = stablehlo.add %v4644, %v2534 : tensor<128xf32>
    %v4646 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4647 = stablehlo.multiply %v4646, %s2b2bt2v : tensor<128xf32>
    %v4648 = stablehlo.add %v4647, %v4645 : tensor<128xf32>
    %v4649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4650 = stablehlo.multiply %v4649, %v4648 : tensor<128xf32>
    %v4651 = stablehlo.subtract %s2b2bt2, %v4650 : tensor<128xf32>
    %v4652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4653 = stablehlo.multiply %v4652, %d3W1 : tensor<256x128x3x3xf32>
    %v4654 = stablehlo.add %v4653, %v2311 : tensor<256x128x3x3xf32>
    %v4655 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4656 = stablehlo.multiply %v4655, %d3W1v : tensor<256x128x3x3xf32>
    %v4657 = stablehlo.add %v4656, %v4654 : tensor<256x128x3x3xf32>
    %v4658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4659 = stablehlo.multiply %v4658, %v4657 : tensor<256x128x3x3xf32>
    %v4660 = stablehlo.subtract %d3W1, %v4659 : tensor<256x128x3x3xf32>
    %v4661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4662 = stablehlo.multiply %v4661, %d3b1 : tensor<256xf32>
    %v4663 = stablehlo.add %v4662, %v2314 : tensor<256xf32>
    %v4664 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4665 = stablehlo.multiply %v4664, %d3b1v : tensor<256xf32>
    %v4666 = stablehlo.add %v4665, %v4663 : tensor<256xf32>
    %v4667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4668 = stablehlo.multiply %v4667, %v4666 : tensor<256xf32>
    %v4669 = stablehlo.subtract %d3b1, %v4668 : tensor<256xf32>
    %v4670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4671 = stablehlo.multiply %v4670, %d3g1 : tensor<256xf32>
    %v4672 = stablehlo.add %v4671, %v2332 : tensor<256xf32>
    %v4673 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4674 = stablehlo.multiply %v4673, %d3g1v : tensor<256xf32>
    %v4675 = stablehlo.add %v4674, %v4672 : tensor<256xf32>
    %v4676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4677 = stablehlo.multiply %v4676, %v4675 : tensor<256xf32>
    %v4678 = stablehlo.subtract %d3g1, %v4677 : tensor<256xf32>
    %v4679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4680 = stablehlo.multiply %v4679, %d3bt1 : tensor<256xf32>
    %v4681 = stablehlo.add %v4680, %v2335 : tensor<256xf32>
    %v4682 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4683 = stablehlo.multiply %v4682, %d3bt1v : tensor<256xf32>
    %v4684 = stablehlo.add %v4683, %v4681 : tensor<256xf32>
    %v4685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4686 = stablehlo.multiply %v4685, %v4684 : tensor<256xf32>
    %v4687 = stablehlo.subtract %d3bt1, %v4686 : tensor<256xf32>
    %v4688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4689 = stablehlo.multiply %v4688, %d3W2 : tensor<256x256x3x3xf32>
    %v4690 = stablehlo.add %v4689, %v2341 : tensor<256x256x3x3xf32>
    %v4691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4692 = stablehlo.multiply %v4691, %d3W2v : tensor<256x256x3x3xf32>
    %v4693 = stablehlo.add %v4692, %v4690 : tensor<256x256x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %v4693 : tensor<256x256x3x3xf32>
    %v4696 = stablehlo.subtract %d3W2, %v4695 : tensor<256x256x3x3xf32>
    %v4697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4698 = stablehlo.multiply %v4697, %d3b2 : tensor<256xf32>
    %v4699 = stablehlo.add %v4698, %v2344 : tensor<256xf32>
    %v4700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4701 = stablehlo.multiply %v4700, %d3b2v : tensor<256xf32>
    %v4702 = stablehlo.add %v4701, %v4699 : tensor<256xf32>
    %v4703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4704 = stablehlo.multiply %v4703, %v4702 : tensor<256xf32>
    %v4705 = stablehlo.subtract %d3b2, %v4704 : tensor<256xf32>
    %v4706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4707 = stablehlo.multiply %v4706, %d3g2 : tensor<256xf32>
    %v4708 = stablehlo.add %v4707, %v2362 : tensor<256xf32>
    %v4709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4710 = stablehlo.multiply %v4709, %d3g2v : tensor<256xf32>
    %v4711 = stablehlo.add %v4710, %v4708 : tensor<256xf32>
    %v4712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4713 = stablehlo.multiply %v4712, %v4711 : tensor<256xf32>
    %v4714 = stablehlo.subtract %d3g2, %v4713 : tensor<256xf32>
    %v4715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4716 = stablehlo.multiply %v4715, %d3bt2 : tensor<256xf32>
    %v4717 = stablehlo.add %v4716, %v2365 : tensor<256xf32>
    %v4718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4719 = stablehlo.multiply %v4718, %d3bt2v : tensor<256xf32>
    %v4720 = stablehlo.add %v4719, %v4717 : tensor<256xf32>
    %v4721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4722 = stablehlo.multiply %v4721, %v4720 : tensor<256xf32>
    %v4723 = stablehlo.subtract %d3bt2, %v4722 : tensor<256xf32>
    %v4724 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4725 = stablehlo.multiply %v4724, %d3Wp : tensor<256x128x3x3xf32>
    %v4726 = stablehlo.add %v4725, %v2373 : tensor<256x128x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4728 = stablehlo.multiply %v4727, %d3Wpv : tensor<256x128x3x3xf32>
    %v4729 = stablehlo.add %v4728, %v4726 : tensor<256x128x3x3xf32>
    %v4730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4731 = stablehlo.multiply %v4730, %v4729 : tensor<256x128x3x3xf32>
    %v4732 = stablehlo.subtract %d3Wp, %v4731 : tensor<256x128x3x3xf32>
    %v4733 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4734 = stablehlo.multiply %v4733, %d3bp : tensor<256xf32>
    %v4735 = stablehlo.add %v4734, %v2376 : tensor<256xf32>
    %v4736 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.multiply %v4736, %d3bpv : tensor<256xf32>
    %v4738 = stablehlo.add %v4737, %v4735 : tensor<256xf32>
    %v4739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4740 = stablehlo.multiply %v4739, %v4738 : tensor<256xf32>
    %v4741 = stablehlo.subtract %d3bp, %v4740 : tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4743 = stablehlo.multiply %v4742, %d3gp : tensor<256xf32>
    %v4744 = stablehlo.add %v4743, %v2394 : tensor<256xf32>
    %v4745 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4746 = stablehlo.multiply %v4745, %d3gpv : tensor<256xf32>
    %v4747 = stablehlo.add %v4746, %v4744 : tensor<256xf32>
    %v4748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4749 = stablehlo.multiply %v4748, %v4747 : tensor<256xf32>
    %v4750 = stablehlo.subtract %d3gp, %v4749 : tensor<256xf32>
    %v4751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4752 = stablehlo.multiply %v4751, %d3btp : tensor<256xf32>
    %v4753 = stablehlo.add %v4752, %v2397 : tensor<256xf32>
    %v4754 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4755 = stablehlo.multiply %v4754, %d3btpv : tensor<256xf32>
    %v4756 = stablehlo.add %v4755, %v4753 : tensor<256xf32>
    %v4757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4758 = stablehlo.multiply %v4757, %v4756 : tensor<256xf32>
    %v4759 = stablehlo.subtract %d3btp, %v4758 : tensor<256xf32>
    %v4760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4761 = stablehlo.multiply %v4760, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4762 = stablehlo.add %v4761, %v2133 : tensor<256x256x3x3xf32>
    %v4763 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4764 = stablehlo.multiply %v4763, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4765 = stablehlo.add %v4764, %v4762 : tensor<256x256x3x3xf32>
    %v4766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4767 = stablehlo.multiply %v4766, %v4765 : tensor<256x256x3x3xf32>
    %v4768 = stablehlo.subtract %s3b0W1, %v4767 : tensor<256x256x3x3xf32>
    %v4769 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4770 = stablehlo.multiply %v4769, %s3b0b1 : tensor<256xf32>
    %v4771 = stablehlo.add %v4770, %v2136 : tensor<256xf32>
    %v4772 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4773 = stablehlo.multiply %v4772, %s3b0b1v : tensor<256xf32>
    %v4774 = stablehlo.add %v4773, %v4771 : tensor<256xf32>
    %v4775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4776 = stablehlo.multiply %v4775, %v4774 : tensor<256xf32>
    %v4777 = stablehlo.subtract %s3b0b1, %v4776 : tensor<256xf32>
    %v4778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4779 = stablehlo.multiply %v4778, %s3b0g1 : tensor<256xf32>
    %v4780 = stablehlo.add %v4779, %v2154 : tensor<256xf32>
    %v4781 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4782 = stablehlo.multiply %v4781, %s3b0g1v : tensor<256xf32>
    %v4783 = stablehlo.add %v4782, %v4780 : tensor<256xf32>
    %v4784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4785 = stablehlo.multiply %v4784, %v4783 : tensor<256xf32>
    %v4786 = stablehlo.subtract %s3b0g1, %v4785 : tensor<256xf32>
    %v4787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4788 = stablehlo.multiply %v4787, %s3b0bt1 : tensor<256xf32>
    %v4789 = stablehlo.add %v4788, %v2157 : tensor<256xf32>
    %v4790 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4791 = stablehlo.multiply %v4790, %s3b0bt1v : tensor<256xf32>
    %v4792 = stablehlo.add %v4791, %v4789 : tensor<256xf32>
    %v4793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4794 = stablehlo.multiply %v4793, %v4792 : tensor<256xf32>
    %v4795 = stablehlo.subtract %s3b0bt1, %v4794 : tensor<256xf32>
    %v4796 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4797 = stablehlo.multiply %v4796, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4798 = stablehlo.add %v4797, %v2163 : tensor<256x256x3x3xf32>
    %v4799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4800 = stablehlo.multiply %v4799, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4801 = stablehlo.add %v4800, %v4798 : tensor<256x256x3x3xf32>
    %v4802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4803 = stablehlo.multiply %v4802, %v4801 : tensor<256x256x3x3xf32>
    %v4804 = stablehlo.subtract %s3b0W2, %v4803 : tensor<256x256x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4806 = stablehlo.multiply %v4805, %s3b0b2 : tensor<256xf32>
    %v4807 = stablehlo.add %v4806, %v2166 : tensor<256xf32>
    %v4808 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4809 = stablehlo.multiply %v4808, %s3b0b2v : tensor<256xf32>
    %v4810 = stablehlo.add %v4809, %v4807 : tensor<256xf32>
    %v4811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4812 = stablehlo.multiply %v4811, %v4810 : tensor<256xf32>
    %v4813 = stablehlo.subtract %s3b0b2, %v4812 : tensor<256xf32>
    %v4814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4815 = stablehlo.multiply %v4814, %s3b0g2 : tensor<256xf32>
    %v4816 = stablehlo.add %v4815, %v2184 : tensor<256xf32>
    %v4817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4818 = stablehlo.multiply %v4817, %s3b0g2v : tensor<256xf32>
    %v4819 = stablehlo.add %v4818, %v4816 : tensor<256xf32>
    %v4820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4821 = stablehlo.multiply %v4820, %v4819 : tensor<256xf32>
    %v4822 = stablehlo.subtract %s3b0g2, %v4821 : tensor<256xf32>
    %v4823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4824 = stablehlo.multiply %v4823, %s3b0bt2 : tensor<256xf32>
    %v4825 = stablehlo.add %v4824, %v2187 : tensor<256xf32>
    %v4826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4827 = stablehlo.multiply %v4826, %s3b0bt2v : tensor<256xf32>
    %v4828 = stablehlo.add %v4827, %v4825 : tensor<256xf32>
    %v4829 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4830 = stablehlo.multiply %v4829, %v4828 : tensor<256xf32>
    %v4831 = stablehlo.subtract %s3b0bt2, %v4830 : tensor<256xf32>
    %v4832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4833 = stablehlo.multiply %v4832, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4834 = stablehlo.add %v4833, %v1996 : tensor<256x256x3x3xf32>
    %v4835 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4836 = stablehlo.multiply %v4835, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4837 = stablehlo.add %v4836, %v4834 : tensor<256x256x3x3xf32>
    %v4838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4839 = stablehlo.multiply %v4838, %v4837 : tensor<256x256x3x3xf32>
    %v4840 = stablehlo.subtract %s3b1W1, %v4839 : tensor<256x256x3x3xf32>
    %v4841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4842 = stablehlo.multiply %v4841, %s3b1b1 : tensor<256xf32>
    %v4843 = stablehlo.add %v4842, %v1999 : tensor<256xf32>
    %v4844 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4845 = stablehlo.multiply %v4844, %s3b1b1v : tensor<256xf32>
    %v4846 = stablehlo.add %v4845, %v4843 : tensor<256xf32>
    %v4847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4848 = stablehlo.multiply %v4847, %v4846 : tensor<256xf32>
    %v4849 = stablehlo.subtract %s3b1b1, %v4848 : tensor<256xf32>
    %v4850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4851 = stablehlo.multiply %v4850, %s3b1g1 : tensor<256xf32>
    %v4852 = stablehlo.add %v4851, %v2017 : tensor<256xf32>
    %v4853 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4854 = stablehlo.multiply %v4853, %s3b1g1v : tensor<256xf32>
    %v4855 = stablehlo.add %v4854, %v4852 : tensor<256xf32>
    %v4856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4857 = stablehlo.multiply %v4856, %v4855 : tensor<256xf32>
    %v4858 = stablehlo.subtract %s3b1g1, %v4857 : tensor<256xf32>
    %v4859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4860 = stablehlo.multiply %v4859, %s3b1bt1 : tensor<256xf32>
    %v4861 = stablehlo.add %v4860, %v2020 : tensor<256xf32>
    %v4862 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4863 = stablehlo.multiply %v4862, %s3b1bt1v : tensor<256xf32>
    %v4864 = stablehlo.add %v4863, %v4861 : tensor<256xf32>
    %v4865 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4866 = stablehlo.multiply %v4865, %v4864 : tensor<256xf32>
    %v4867 = stablehlo.subtract %s3b1bt1, %v4866 : tensor<256xf32>
    %v4868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4869 = stablehlo.multiply %v4868, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4870 = stablehlo.add %v4869, %v2026 : tensor<256x256x3x3xf32>
    %v4871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4872 = stablehlo.multiply %v4871, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4873 = stablehlo.add %v4872, %v4870 : tensor<256x256x3x3xf32>
    %v4874 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4875 = stablehlo.multiply %v4874, %v4873 : tensor<256x256x3x3xf32>
    %v4876 = stablehlo.subtract %s3b1W2, %v4875 : tensor<256x256x3x3xf32>
    %v4877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4878 = stablehlo.multiply %v4877, %s3b1b2 : tensor<256xf32>
    %v4879 = stablehlo.add %v4878, %v2029 : tensor<256xf32>
    %v4880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4881 = stablehlo.multiply %v4880, %s3b1b2v : tensor<256xf32>
    %v4882 = stablehlo.add %v4881, %v4879 : tensor<256xf32>
    %v4883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4884 = stablehlo.multiply %v4883, %v4882 : tensor<256xf32>
    %v4885 = stablehlo.subtract %s3b1b2, %v4884 : tensor<256xf32>
    %v4886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4887 = stablehlo.multiply %v4886, %s3b1g2 : tensor<256xf32>
    %v4888 = stablehlo.add %v4887, %v2047 : tensor<256xf32>
    %v4889 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4890 = stablehlo.multiply %v4889, %s3b1g2v : tensor<256xf32>
    %v4891 = stablehlo.add %v4890, %v4888 : tensor<256xf32>
    %v4892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4893 = stablehlo.multiply %v4892, %v4891 : tensor<256xf32>
    %v4894 = stablehlo.subtract %s3b1g2, %v4893 : tensor<256xf32>
    %v4895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4896 = stablehlo.multiply %v4895, %s3b1bt2 : tensor<256xf32>
    %v4897 = stablehlo.add %v4896, %v2050 : tensor<256xf32>
    %v4898 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4899 = stablehlo.multiply %v4898, %s3b1bt2v : tensor<256xf32>
    %v4900 = stablehlo.add %v4899, %v4897 : tensor<256xf32>
    %v4901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4902 = stablehlo.multiply %v4901, %v4900 : tensor<256xf32>
    %v4903 = stablehlo.subtract %s3b1bt2, %v4902 : tensor<256xf32>
    %v4904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4905 = stablehlo.multiply %v4904, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4906 = stablehlo.add %v4905, %v1859 : tensor<256x256x3x3xf32>
    %v4907 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4908 = stablehlo.multiply %v4907, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4909 = stablehlo.add %v4908, %v4906 : tensor<256x256x3x3xf32>
    %v4910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4911 = stablehlo.multiply %v4910, %v4909 : tensor<256x256x3x3xf32>
    %v4912 = stablehlo.subtract %s3b2W1, %v4911 : tensor<256x256x3x3xf32>
    %v4913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4914 = stablehlo.multiply %v4913, %s3b2b1 : tensor<256xf32>
    %v4915 = stablehlo.add %v4914, %v1862 : tensor<256xf32>
    %v4916 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4917 = stablehlo.multiply %v4916, %s3b2b1v : tensor<256xf32>
    %v4918 = stablehlo.add %v4917, %v4915 : tensor<256xf32>
    %v4919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4920 = stablehlo.multiply %v4919, %v4918 : tensor<256xf32>
    %v4921 = stablehlo.subtract %s3b2b1, %v4920 : tensor<256xf32>
    %v4922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4923 = stablehlo.multiply %v4922, %s3b2g1 : tensor<256xf32>
    %v4924 = stablehlo.add %v4923, %v1880 : tensor<256xf32>
    %v4925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4926 = stablehlo.multiply %v4925, %s3b2g1v : tensor<256xf32>
    %v4927 = stablehlo.add %v4926, %v4924 : tensor<256xf32>
    %v4928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4929 = stablehlo.multiply %v4928, %v4927 : tensor<256xf32>
    %v4930 = stablehlo.subtract %s3b2g1, %v4929 : tensor<256xf32>
    %v4931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4932 = stablehlo.multiply %v4931, %s3b2bt1 : tensor<256xf32>
    %v4933 = stablehlo.add %v4932, %v1883 : tensor<256xf32>
    %v4934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4935 = stablehlo.multiply %v4934, %s3b2bt1v : tensor<256xf32>
    %v4936 = stablehlo.add %v4935, %v4933 : tensor<256xf32>
    %v4937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4938 = stablehlo.multiply %v4937, %v4936 : tensor<256xf32>
    %v4939 = stablehlo.subtract %s3b2bt1, %v4938 : tensor<256xf32>
    %v4940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4941 = stablehlo.multiply %v4940, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4942 = stablehlo.add %v4941, %v1889 : tensor<256x256x3x3xf32>
    %v4943 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4944 = stablehlo.multiply %v4943, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4945 = stablehlo.add %v4944, %v4942 : tensor<256x256x3x3xf32>
    %v4946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4947 = stablehlo.multiply %v4946, %v4945 : tensor<256x256x3x3xf32>
    %v4948 = stablehlo.subtract %s3b2W2, %v4947 : tensor<256x256x3x3xf32>
    %v4949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4950 = stablehlo.multiply %v4949, %s3b2b2 : tensor<256xf32>
    %v4951 = stablehlo.add %v4950, %v1892 : tensor<256xf32>
    %v4952 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4953 = stablehlo.multiply %v4952, %s3b2b2v : tensor<256xf32>
    %v4954 = stablehlo.add %v4953, %v4951 : tensor<256xf32>
    %v4955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4956 = stablehlo.multiply %v4955, %v4954 : tensor<256xf32>
    %v4957 = stablehlo.subtract %s3b2b2, %v4956 : tensor<256xf32>
    %v4958 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4959 = stablehlo.multiply %v4958, %s3b2g2 : tensor<256xf32>
    %v4960 = stablehlo.add %v4959, %v1910 : tensor<256xf32>
    %v4961 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4962 = stablehlo.multiply %v4961, %s3b2g2v : tensor<256xf32>
    %v4963 = stablehlo.add %v4962, %v4960 : tensor<256xf32>
    %v4964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4965 = stablehlo.multiply %v4964, %v4963 : tensor<256xf32>
    %v4966 = stablehlo.subtract %s3b2g2, %v4965 : tensor<256xf32>
    %v4967 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4968 = stablehlo.multiply %v4967, %s3b2bt2 : tensor<256xf32>
    %v4969 = stablehlo.add %v4968, %v1913 : tensor<256xf32>
    %v4970 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4971 = stablehlo.multiply %v4970, %s3b2bt2v : tensor<256xf32>
    %v4972 = stablehlo.add %v4971, %v4969 : tensor<256xf32>
    %v4973 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4974 = stablehlo.multiply %v4973, %v4972 : tensor<256xf32>
    %v4975 = stablehlo.subtract %s3b2bt2, %v4974 : tensor<256xf32>
    %v4976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4977 = stablehlo.multiply %v4976, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4978 = stablehlo.add %v4977, %v1722 : tensor<256x256x3x3xf32>
    %v4979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4980 = stablehlo.multiply %v4979, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4981 = stablehlo.add %v4980, %v4978 : tensor<256x256x3x3xf32>
    %v4982 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4983 = stablehlo.multiply %v4982, %v4981 : tensor<256x256x3x3xf32>
    %v4984 = stablehlo.subtract %s3b3W1, %v4983 : tensor<256x256x3x3xf32>
    %v4985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4986 = stablehlo.multiply %v4985, %s3b3b1 : tensor<256xf32>
    %v4987 = stablehlo.add %v4986, %v1725 : tensor<256xf32>
    %v4988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4989 = stablehlo.multiply %v4988, %s3b3b1v : tensor<256xf32>
    %v4990 = stablehlo.add %v4989, %v4987 : tensor<256xf32>
    %v4991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4992 = stablehlo.multiply %v4991, %v4990 : tensor<256xf32>
    %v4993 = stablehlo.subtract %s3b3b1, %v4992 : tensor<256xf32>
    %v4994 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4995 = stablehlo.multiply %v4994, %s3b3g1 : tensor<256xf32>
    %v4996 = stablehlo.add %v4995, %v1743 : tensor<256xf32>
    %v4997 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4998 = stablehlo.multiply %v4997, %s3b3g1v : tensor<256xf32>
    %v4999 = stablehlo.add %v4998, %v4996 : tensor<256xf32>
    %v5000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5001 = stablehlo.multiply %v5000, %v4999 : tensor<256xf32>
    %v5002 = stablehlo.subtract %s3b3g1, %v5001 : tensor<256xf32>
    %v5003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5004 = stablehlo.multiply %v5003, %s3b3bt1 : tensor<256xf32>
    %v5005 = stablehlo.add %v5004, %v1746 : tensor<256xf32>
    %v5006 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5007 = stablehlo.multiply %v5006, %s3b3bt1v : tensor<256xf32>
    %v5008 = stablehlo.add %v5007, %v5005 : tensor<256xf32>
    %v5009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5010 = stablehlo.multiply %v5009, %v5008 : tensor<256xf32>
    %v5011 = stablehlo.subtract %s3b3bt1, %v5010 : tensor<256xf32>
    %v5012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5013 = stablehlo.multiply %v5012, %s3b3W2 : tensor<256x256x3x3xf32>
    %v5014 = stablehlo.add %v5013, %v1752 : tensor<256x256x3x3xf32>
    %v5015 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5016 = stablehlo.multiply %v5015, %s3b3W2v : tensor<256x256x3x3xf32>
    %v5017 = stablehlo.add %v5016, %v5014 : tensor<256x256x3x3xf32>
    %v5018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %v5017 : tensor<256x256x3x3xf32>
    %v5020 = stablehlo.subtract %s3b3W2, %v5019 : tensor<256x256x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5022 = stablehlo.multiply %v5021, %s3b3b2 : tensor<256xf32>
    %v5023 = stablehlo.add %v5022, %v1755 : tensor<256xf32>
    %v5024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5025 = stablehlo.multiply %v5024, %s3b3b2v : tensor<256xf32>
    %v5026 = stablehlo.add %v5025, %v5023 : tensor<256xf32>
    %v5027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5028 = stablehlo.multiply %v5027, %v5026 : tensor<256xf32>
    %v5029 = stablehlo.subtract %s3b3b2, %v5028 : tensor<256xf32>
    %v5030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5031 = stablehlo.multiply %v5030, %s3b3g2 : tensor<256xf32>
    %v5032 = stablehlo.add %v5031, %v1773 : tensor<256xf32>
    %v5033 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5034 = stablehlo.multiply %v5033, %s3b3g2v : tensor<256xf32>
    %v5035 = stablehlo.add %v5034, %v5032 : tensor<256xf32>
    %v5036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5037 = stablehlo.multiply %v5036, %v5035 : tensor<256xf32>
    %v5038 = stablehlo.subtract %s3b3g2, %v5037 : tensor<256xf32>
    %v5039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5040 = stablehlo.multiply %v5039, %s3b3bt2 : tensor<256xf32>
    %v5041 = stablehlo.add %v5040, %v1776 : tensor<256xf32>
    %v5042 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5043 = stablehlo.multiply %v5042, %s3b3bt2v : tensor<256xf32>
    %v5044 = stablehlo.add %v5043, %v5041 : tensor<256xf32>
    %v5045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5046 = stablehlo.multiply %v5045, %v5044 : tensor<256xf32>
    %v5047 = stablehlo.subtract %s3b3bt2, %v5046 : tensor<256xf32>
    %v5048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5049 = stablehlo.multiply %v5048, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5050 = stablehlo.add %v5049, %v1585 : tensor<256x256x3x3xf32>
    %v5051 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5052 = stablehlo.multiply %v5051, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5053 = stablehlo.add %v5052, %v5050 : tensor<256x256x3x3xf32>
    %v5054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5055 = stablehlo.multiply %v5054, %v5053 : tensor<256x256x3x3xf32>
    %v5056 = stablehlo.subtract %s3b4W1, %v5055 : tensor<256x256x3x3xf32>
    %v5057 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5058 = stablehlo.multiply %v5057, %s3b4b1 : tensor<256xf32>
    %v5059 = stablehlo.add %v5058, %v1588 : tensor<256xf32>
    %v5060 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5061 = stablehlo.multiply %v5060, %s3b4b1v : tensor<256xf32>
    %v5062 = stablehlo.add %v5061, %v5059 : tensor<256xf32>
    %v5063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5064 = stablehlo.multiply %v5063, %v5062 : tensor<256xf32>
    %v5065 = stablehlo.subtract %s3b4b1, %v5064 : tensor<256xf32>
    %v5066 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.multiply %v5066, %s3b4g1 : tensor<256xf32>
    %v5068 = stablehlo.add %v5067, %v1606 : tensor<256xf32>
    %v5069 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5070 = stablehlo.multiply %v5069, %s3b4g1v : tensor<256xf32>
    %v5071 = stablehlo.add %v5070, %v5068 : tensor<256xf32>
    %v5072 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5073 = stablehlo.multiply %v5072, %v5071 : tensor<256xf32>
    %v5074 = stablehlo.subtract %s3b4g1, %v5073 : tensor<256xf32>
    %v5075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5076 = stablehlo.multiply %v5075, %s3b4bt1 : tensor<256xf32>
    %v5077 = stablehlo.add %v5076, %v1609 : tensor<256xf32>
    %v5078 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5079 = stablehlo.multiply %v5078, %s3b4bt1v : tensor<256xf32>
    %v5080 = stablehlo.add %v5079, %v5077 : tensor<256xf32>
    %v5081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5082 = stablehlo.multiply %v5081, %v5080 : tensor<256xf32>
    %v5083 = stablehlo.subtract %s3b4bt1, %v5082 : tensor<256xf32>
    %v5084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5085 = stablehlo.multiply %v5084, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5086 = stablehlo.add %v5085, %v1615 : tensor<256x256x3x3xf32>
    %v5087 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5088 = stablehlo.multiply %v5087, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5089 = stablehlo.add %v5088, %v5086 : tensor<256x256x3x3xf32>
    %v5090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5091 = stablehlo.multiply %v5090, %v5089 : tensor<256x256x3x3xf32>
    %v5092 = stablehlo.subtract %s3b4W2, %v5091 : tensor<256x256x3x3xf32>
    %v5093 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5094 = stablehlo.multiply %v5093, %s3b4b2 : tensor<256xf32>
    %v5095 = stablehlo.add %v5094, %v1618 : tensor<256xf32>
    %v5096 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5097 = stablehlo.multiply %v5096, %s3b4b2v : tensor<256xf32>
    %v5098 = stablehlo.add %v5097, %v5095 : tensor<256xf32>
    %v5099 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5100 = stablehlo.multiply %v5099, %v5098 : tensor<256xf32>
    %v5101 = stablehlo.subtract %s3b4b2, %v5100 : tensor<256xf32>
    %v5102 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5103 = stablehlo.multiply %v5102, %s3b4g2 : tensor<256xf32>
    %v5104 = stablehlo.add %v5103, %v1636 : tensor<256xf32>
    %v5105 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5106 = stablehlo.multiply %v5105, %s3b4g2v : tensor<256xf32>
    %v5107 = stablehlo.add %v5106, %v5104 : tensor<256xf32>
    %v5108 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5109 = stablehlo.multiply %v5108, %v5107 : tensor<256xf32>
    %v5110 = stablehlo.subtract %s3b4g2, %v5109 : tensor<256xf32>
    %v5111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5112 = stablehlo.multiply %v5111, %s3b4bt2 : tensor<256xf32>
    %v5113 = stablehlo.add %v5112, %v1639 : tensor<256xf32>
    %v5114 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5115 = stablehlo.multiply %v5114, %s3b4bt2v : tensor<256xf32>
    %v5116 = stablehlo.add %v5115, %v5113 : tensor<256xf32>
    %v5117 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5118 = stablehlo.multiply %v5117, %v5116 : tensor<256xf32>
    %v5119 = stablehlo.subtract %s3b4bt2, %v5118 : tensor<256xf32>
    %v5120 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5121 = stablehlo.multiply %v5120, %d4W1 : tensor<512x256x3x3xf32>
    %v5122 = stablehlo.add %v5121, %v1416 : tensor<512x256x3x3xf32>
    %v5123 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5124 = stablehlo.multiply %v5123, %d4W1v : tensor<512x256x3x3xf32>
    %v5125 = stablehlo.add %v5124, %v5122 : tensor<512x256x3x3xf32>
    %v5126 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5127 = stablehlo.multiply %v5126, %v5125 : tensor<512x256x3x3xf32>
    %v5128 = stablehlo.subtract %d4W1, %v5127 : tensor<512x256x3x3xf32>
    %v5129 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5130 = stablehlo.multiply %v5129, %d4b1 : tensor<512xf32>
    %v5131 = stablehlo.add %v5130, %v1419 : tensor<512xf32>
    %v5132 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5133 = stablehlo.multiply %v5132, %d4b1v : tensor<512xf32>
    %v5134 = stablehlo.add %v5133, %v5131 : tensor<512xf32>
    %v5135 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5136 = stablehlo.multiply %v5135, %v5134 : tensor<512xf32>
    %v5137 = stablehlo.subtract %d4b1, %v5136 : tensor<512xf32>
    %v5138 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5139 = stablehlo.multiply %v5138, %d4g1 : tensor<512xf32>
    %v5140 = stablehlo.add %v5139, %v1437 : tensor<512xf32>
    %v5141 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5142 = stablehlo.multiply %v5141, %d4g1v : tensor<512xf32>
    %v5143 = stablehlo.add %v5142, %v5140 : tensor<512xf32>
    %v5144 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5145 = stablehlo.multiply %v5144, %v5143 : tensor<512xf32>
    %v5146 = stablehlo.subtract %d4g1, %v5145 : tensor<512xf32>
    %v5147 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5148 = stablehlo.multiply %v5147, %d4bt1 : tensor<512xf32>
    %v5149 = stablehlo.add %v5148, %v1440 : tensor<512xf32>
    %v5150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5151 = stablehlo.multiply %v5150, %d4bt1v : tensor<512xf32>
    %v5152 = stablehlo.add %v5151, %v5149 : tensor<512xf32>
    %v5153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5154 = stablehlo.multiply %v5153, %v5152 : tensor<512xf32>
    %v5155 = stablehlo.subtract %d4bt1, %v5154 : tensor<512xf32>
    %v5156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5157 = stablehlo.multiply %v5156, %d4W2 : tensor<512x512x3x3xf32>
    %v5158 = stablehlo.add %v5157, %v1446 : tensor<512x512x3x3xf32>
    %v5159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5160 = stablehlo.multiply %v5159, %d4W2v : tensor<512x512x3x3xf32>
    %v5161 = stablehlo.add %v5160, %v5158 : tensor<512x512x3x3xf32>
    %v5162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5163 = stablehlo.multiply %v5162, %v5161 : tensor<512x512x3x3xf32>
    %v5164 = stablehlo.subtract %d4W2, %v5163 : tensor<512x512x3x3xf32>
    %v5165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5166 = stablehlo.multiply %v5165, %d4b2 : tensor<512xf32>
    %v5167 = stablehlo.add %v5166, %v1449 : tensor<512xf32>
    %v5168 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5169 = stablehlo.multiply %v5168, %d4b2v : tensor<512xf32>
    %v5170 = stablehlo.add %v5169, %v5167 : tensor<512xf32>
    %v5171 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5172 = stablehlo.multiply %v5171, %v5170 : tensor<512xf32>
    %v5173 = stablehlo.subtract %d4b2, %v5172 : tensor<512xf32>
    %v5174 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5175 = stablehlo.multiply %v5174, %d4g2 : tensor<512xf32>
    %v5176 = stablehlo.add %v5175, %v1467 : tensor<512xf32>
    %v5177 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5178 = stablehlo.multiply %v5177, %d4g2v : tensor<512xf32>
    %v5179 = stablehlo.add %v5178, %v5176 : tensor<512xf32>
    %v5180 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5181 = stablehlo.multiply %v5180, %v5179 : tensor<512xf32>
    %v5182 = stablehlo.subtract %d4g2, %v5181 : tensor<512xf32>
    %v5183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5184 = stablehlo.multiply %v5183, %d4bt2 : tensor<512xf32>
    %v5185 = stablehlo.add %v5184, %v1470 : tensor<512xf32>
    %v5186 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5187 = stablehlo.multiply %v5186, %d4bt2v : tensor<512xf32>
    %v5188 = stablehlo.add %v5187, %v5185 : tensor<512xf32>
    %v5189 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5190 = stablehlo.multiply %v5189, %v5188 : tensor<512xf32>
    %v5191 = stablehlo.subtract %d4bt2, %v5190 : tensor<512xf32>
    %v5192 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5193 = stablehlo.multiply %v5192, %d4Wp : tensor<512x256x3x3xf32>
    %v5194 = stablehlo.add %v5193, %v1478 : tensor<512x256x3x3xf32>
    %v5195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5196 = stablehlo.multiply %v5195, %d4Wpv : tensor<512x256x3x3xf32>
    %v5197 = stablehlo.add %v5196, %v5194 : tensor<512x256x3x3xf32>
    %v5198 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5199 = stablehlo.multiply %v5198, %v5197 : tensor<512x256x3x3xf32>
    %v5200 = stablehlo.subtract %d4Wp, %v5199 : tensor<512x256x3x3xf32>
    %v5201 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5202 = stablehlo.multiply %v5201, %d4bp : tensor<512xf32>
    %v5203 = stablehlo.add %v5202, %v1481 : tensor<512xf32>
    %v5204 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5205 = stablehlo.multiply %v5204, %d4bpv : tensor<512xf32>
    %v5206 = stablehlo.add %v5205, %v5203 : tensor<512xf32>
    %v5207 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5208 = stablehlo.multiply %v5207, %v5206 : tensor<512xf32>
    %v5209 = stablehlo.subtract %d4bp, %v5208 : tensor<512xf32>
    %v5210 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5211 = stablehlo.multiply %v5210, %d4gp : tensor<512xf32>
    %v5212 = stablehlo.add %v5211, %v1499 : tensor<512xf32>
    %v5213 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5214 = stablehlo.multiply %v5213, %d4gpv : tensor<512xf32>
    %v5215 = stablehlo.add %v5214, %v5212 : tensor<512xf32>
    %v5216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5217 = stablehlo.multiply %v5216, %v5215 : tensor<512xf32>
    %v5218 = stablehlo.subtract %d4gp, %v5217 : tensor<512xf32>
    %v5219 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5220 = stablehlo.multiply %v5219, %d4btp : tensor<512xf32>
    %v5221 = stablehlo.add %v5220, %v1502 : tensor<512xf32>
    %v5222 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5223 = stablehlo.multiply %v5222, %d4btpv : tensor<512xf32>
    %v5224 = stablehlo.add %v5223, %v5221 : tensor<512xf32>
    %v5225 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5226 = stablehlo.multiply %v5225, %v5224 : tensor<512xf32>
    %v5227 = stablehlo.subtract %d4btp, %v5226 : tensor<512xf32>
    %v5228 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5229 = stablehlo.multiply %v5228, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5230 = stablehlo.add %v5229, %v1238 : tensor<512x512x3x3xf32>
    %v5231 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5232 = stablehlo.multiply %v5231, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5233 = stablehlo.add %v5232, %v5230 : tensor<512x512x3x3xf32>
    %v5234 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5235 = stablehlo.multiply %v5234, %v5233 : tensor<512x512x3x3xf32>
    %v5236 = stablehlo.subtract %s4b0W1, %v5235 : tensor<512x512x3x3xf32>
    %v5237 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5238 = stablehlo.multiply %v5237, %s4b0b1 : tensor<512xf32>
    %v5239 = stablehlo.add %v5238, %v1241 : tensor<512xf32>
    %v5240 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5241 = stablehlo.multiply %v5240, %s4b0b1v : tensor<512xf32>
    %v5242 = stablehlo.add %v5241, %v5239 : tensor<512xf32>
    %v5243 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5244 = stablehlo.multiply %v5243, %v5242 : tensor<512xf32>
    %v5245 = stablehlo.subtract %s4b0b1, %v5244 : tensor<512xf32>
    %v5246 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5247 = stablehlo.multiply %v5246, %s4b0g1 : tensor<512xf32>
    %v5248 = stablehlo.add %v5247, %v1259 : tensor<512xf32>
    %v5249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5250 = stablehlo.multiply %v5249, %s4b0g1v : tensor<512xf32>
    %v5251 = stablehlo.add %v5250, %v5248 : tensor<512xf32>
    %v5252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5253 = stablehlo.multiply %v5252, %v5251 : tensor<512xf32>
    %v5254 = stablehlo.subtract %s4b0g1, %v5253 : tensor<512xf32>
    %v5255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5256 = stablehlo.multiply %v5255, %s4b0bt1 : tensor<512xf32>
    %v5257 = stablehlo.add %v5256, %v1262 : tensor<512xf32>
    %v5258 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5259 = stablehlo.multiply %v5258, %s4b0bt1v : tensor<512xf32>
    %v5260 = stablehlo.add %v5259, %v5257 : tensor<512xf32>
    %v5261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5262 = stablehlo.multiply %v5261, %v5260 : tensor<512xf32>
    %v5263 = stablehlo.subtract %s4b0bt1, %v5262 : tensor<512xf32>
    %v5264 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5265 = stablehlo.multiply %v5264, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5266 = stablehlo.add %v5265, %v1268 : tensor<512x512x3x3xf32>
    %v5267 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5268 = stablehlo.multiply %v5267, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5269 = stablehlo.add %v5268, %v5266 : tensor<512x512x3x3xf32>
    %v5270 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5271 = stablehlo.multiply %v5270, %v5269 : tensor<512x512x3x3xf32>
    %v5272 = stablehlo.subtract %s4b0W2, %v5271 : tensor<512x512x3x3xf32>
    %v5273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5274 = stablehlo.multiply %v5273, %s4b0b2 : tensor<512xf32>
    %v5275 = stablehlo.add %v5274, %v1271 : tensor<512xf32>
    %v5276 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5277 = stablehlo.multiply %v5276, %s4b0b2v : tensor<512xf32>
    %v5278 = stablehlo.add %v5277, %v5275 : tensor<512xf32>
    %v5279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5280 = stablehlo.multiply %v5279, %v5278 : tensor<512xf32>
    %v5281 = stablehlo.subtract %s4b0b2, %v5280 : tensor<512xf32>
    %v5282 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5283 = stablehlo.multiply %v5282, %s4b0g2 : tensor<512xf32>
    %v5284 = stablehlo.add %v5283, %v1289 : tensor<512xf32>
    %v5285 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5286 = stablehlo.multiply %v5285, %s4b0g2v : tensor<512xf32>
    %v5287 = stablehlo.add %v5286, %v5284 : tensor<512xf32>
    %v5288 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5289 = stablehlo.multiply %v5288, %v5287 : tensor<512xf32>
    %v5290 = stablehlo.subtract %s4b0g2, %v5289 : tensor<512xf32>
    %v5291 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5292 = stablehlo.multiply %v5291, %s4b0bt2 : tensor<512xf32>
    %v5293 = stablehlo.add %v5292, %v1292 : tensor<512xf32>
    %v5294 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5295 = stablehlo.multiply %v5294, %s4b0bt2v : tensor<512xf32>
    %v5296 = stablehlo.add %v5295, %v5293 : tensor<512xf32>
    %v5297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5298 = stablehlo.multiply %v5297, %v5296 : tensor<512xf32>
    %v5299 = stablehlo.subtract %s4b0bt2, %v5298 : tensor<512xf32>
    %v5300 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5301 = stablehlo.multiply %v5300, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5302 = stablehlo.add %v5301, %v1101 : tensor<512x512x3x3xf32>
    %v5303 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5304 = stablehlo.multiply %v5303, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5305 = stablehlo.add %v5304, %v5302 : tensor<512x512x3x3xf32>
    %v5306 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5307 = stablehlo.multiply %v5306, %v5305 : tensor<512x512x3x3xf32>
    %v5308 = stablehlo.subtract %s4b1W1, %v5307 : tensor<512x512x3x3xf32>
    %v5309 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5310 = stablehlo.multiply %v5309, %s4b1b1 : tensor<512xf32>
    %v5311 = stablehlo.add %v5310, %v1104 : tensor<512xf32>
    %v5312 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5313 = stablehlo.multiply %v5312, %s4b1b1v : tensor<512xf32>
    %v5314 = stablehlo.add %v5313, %v5311 : tensor<512xf32>
    %v5315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5316 = stablehlo.multiply %v5315, %v5314 : tensor<512xf32>
    %v5317 = stablehlo.subtract %s4b1b1, %v5316 : tensor<512xf32>
    %v5318 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5319 = stablehlo.multiply %v5318, %s4b1g1 : tensor<512xf32>
    %v5320 = stablehlo.add %v5319, %v1122 : tensor<512xf32>
    %v5321 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5322 = stablehlo.multiply %v5321, %s4b1g1v : tensor<512xf32>
    %v5323 = stablehlo.add %v5322, %v5320 : tensor<512xf32>
    %v5324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5325 = stablehlo.multiply %v5324, %v5323 : tensor<512xf32>
    %v5326 = stablehlo.subtract %s4b1g1, %v5325 : tensor<512xf32>
    %v5327 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5328 = stablehlo.multiply %v5327, %s4b1bt1 : tensor<512xf32>
    %v5329 = stablehlo.add %v5328, %v1125 : tensor<512xf32>
    %v5330 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5331 = stablehlo.multiply %v5330, %s4b1bt1v : tensor<512xf32>
    %v5332 = stablehlo.add %v5331, %v5329 : tensor<512xf32>
    %v5333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5334 = stablehlo.multiply %v5333, %v5332 : tensor<512xf32>
    %v5335 = stablehlo.subtract %s4b1bt1, %v5334 : tensor<512xf32>
    %v5336 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5337 = stablehlo.multiply %v5336, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5338 = stablehlo.add %v5337, %v1131 : tensor<512x512x3x3xf32>
    %v5339 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5340 = stablehlo.multiply %v5339, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5341 = stablehlo.add %v5340, %v5338 : tensor<512x512x3x3xf32>
    %v5342 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5343 = stablehlo.multiply %v5342, %v5341 : tensor<512x512x3x3xf32>
    %v5344 = stablehlo.subtract %s4b1W2, %v5343 : tensor<512x512x3x3xf32>
    %v5345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5346 = stablehlo.multiply %v5345, %s4b1b2 : tensor<512xf32>
    %v5347 = stablehlo.add %v5346, %v1134 : tensor<512xf32>
    %v5348 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5349 = stablehlo.multiply %v5348, %s4b1b2v : tensor<512xf32>
    %v5350 = stablehlo.add %v5349, %v5347 : tensor<512xf32>
    %v5351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5352 = stablehlo.multiply %v5351, %v5350 : tensor<512xf32>
    %v5353 = stablehlo.subtract %s4b1b2, %v5352 : tensor<512xf32>
    %v5354 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5355 = stablehlo.multiply %v5354, %s4b1g2 : tensor<512xf32>
    %v5356 = stablehlo.add %v5355, %v1152 : tensor<512xf32>
    %v5357 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5358 = stablehlo.multiply %v5357, %s4b1g2v : tensor<512xf32>
    %v5359 = stablehlo.add %v5358, %v5356 : tensor<512xf32>
    %v5360 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5361 = stablehlo.multiply %v5360, %v5359 : tensor<512xf32>
    %v5362 = stablehlo.subtract %s4b1g2, %v5361 : tensor<512xf32>
    %v5363 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5364 = stablehlo.multiply %v5363, %s4b1bt2 : tensor<512xf32>
    %v5365 = stablehlo.add %v5364, %v1155 : tensor<512xf32>
    %v5366 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5367 = stablehlo.multiply %v5366, %s4b1bt2v : tensor<512xf32>
    %v5368 = stablehlo.add %v5367, %v5365 : tensor<512xf32>
    %v5369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5370 = stablehlo.multiply %v5369, %v5368 : tensor<512xf32>
    %v5371 = stablehlo.subtract %s4b1bt2, %v5370 : tensor<512xf32>
    %v5372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5373 = stablehlo.multiply %v5372, %Wd : tensor<512x1000xf32>
    %v5374 = stablehlo.add %v5373, %v1012 : tensor<512x1000xf32>
    %v5375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5376 = stablehlo.multiply %v5375, %Wdv : tensor<512x1000xf32>
    %v5377 = stablehlo.add %v5376, %v5374 : tensor<512x1000xf32>
    %v5378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5379 = stablehlo.multiply %v5378, %v5377 : tensor<512x1000xf32>
    %v5380 = stablehlo.subtract %Wd, %v5379 : tensor<512x1000xf32>
    %v5381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5382 = stablehlo.multiply %v5381, %bd : tensor<1000xf32>
    %v5383 = stablehlo.add %v5382, %v1014 : tensor<1000xf32>
    %v5384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5385 = stablehlo.multiply %v5384, %bdv : tensor<1000xf32>
    %v5386 = stablehlo.add %v5385, %v5383 : tensor<1000xf32>
    %v5387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5388 = stablehlo.multiply %v5387, %v5386 : tensor<1000xf32>
    %v5389 = stablehlo.subtract %bd, %v5388 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1000 : tensor<256x1000xf32>
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
    return %v4084, %v4093, %v4102, %v4111, %v4120, %v4129, %v4138, %v4147, %v4156, %v4165, %v4174, %v4183, %v4192, %v4201, %v4210, %v4219, %v4228, %v4237, %v4246, %v4255, %v4264, %v4273, %v4282, %v4291, %v4300, %v4309, %v4318, %v4327, %v4336, %v4345, %v4354, %v4363, %v4372, %v4381, %v4390, %v4399, %v4408, %v4417, %v4426, %v4435, %v4444, %v4453, %v4462, %v4471, %v4480, %v4489, %v4498, %v4507, %v4516, %v4525, %v4534, %v4543, %v4552, %v4561, %v4570, %v4579, %v4588, %v4597, %v4606, %v4615, %v4624, %v4633, %v4642, %v4651, %v4660, %v4669, %v4678, %v4687, %v4696, %v4705, %v4714, %v4723, %v4732, %v4741, %v4750, %v4759, %v4768, %v4777, %v4786, %v4795, %v4804, %v4813, %v4822, %v4831, %v4840, %v4849, %v4858, %v4867, %v4876, %v4885, %v4894, %v4903, %v4912, %v4921, %v4930, %v4939, %v4948, %v4957, %v4966, %v4975, %v4984, %v4993, %v5002, %v5011, %v5020, %v5029, %v5038, %v5047, %v5056, %v5065, %v5074, %v5083, %v5092, %v5101, %v5110, %v5119, %v5128, %v5137, %v5146, %v5155, %v5164, %v5173, %v5182, %v5191, %v5200, %v5209, %v5218, %v5227, %v5236, %v5245, %v5254, %v5263, %v5272, %v5281, %v5290, %v5299, %v5308, %v5317, %v5326, %v5335, %v5344, %v5353, %v5362, %v5371, %v5380, %v5389, %sWm, %sbim, %sgm, %sbtm, %s1b0W1m, %s1b0b1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0b2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1b1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1b2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2b1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2b2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2b1m, %d2g1m, %d2bt1m, %d2W2m, %d2b2m, %d2g2m, %d2bt2m, %d2Wpm, %d2bpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0b1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0b2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1b1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1b2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2b1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2b2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3b1m, %d3g1m, %d3bt1m, %d3W2m, %d3b2m, %d3g2m, %d3bt2m, %d3Wpm, %d3bpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0b1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0b2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1b1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1b2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2b1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2b2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3b1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3b2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4b1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4b2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4b1m, %d4g1m, %d4bt1m, %d4W2m, %d4b2m, %d4g2m, %d4bt2m, %d4Wpm, %d4bpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0b1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0b2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1b1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1b2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4081, %v4090, %v4099, %v4108, %v4117, %v4126, %v4135, %v4144, %v4153, %v4162, %v4171, %v4180, %v4189, %v4198, %v4207, %v4216, %v4225, %v4234, %v4243, %v4252, %v4261, %v4270, %v4279, %v4288, %v4297, %v4306, %v4315, %v4324, %v4333, %v4342, %v4351, %v4360, %v4369, %v4378, %v4387, %v4396, %v4405, %v4414, %v4423, %v4432, %v4441, %v4450, %v4459, %v4468, %v4477, %v4486, %v4495, %v4504, %v4513, %v4522, %v4531, %v4540, %v4549, %v4558, %v4567, %v4576, %v4585, %v4594, %v4603, %v4612, %v4621, %v4630, %v4639, %v4648, %v4657, %v4666, %v4675, %v4684, %v4693, %v4702, %v4711, %v4720, %v4729, %v4738, %v4747, %v4756, %v4765, %v4774, %v4783, %v4792, %v4801, %v4810, %v4819, %v4828, %v4837, %v4846, %v4855, %v4864, %v4873, %v4882, %v4891, %v4900, %v4909, %v4918, %v4927, %v4936, %v4945, %v4954, %v4963, %v4972, %v4981, %v4990, %v4999, %v5008, %v5017, %v5026, %v5035, %v5044, %v5053, %v5062, %v5071, %v5080, %v5089, %v5098, %v5107, %v5116, %v5125, %v5134, %v5143, %v5152, %v5161, %v5170, %v5179, %v5188, %v5197, %v5206, %v5215, %v5224, %v5233, %v5242, %v5251, %v5260, %v5269, %v5278, %v5287, %v5296, %v5305, %v5314, %v5323, %v5332, %v5341, %v5350, %v5359, %v5368, %v5377, %v5386, %loss, %bc1, %bc2, %v3504, %v3515, %v3520, %v3531, %v3536, %v3547, %v3552, %v3563, %v3568, %v3579, %v3584, %v3595, %v3600, %v3611, %v3616, %v3627, %v3632, %v3643, %v3648, %v3659, %v3664, %v3675, %v3680, %v3691, %v3696, %v3707, %v3712, %v3723, %v3728, %v3739, %v3744, %v3755, %v3760, %v3771, %v3776, %v3787, %v3792, %v3803, %v3808, %v3819, %v3824, %v3835, %v3840, %v3851, %v3856, %v3867, %v3872, %v3883, %v3888, %v3899, %v3904, %v3915, %v3920, %v3931, %v3936, %v3947, %v3952, %v3963, %v3968, %v3979, %v3984, %v3995, %v4000, %v4011, %v4016, %v4027, %v4032, %v4043, %v4048, %v4059, %v4064, %v4075 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}
