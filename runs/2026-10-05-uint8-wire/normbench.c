// Standalone timing of lean_mlir_u8_norm's loop at the MNv4 batch (512 x 224² x 3).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){ size_t B=512,hw=224*224,f=3*hw; uint8_t*base=malloc(4*B*f); for(size_t i=0;i<B*f;i++)base[i]=i*7;
 float m[3]={123.675f,116.28f,103.53f},sd[3]={58.395f,57.12f,57.375f}; uint8_t*tmp=malloc(f);
 for(int r=0;r<3;r++){ double t=now();
 for(size_t i=B;i-->0;){ memcpy(tmp,base+i*f,f); float*dst=(float*)(base+4*i*f);
  for(size_t c=0;c<3;c++){const float mc=m[c],sc=sd[c];float*d=dst+c*hw;const uint8_t*src=tmp+c;
   for(size_t p=0;p<hw;p++) d[p]=((float)src[3*p]-mc)/sc;}}
 printf("%.1f ms per batch\n",(now()-t)*1e3);} return 0;}
