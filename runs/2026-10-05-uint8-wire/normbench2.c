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
  float*d0=dst,*d1=dst+hw,*d2=dst+2*hw; const float m0=m[0],m1=m[1],m2=m[2],s0=sd[0],s1=sd[1],s2=sd[2];
  for(size_t p=0;p<hw;p++){const uint8_t*q=tmp+3*p; d0[p]=((float)q[0]-m0)/s0; d1[p]=((float)q[1]-m1)/s1; d2[p]=((float)q[2]-m2)/s2;}}
 printf("%.1f ms per batch\n",(now()-t)*1e3);} return 0;}
