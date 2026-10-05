// The kernel as it goes into ffi/f32_helpers.c: generic + AVX2 clone, runtime dispatch.
// Checks the two produce identical bits on every byte value x channel, then times both.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#define U8NORM_BODY \
    float* d0 = dst; float* d1 = dst + hw; float* d2 = dst + 2 * hw; \
    const float m0 = ms[0], m1 = ms[1], m2 = ms[2], s0 = ms[3], s1 = ms[4], s2 = ms[5]; \
    for (size_t p = 0; p < hw; p++) { const uint8_t* q = src + 3 * p; \
        d0[p] = ((float)q[0] - m0) / s0; d1[p] = ((float)q[1] - m1) / s1; d2[p] = ((float)q[2] - m2) / s2; }
static void u8norm_generic(const uint8_t* src, float* dst, size_t hw, const float* ms) { U8NORM_BODY }
#if defined(__x86_64__)
__attribute__((target("avx2"))) static void u8norm_avx2(const uint8_t* src, float* dst, size_t hw, const float* ms) { U8NORM_BODY }
#endif
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  float ms[6]={123.675f,116.28f,103.53f,58.395f,57.12f,57.375f};
  size_t hw=224*224; uint8_t*src=malloc(3*hw); float*a=malloc(12*hw),*b=malloc(12*hw);
  for(size_t i=0;i<3*hw;i++) src[i]=(uint8_t)(i*131+7);   // every byte value in every channel
  u8norm_generic(src,a,hw,ms); u8norm_avx2(src,b,hw,ms);
  printf("bit-identical generic vs avx2: %s\n", memcmp(a,b,12*hw)?"NO":"yes");
  size_t B=512; float*out=malloc(12*hw*B); uint8_t*in=malloc(3*hw*B); memset(in,77,3*hw*B); memset(out,0,12*hw*B);
  for(int k=0;k<2;k++){ double t=now(); for(size_t i=0;i<B;i++) u8norm_generic(in+i*3*hw,out+i*3*hw,hw,ms);
    double g=now()-t; t=now(); for(size_t i=0;i<B;i++) u8norm_avx2(in+i*3*hw,out+i*3*hw,hw,ms);
    printf("generic %.1f ms, avx2 %.1f ms per 512\n", g*1e3,(now()-t)*1e3); }
  return 0; }
