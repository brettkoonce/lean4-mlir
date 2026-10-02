// tier2_run.c — tier 2's device side: one verified_mlir/ artifact, one call, through OUR shim.
//
// scripts/platform/tier2.py writes the inputs and a spec, runs this, and compares what comes
// back with the XLA:CPU golden. Kept in C so the platform under test needs no JAX: the path
// exercised is the one every trainer takes (iree_ffi_session_create → iree_ffi_invoke_f32).
//
//   tier2_run <artifact.mlir> <module.fn> <spec.txt> <inputs.bin> <outputs.bin>
//
// spec.txt:  "<n_in> <n_out>", then one "<rank> <d0> <d1> …" line per input, then one
//            "<elements>" line per output. inputs.bin / outputs.bin: the tensors' f32 data,
//            concatenated in signature order.
//
// build:  gcc -O2 -Iffi scripts/platform/tier2_run.c -L<out> -lpjrt_ffi -ldl -Wl,-rpath,<out>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "iree_ffi.h"

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
  if (argc != 6) {
    fprintf(stderr, "usage: tier2_run <artifact.mlir> <module.fn> <spec> <inputs.bin> <outputs.bin>\n");
    return 2;
  }
  FILE* sp = fopen(argv[3], "r");
  if (!sp) { perror(argv[3]); return 1; }
  int n_in, n_out;
  if (fscanf(sp, "%d %d", &n_in, &n_out) != 2) { fprintf(stderr, "FAIL bad spec\n"); return 1; }
  int32_t* ranks = calloc(n_in, sizeof(int32_t));
  int64_t* dims = calloc((size_t)n_in * 8, sizeof(int64_t));
  int64_t* in_n = calloc(n_in, sizeof(int64_t));
  int64_t* totals = calloc(n_out, sizeof(int64_t));
  size_t nd = 0, in_total = 0, out_total = 0;
  for (int i = 0; i < n_in; i++) {
    if (fscanf(sp, "%d", &ranks[i]) != 1 || ranks[i] > 8) { fprintf(stderr, "FAIL bad spec rank\n"); return 1; }
    in_n[i] = 1;
    for (int r = 0; r < ranks[i]; r++) {
      if (fscanf(sp, "%ld", &dims[nd]) != 1) { fprintf(stderr, "FAIL bad spec dims\n"); return 1; }
      in_n[i] *= dims[nd++];
    }
    in_total += in_n[i];
  }
  for (int o = 0; o < n_out; o++) {
    if (fscanf(sp, "%ld", &totals[o]) != 1) { fprintf(stderr, "FAIL bad spec outputs\n"); return 1; }
    out_total += totals[o];
  }
  fclose(sp);

  float* in_buf = malloc(in_total * sizeof(float));
  float* out_buf = calloc(out_total, sizeof(float));
  FILE* fi = fopen(argv[4], "rb");
  if (!fi || fread(in_buf, sizeof(float), in_total, fi) != in_total) {
    fprintf(stderr, "FAIL short read of %s (want %zu floats)\n", argv[4], in_total);
    return 1;
  }
  fclose(fi);
  const float** in = malloc(n_in * sizeof(float*));
  float** outs = malloc(n_out * sizeof(float*));
  for (int i = 0, off = 0; i < n_in; off += in_n[i], i++) in[i] = in_buf + off;
  for (int o = 0, off = 0; o < n_out; off += totals[o], o++) outs[o] = out_buf + off;

  double t0 = now_ms();
  iree_ffi_session_t* s = iree_ffi_session_create(argv[1]);
  if (!s) { fprintf(stderr, "FAIL session_create\n"); return 1; }
  double t1 = now_ms();
  int rc = iree_ffi_invoke_f32(s, argv[2], n_in, ranks, dims, in, n_out, totals, outs);
  double t2 = now_ms();
  iree_ffi_session_release(s);
  if (rc) { fprintf(stderr, "FAIL invoke rc=%d\n", rc); return 1; }

  FILE* fo = fopen(argv[5], "wb");
  if (!fo || fwrite(out_buf, sizeof(float), out_total, fo) != out_total) { perror(argv[5]); return 1; }
  fclose(fo);
  printf("compile %.0f ms, first call %.0f ms\n", t1 - t0, t2 - t1);
  return 0;
}
