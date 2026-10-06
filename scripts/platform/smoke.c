// smoke.c — tier 0's one compile + execute, through OUR shim (libpjrt_ffi.so) rather than
// the bare plugin: [1,2,3,4] + [10,20,30,40] must come back as [11,22,33,44]. Probe passing
// and this failing puts the fault in the shim's client setup (allocator options, compile
// options blob), not in the plugin.
//
//
// `smoke --erfc <erfc.mlir>` is the second tier-0 question: does this plugin lower `chlo.erfc`?
// StableHLO has no error function; the exact GELU of the ViT and ConvNeXt `…erf…` renders is
// `(x/2) · erfc(−x/√2)` through that one CHLO op, and a plugin without it fails every one of them
// at compile. Eight known answers, to 1e-5 relative (the tail at x = 4 included, where a
// `1 − erf` lowering returns 0).
//
// build:  gcc -O2 -Iffi scripts/platform/smoke.c -L<out> -lpjrt_ffi -ldl -Wl,-rpath,<out>
// run:    smoke scripts/platform/fixtures/add.mlir
//         smoke --erfc scripts/platform/fixtures/erfc.mlir
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "iree_ffi.h"

static int erfc_smoke(const char* path) {
  iree_ffi_session_t* s = iree_ffi_session_create(path);
  if (!s) { fprintf(stderr, "FAIL session_create (chlo.erfc did not compile)\n"); return 1; }
  float x[8] = {-3, -1, -0.5f, 0, 0.5f, 1, 2, 4}, got[8] = {0};
  const double want[8] = {1.9999779095030015, 1.842700792949715, 1.5204998778130465, 1.0,
                          0.4795001221869535, 0.15729920705028513, 0.004677734981047266,
                          1.541725790028002e-08};
  int32_t ranks[1] = {1};
  int64_t dims[1] = {8};
  const float* in[1] = {x};
  int64_t totals[1] = {8};
  float* outs[1] = {got};
  int rc = iree_ffi_invoke_f32(s, "m.erfc", 1, ranks, dims, in, 1, totals, outs);
  iree_ffi_session_release(s);
  if (rc) { fprintf(stderr, "FAIL invoke rc=%d\n", rc); return 1; }
  double worst = 0;
  for (int i = 0; i < 8; i++) {
    double rel = fabs((double)got[i] - want[i]) / want[i];
    if (rel > worst) worst = rel;
    printf("erfc(%4.1f) = %.9g, expected %.9g\n", x[i], got[i], want[i]);
  }
  printf("max relative error %.2e\n", worst);
  if (!(worst <= 1e-5)) { printf("FAIL wrong answer\n"); return 1; }
  printf("chlo.erfc OK\n");
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 3 && !strcmp(argv[1], "--erfc")) return erfc_smoke(argv[2]);
  if (argc != 2) { fprintf(stderr, "usage: smoke <add.mlir> | smoke --erfc <erfc.mlir>\n"); return 2; }
  iree_ffi_session_t* s = iree_ffi_session_create(argv[1]);
  if (!s) { fprintf(stderr, "FAIL session_create\n"); return 1; }

  float x[4] = {1, 2, 3, 4}, y[4] = {10, 20, 30, 40}, got[4] = {0};
  int32_t ranks[2] = {1, 1};
  int64_t dims[2] = {4, 4};
  const float* in[2] = {x, y};
  int64_t totals[1] = {4};
  float* outs[1] = {got};
  int rc = iree_ffi_invoke_f32(s, "m.add", 2, ranks, dims, in, 1, totals, outs);
  iree_ffi_session_release(s);
  if (rc) { fprintf(stderr, "FAIL invoke rc=%d\n", rc); return 1; }

  printf("got [%.1f %.1f %.1f %.1f], expected [11.0 22.0 33.0 44.0]\n",
         got[0], got[1], got[2], got[3]);
  for (int i = 0; i < 4; i++)
    if (got[i] != 11.0f * (i + 1)) { printf("FAIL wrong answer\n"); return 1; }
  printf("compile + execute OK\n");
  return 0;
}
