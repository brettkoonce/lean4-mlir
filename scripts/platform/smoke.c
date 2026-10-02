// smoke.c — tier 0's one compile + execute, through OUR shim (libpjrt_ffi.so) rather than
// the bare plugin: [1,2,3,4] + [10,20,30,40] must come back as [11,22,33,44]. Probe passing
// and this failing puts the fault in the shim's client setup (allocator options, compile
// options blob), not in the plugin.
//
// build:  gcc -O2 -Iffi scripts/platform/smoke.c -L<out> -lpjrt_ffi -ldl -Wl,-rpath,<out>
// run:    smoke scripts/platform/fixtures/add.mlir
#include <stdint.h>
#include <stdio.h>

#include "iree_ffi.h"

int main(int argc, char** argv) {
  if (argc != 2) { fprintf(stderr, "usage: smoke <add.mlir>\n"); return 2; }
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
