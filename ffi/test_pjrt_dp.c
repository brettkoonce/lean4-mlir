// test_pjrt_dp.c — known-answer test for the shim's DATA-PARALLEL invoke.
//
// The caller hands in ONE logical batch of 4 and marks it sharded; replica 0
// must see [1,2], replica 1 [10,20], and the in-graph all_reduce must leave both
// holding [11,22]. Reading back replica 0 must therefore give [11,22] — if the
// sharding were wrong we would see [2,4], and if the collective were missing,
// [1,2].
#include <stdio.h>
#include <stdint.h>
#include "iree_ffi.h"

int pjrt_ffi_invoke_f32_dp(
    iree_ffi_session_t*, const char*, int, int,
    const int32_t*, const int64_t*, const float* const*,
    const unsigned char*, int, const int64_t*, float* const*);

int main(int argc, char** argv) {
  iree_ffi_session_t* s = iree_ffi_session_create(argv[1]);
  if (!s) { fprintf(stderr, "session_create failed\n"); return 1; }

  float x[4] = {1.f, 2.f, 10.f, 20.f};       // one logical batch of 4
  float got[2] = {0.f, 0.f};

  int32_t ranks[1]        = {1};
  int64_t dims[1]         = {4};             // FULL batch; the shim splits it
  const float* data[1]    = {x};
  unsigned char shard[1]  = {1};             // <- sharded across replicas
  int64_t totals[1]       = {2};             // per-replica output size
  float* outs[1]          = {got};

  int rc = pjrt_ffi_invoke_f32_dp(s, "m.dp_shard", 2, 1, ranks, dims, data,
                                  shard, 1, totals, outs);
  if (rc) { fprintf(stderr, "dp invoke failed rc=%d\n", rc); return 1; }

  printf("replica 0 got: [%.1f %.1f]   expected [11.0 22.0]\n", got[0], got[1]);
  int ok = (got[0] == 11.f && got[1] == 22.f);
  if (!ok) {
    if (got[0] == 2.f)  printf("  -> looks like NO sharding (both replicas got [1,2])\n");
    if (got[0] == 1.f)  printf("  -> looks like NO collective (replica 0's own slice)\n");
  }
  printf("%s\n", ok ? "DP INVOKE CORRECT: sharding + all_reduce + replica-0 readback"
                    : "*** DP INVOKE WRONG ***");
  iree_ffi_session_release(s);
  return !ok;
}
