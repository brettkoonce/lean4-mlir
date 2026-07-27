// test_pjrt_guards.c — negative tests for the pjrt_ffi.c guards.
//
// `planning/xla_pjrt_ladder.md` §3 exists because a driver that silently dropped
// 295 of 513 outputs ran at full speed and looked exactly like success. A guard
// that has never been observed to fire is not known to work, so each guard here
// is deliberately violated and must be rejected.
//
// build & run:
//   gcc -O2 -Iffi ffi/test_pjrt_guards.c -Lffi -lpjrt_ffi -ldl \
//       -Wl,-rpath,./ffi -o /tmp/test_pjrt_guards && /tmp/test_pjrt_guards
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "iree_ffi.h"

#define B 128
#define IN 784
#define OUT 10

static int failures = 0;

static void expect_reject(int rc, const char* what) {
  if (rc == 0) {
    printf("  ✗ %s — WAS ACCEPTED (guard did not fire)\n", what);
    failures++;
  } else {
    printf("  ✓ %s — rejected (rc=%d)\n", what, rc);
  }
}

static void expect_accept(int rc, const char* what) {
  if (rc != 0) {
    printf("  ✗ %s — was rejected (rc=%d), expected success\n", what, rc);
    failures++;
  } else {
    printf("  ✓ %s — accepted\n", what);
  }
}

int main(void) {
  static float x[B * IN], W0[IN * OUT], b0[OUT], onehot[B * OUT];
  static float outW[IN * OUT], outB[OUT];
  memset(x, 0, sizeof(x));
  memset(W0, 0, sizeof(W0));
  memset(b0, 0, sizeof(b0));
  memset(onehot, 0, sizeof(onehot));

  iree_ffi_session_t* s =
      iree_ffi_session_create("verified_mlir/linear_train_step.mlir");
  if (!s) { fprintf(stderr, "session_create failed\n"); return 1; }

  int32_t ranks[4] = {2, 2, 1, 2};
  int64_t dims[7] = {B, IN, IN, OUT, OUT, B, OUT};
  const float* in[4] = {x, W0, b0, onehot};
  int64_t totals2[2] = {(int64_t)IN * OUT, OUT};
  float* outs2[2] = {outW, outB};

  printf("baseline (should succeed):\n");
  expect_accept(iree_ffi_invoke_f32(s, "m.linear_train_step", 4, ranks, dims, in,
                                    2, totals2, outs2),
                "correct call");

  printf("\nG4 — no dropped state:\n");
  {
    // The graph returns 2 tensors; ask for only 1. Under the old "continue past
    // an unrecognised output" behaviour this silently discards the bias update.
    int64_t totals1[1] = {(int64_t)IN * OUT};
    float* outs1[1] = {outW};
    expect_reject(iree_ffi_invoke_f32(s, "m.linear_train_step", 4, ranks, dims, in,
                                      1, totals1, outs1),
                  "asking for 1 of 2 outputs");
  }

  printf("\nentry-name check:\n");
  expect_reject(iree_ffi_invoke_f32(s, "m.linear_fwd", 4, ranks, dims, in,
                                    2, totals2, outs2),
                "calling a graph this session does not hold");

  printf("\noutput shape check:\n");
  {
    // Claim b0 is 20 elements instead of 10 — a stride error of exactly the kind
    // that reads garbage when unchecked.
    int64_t bad[2] = {(int64_t)IN * OUT, 20};
    expect_reject(iree_ffi_invoke_f32(s, "m.linear_train_step", 4, ranks, dims, in,
                                      2, bad, outs2),
                  "wrong declared output size");
  }

  iree_ffi_session_release(s);
  printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all guards fire",
         failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
