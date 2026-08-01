// pjrt_ffi.c — XLA backend for the trainer FFI.
//
// Implements the SAME C interface as `iree_ffi.c` (see `iree_ffi.h`), but backed
// by XLA through the PJRT C API instead of the IREE runtime. Because the
// interface is identical, `iree_lean_ffi.c` and every Lean trainer above it are
// unchanged — you pick a backend by choosing which shim gets linked.
//
// What differs from the IREE shim, and it is only this: a session is created
// from a **`.mlir` source file** (the Lean-emitted StableHLO in `verified_mlir/`)
// rather than a precompiled `.vmfb`. There is no separate `iree-compile` step;
// XLA compiles the StableHLO in-process. The only edit made to the Lean output
// is renaming the entry function to `@main`, which PJRT requires.
//
// The plugin is loaded with dlopen at run time — no link-time dependency on XLA,
// JAX, or Python. Override the path with $PJRT_PLUGIN.
//
// See planning/xla_pjrt_ladder.md for the ladder this is rung 0 of, and §3 there
// for the four gates. The G4 ("no dropped state") assertion that doc asks for is
// implemented in `iree_ffi_invoke_f32` below.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <time.h>

#include "pjrt_c_api.h"
#include "pjrt_compile_options.h"
#include "iree_ffi.h"

// ─── default plugin search path ────────────────────────────────────────────
// $PJRT_PLUGIN always wins. Absent it, try each candidate in order and take the
// first that dlopens. This used to be ONE hardcoded absolute path to a ROCm
// plugin in a checkout that no longer exists, which meant every non-AMD box —
// and, after the repo moved, the AMD one too — died in dlopen with a path that
// told you nothing about what it wanted. The list is vendor-symmetric on
// purpose: the shim is plugin-agnostic and there is no reason for the DEFAULT
// to pick a side. Relative entries resolve against the CWD, which for every
// `lake run`/`lake exe` path is the repo root, so the in-repo `.venv` is found
// without anyone exporting anything.
//
// Ordering rationale: repo-local venv first (it is the pinned env — see
// requirements-cuda-lock.txt), then the historical absolute paths, so a box
// that was working before keeps working.
static const char* const kDefaultPlugins[] = {
  ".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so",
  ".venv/lib/python3.12/site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so",
  "/home/skoonce/lean/claude_max/lean4-jax/.venv/lib/python3.12/"
  "site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so",
};
#define N_DEFAULT_PLUGINS (sizeof(kDefaultPlugins)/sizeof(kDefaultPlugins[0]))

// ─── process-global PJRT client ────────────────────────────────────────────
// One client per process, refcounted across sessions. This is not an
// optimization: each StreamExecutor GPU client reserves ~19 GB for its BFC
// allocator, so two clients on one device fight over memory. The trainers make
// two sessions (train step + forward), which must share.

static const PJRT_Api* g_api = NULL;
static PJRT_Client* g_client = NULL;
static PJRT_Device* g_device = NULL;
static int g_client_refs = 0;
// $PJRT_REPLICAS: how many devices this process drives. 1 = single-device, the
// default and the only mode the IREE-compatible entry points support. >1 selects
// matching num_replicas compile options and enables pjrt_ffi_invoke_f32_dp.
static int g_replicas = 1;
static PJRT_Device* g_devices[8];

struct iree_ffi_session_t {
  PJRT_LoadedExecutable* exe;
  char* entry;      // original func name, before the @main rename
  int num_outputs;  // from the compiled executable — used by the G4 guard
  int replicas;     // what THIS graph was compiled for (see session_create)
};

// Presence marker. `lean_iree_backend_name` in iree_lean_ffi.c dlsym()s this to
// tell which shim got linked, so the Lean driver knows whether to run
// `iree-compile` first or hand the .mlir straight to XLA. Detecting it beats an
// env var: the binary cannot disagree with itself.
void pjrt_ffi_marker(void) {}

static int trace_enabled(void) {
  static int t = -1;
  if (t < 0) { const char* e = getenv("PJRT_FFI_TRACE"); t = (e && atoi(e)) ? 1 : 0; }
  return t;
}

// ─── transfer accounting (handoff §2d.3) ───────────────────────────────────
//
// OPT-IN via $PJRT_FFI_TIMING=<report interval in steps>; zero cost and zero
// behaviour change when unset, which is the point — every cross-backend gate in
// the repo depends on IREE and XLA running the same code path, so this must not
// become a second path.
//
// What it answers: how much of a step is the `[theta|m|v]` host<->device round
// trip, SPECIFICALLY, as opposed to the whole step. Device-resident parameters
// remove exactly the param share and nothing else, so that share is the ceiling
// on what the ~3-4 session job can buy.
//
// Both transfer phases issue every buffer before awaiting any (deliberately —
// see the h2d comment in the invoke), so `issue` and `await` are timed
// separately: if issue dominates, the per-input split is exact; if await
// dominates, the split is by bytes over a shared link and is reported as such.
//
// Input classification. The train-step call sites in `iree_lean_ffi.c` lay the
// inputs out as x, params..., onehot; the DP call site says so exactly via
// `shard_mask` (replicated == parameter). Single-device has no mask, so the
// first and last inputs are bucketed separately from the middle rather than
// guessed at — the forward session has NO onehot, so folding its tail into
// "data" would silently misreport one param tensor. Run with
// LEAN_MLIR_SKIP_EVAL=1 and only the train step is timed at all.

static int timing_interval(void) {
  static int n = -1;
  if (n < 0) { const char* e = getenv("PJRT_FFI_TIMING"); n = (e && atoi(e) > 0) ? atoi(e) : 0; }
  return n;
}

// ─── pinned d2h staging (§2d.3, the bandwidth leg) ─────────────────────────
//
// OPT-IN via $PJRT_FFI_PINNED=1. Measured: d2h moves bytes at ~11 GB/s while
// h2d hits ~26 GB/s (PCIe 4.0 x16 line rate) on identical buffers. The standard
// explanation is that a d2h into PAGEABLE host memory — and these destinations
// are Lean-owned `ByteArray` slices — cannot be DMA'd into directly, so the
// runtime bounces through a pinned staging buffer and then memcpys out. Serial,
// that predicts 1/(1/26 + 1/15.5) = 9.7 GB/s against the 11.0 measured, where
// 15.5 GB/s is this host's single-threaded memcpy at 260 MB.
//
// ⚠⚠ MEASURED 2026-07-30, AND THE HYPOTHESIS ABOVE IS **REFUTED**. Keep this
// flag OFF. DMA-ing into an arena that is definitely pinned is **no faster**
// than into Lean's pageable ByteArray — R34 bs32, three runs each: pageable
// 62.3 / 60.7 / 61.6 ms against pinned 59.3 / 64.2 / 62.0 ms, overlapping
// distributions. If a staging bounce were the mechanism, the pinned DMA should
// have hit line rate (~10 ms for 260 MB) and it does not move at all. So d2h's
// ~11 GB/s marginal bandwidth is simply what this path costs; it is NOT a
// pageable-destination artefact, and pinning does not unlock it.
//
// Turning it on is a **17% REGRESSION** (161 -> 189 ms/step): the explicit
// memcpy out of the arena costs ~30.5 ms and buys no DMA gain. The route to
// d2h is therefore to have FEWER BUFFERS, not faster ones — i.e. device
// residency (§2d.3), which deletes them.
//
// It is kept, opt-in and off by default, as the falsification instrument: this
// is a hardware/driver-specific negative, and on a box with a different PCIe or
// ROCm setup pinning might well help. Re-checking costs two minutes with
// PJRT_FFI_PINNED=1 PJRT_FFI_TIMING=10 and beats re-deriving the argument.
//
// PJRT exposes no pinned-host allocator, so `hipHostMalloc` is dlsym'd out of
// the ROCm runtime — the same dlopen-don't-link discipline the plugin uses.

static void* (*g_hip_host_malloc)(void**, size_t, unsigned int) = NULL;
static void* g_pin = NULL;
static size_t g_pin_sz = 0;

static int pinned_enabled(void) {
  static int t = -1;
  if (t < 0) { const char* e = getenv("PJRT_FFI_PINNED"); t = (e && atoi(e)) ? 1 : 0; }
  return t;
}

// ─── fault injection for the phase-3 gate ──────────────────────────────────
//
// $PJRT_FFI_FAULT=1 flips the low mantissa bit of ONE returned float — the
// smallest possible transport fault, 1 ULP in 68 million values.
//
// This exists because `scripts/residency_gate.sh`'s other control (a perturbed
// initialisation) proves only that the harness can see *a* difference; it does
// not prove the harness can see a *transport* difference, which is the entire
// thing the gate is for. Handoff §4: a tie that is bit-exact everywhere is
// indistinguishable from a harness comparing a buffer with itself, and the way
// out is a control that perturbs the specific mechanism under test.
//
// It is deliberately the WEAKEST fault that is still a fault. A gate that
// catches one flipped mantissa bit will catch a dropped buffer, a stale
// retained handle, or an off-by-one replica offset — the plausible ways device
// residency goes wrong.
static int fault_enabled(void) {
  static int t = -1;
  if (t < 0) { const char* e = getenv("PJRT_FFI_FAULT"); t = (e && atoi(e)) ? 1 : 0; }
  return t;
}

// Returns a pinned arena of at least `n` bytes, or NULL if HIP is unavailable
// (in which case the caller silently stays on the direct path — this is an
// optimisation, never a correctness requirement).
static void* pin_arena(size_t n) {
  if (g_pin && g_pin_sz >= n) return g_pin;
  if (!g_hip_host_malloc) {
    // Try ROCm first, then CUDA. `cudaHostAlloc` has the SAME (void**, size_t, unsigned)
    // signature and the same success==0 convention as `hipHostMalloc`, so one function
    // pointer serves both and the call site below needs no branch. Without the CUDA leg
    // $PJRT_FFI_PINNED was a silent no-op on NVIDIA — it printed "no libamdhip64" and fell
    // back to the direct path, so the flag looked supported and did nothing.
    void* h = dlopen("libamdhip64.so.7", RTLD_LAZY);
    if (!h) h = dlopen("libamdhip64.so", RTLD_LAZY);
    if (h) *(void**)&g_hip_host_malloc = dlsym(h, "hipHostMalloc");
    if (!g_hip_host_malloc) {
      void* c = dlopen("libcudart.so.12", RTLD_LAZY);
      if (!c) c = dlopen("libcudart.so", RTLD_LAZY);
      if (c) *(void**)&g_hip_host_malloc = dlsym(c, "cudaHostAlloc");
    }
    if (!g_hip_host_malloc) {
      fprintf(stderr, "[pjrt_ffi] PJRT_FFI_PINNED: no hipHostMalloc (libamdhip64) and no "
                      "cudaHostAlloc (libcudart) — staying on the direct path\n");
      return NULL;
    }
  }
  void* p = NULL;
  if (g_hip_host_malloc(&p, n, 0u) != NULL || !p) {   // hipSuccess == 0
    fprintf(stderr, "[pjrt_ffi] PJRT_FFI_PINNED: hipHostMalloc(%zu) failed\n", n);
    return NULL;
  }
  g_pin = p; g_pin_sz = n;
  fprintf(stderr, "[pjrt_ffi] pinned d2h arena: %.1f MB\n", n / 1048576.0);
  return g_pin;
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

typedef struct {
  char entry[64];
  long long calls;      // calls in the CURRENT window (zeroed at each report)
  long long done;       // calls before the current window started
  int window;
  double h2d_issue_param, h2d_issue_head, h2d_issue_tail, h2d_await;
  double exec, exec_await;
  double d2h_issue, d2h_await, d2h_copy;
  double h2d_param_mb, h2d_head_mb, h2d_tail_mb, d2h_mb;
} tacct_t;

static tacct_t g_acct[4];
static int g_acct_n = 0;

static tacct_t* acct_for(const char* entry) {
  for (int i = 0; i < g_acct_n; i++)
    if (strcmp(g_acct[i].entry, entry) == 0) return &g_acct[i];
  if (g_acct_n == (int)(sizeof(g_acct) / sizeof(g_acct[0]))) return &g_acct[0];
  tacct_t* a = &g_acct[g_acct_n++];
  memset(a, 0, sizeof(*a));
  snprintf(a->entry, sizeof(a->entry), "%s", entry);
  return a;
}

// Windowed, NOT cumulative: each report covers only the steps since the last
// one, then the window resets. A cumulative average would fold the ~2 s
// first-compile and the cold-cache warmup steps into every later number, which
// is the wall-clock-minus-compile mistake one level down.
static void acct_report(tacct_t* a) {
  double n = (double)a->calls;
  if (n <= 0) return;
  double h2d_param = a->h2d_issue_param, h2d_data = a->h2d_issue_head + a->h2d_issue_tail;
  double h2d_issue = h2d_param + h2d_data;
  // Attribute the batched await to the two buckets by bytes — the transfers
  // share one link, so bytes is the right first-order split. Reported alongside
  // the raw issue/await times so the attribution is visible, not implied.
  double mb_total = a->h2d_param_mb + a->h2d_head_mb + a->h2d_tail_mb;
  double param_frac = mb_total > 0 ? a->h2d_param_mb / mb_total : 0;
  double param_h2d = h2d_param + a->h2d_await * param_frac;
  double step = (h2d_issue + a->h2d_await + a->exec + a->exec_await
                 + a->d2h_issue + a->d2h_await + a->d2h_copy) / n;
  double param_rt = (param_h2d + a->d2h_issue + a->d2h_await + a->d2h_copy) / n;
  fprintf(stderr,
      "[pjrt_ffi:timing] @%s  window %d, steps %lld..%lld  step=%.1f ms\n"
      "    h2d   issue param %.1f / head %.1f / tail %.1f ms   await %.1f ms   "
      "(%.0f + %.0f + %.0f MB)\n"
      "    COMPUTE %.1f ms (launch %.1f + device %.1f)\n"
      // ⚠ the GB/s here is bytes/await, so it CONFLATES the per-buffer term with
      // the byte term and reads far below the marginal bandwidth. Effective, not
      // marginal — separate the two by fitting across nets with different
      // buffer-count/byte ratios, as §2d.3 does.
      "    d2h   issue %.1f ms  await(DMA) %.1f ms = %.1f GB/s eff   memcpy %.1f ms   (%.0f MB)\n"
      "    >> PARAM round trip %.1f ms = %.1f%% of the step "
      "(h2d %.1f + d2h %.1f), link %.1f GB/s\n",
      a->entry, ++a->window, a->done + 1, a->done + a->calls, step,
      h2d_param / n, a->h2d_issue_head / n, a->h2d_issue_tail / n, a->h2d_await / n,
      a->h2d_param_mb / n, a->h2d_head_mb / n, a->h2d_tail_mb / n,
      (a->exec + a->exec_await) / n, a->exec / n, a->exec_await / n,
      a->d2h_issue / n, a->d2h_await / n,
      a->d2h_await > 0 ? (a->d2h_mb / a->d2h_await) * 1.048576 : 0.0,
      a->d2h_copy / n, a->d2h_mb / n,
      param_rt, step > 0 ? 100.0 * param_rt / step : 0.0,
      param_h2d / n, (a->d2h_issue + a->d2h_await + a->d2h_copy) / n,
      // MiB per ms -> GB/s
      param_rt > 0 ? ((a->h2d_param_mb + a->d2h_mb) / n / param_rt) * 1.048576 : 0.0);

  char keep[64];
  memcpy(keep, a->entry, sizeof(keep));
  int w = a->window;
  long long d = a->done + a->calls;
  memset(a, 0, sizeof(*a));
  memcpy(a->entry, keep, sizeof(keep));
  a->window = w;
  a->done = d;
}

static void acct_report_all(void) {
  for (int i = 0; i < g_acct_n; i++) acct_report(&g_acct[i]);
}

// One-shot dump of the per-buffer sizes, so the bucketing above is verifiable
// against the graph rather than assumed.
static void acct_layout_once(const char* entry, int n_inputs, const int64_t* dims,
                             const int32_t* ranks, int n_outputs,
                             const int64_t* out_totals) {
  static int done = 0;
  if (done) return;
  done = 1;
  fprintf(stderr, "[pjrt_ffi:timing] @%s buffer layout — %d in, %d out\n",
          entry, n_inputs, n_outputs);
  int off = 0;
  for (int i = 0; i < n_inputs; i++) {
    int64_t e = 1;
    for (int k = 0; k < ranks[i]; k++) e *= dims[off + k];
    off += ranks[i];
    if (i < 3 || i >= n_inputs - 2)
      fprintf(stderr, "    in[%d] rank %d  %lld elems  %.2f MB%s\n", i, ranks[i],
              (long long)e, e * 4.0 / 1048576.0,
              i == 0 ? "   <- head" : (i == n_inputs - 1 ? "   <- tail" : ""));
    else if (i == 3)
      fprintf(stderr, "    ... %d middle inputs (bucketed as params)\n", n_inputs - 5);
  }
  int64_t ot = 0;
  for (int i = 0; i < n_outputs; i++) ot += out_totals[i];
  fprintf(stderr, "    out: %d buffers, %lld elems, %.2f MB total\n",
          n_outputs, (long long)ot, ot * 4.0 / 1048576.0);
}

// Report and consume a PJRT_Error. Returns 1 if `err` was non-NULL.
static int check(PJRT_Error* err, const char* what) {
  if (!err) return 0;
  PJRT_Error_Message_Args m = {0};
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = err;
  g_api->PJRT_Error_Message(&m);
  fprintf(stderr, "[pjrt_ffi] %s: %.*s\n", what, (int)m.message_size, m.message);
  PJRT_Error_Destroy_Args d = {0};
  d.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  d.error = err;
  g_api->PJRT_Error_Destroy(&d);
  return 1;
}

static int await_event(PJRT_Event* ev, const char* what) {
  PJRT_Event_Await_Args a = {0};
  a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  a.event = ev;
  int rc = check(g_api->PJRT_Event_Await(&a), what);
  PJRT_Event_Destroy_Args d = {0};
  d.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  d.event = ev;
  g_api->PJRT_Event_Destroy(&d);
  return rc;
}

static int ensure_client(void) {
  if (g_client) { g_client_refs++; return 0; }

  const char* plugin = getenv("PJRT_PLUGIN");
  void* h = NULL;
  if (plugin) {
    // Explicit request: one attempt, and report THAT path on failure. Silently
    // falling back to a default here would be worse than failing — it would run
    // the wrong backend under a name the user chose.
    h = dlopen(plugin, RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
      fprintf(stderr, "[pjrt_ffi] dlopen(%s): %s\n", plugin, dlerror());
      fprintf(stderr, "[pjrt_ffi] $PJRT_PLUGIN is set but did not load.\n");
      return 1;
    }
  } else {
    for (size_t i = 0; i < N_DEFAULT_PLUGINS && !h; i++) {
      h = dlopen(kDefaultPlugins[i], RTLD_LAZY | RTLD_LOCAL);
      if (h) plugin = kDefaultPlugins[i];
    }
    if (!h) {
      fprintf(stderr, "[pjrt_ffi] no PJRT plugin found. Tried, in order:\n");
      for (size_t i = 0; i < N_DEFAULT_PLUGINS; i++)
        fprintf(stderr, "[pjrt_ffi]   %s\n", kDefaultPlugins[i]);
      fprintf(stderr,
              "[pjrt_ffi] set $PJRT_PLUGIN to your plugin .so (the jax cuda/rocm\n"
              "[pjrt_ffi] plugin ships one under site-packages/jax_plugins/).\n");
      return 1;
    }
    if (trace_enabled())
      fprintf(stderr, "[pjrt_ffi] plugin (default search): %s\n", plugin);
  }
  const PJRT_Api* (*GetPjrtApi)(void) =
      (const PJRT_Api* (*)(void))dlsym(h, "GetPjrtApi");
  if (!GetPjrtApi) {
    fprintf(stderr, "[pjrt_ffi] dlsym(GetPjrtApi): %s\n", dlerror());
    return 1;
  }
  g_api = GetPjrtApi();

  PJRT_Client_Create_Args ca = {0};
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (check(g_api->PJRT_Client_Create(&ca), "Client_Create")) return 1;
  g_client = ca.client;

  PJRT_Client_AddressableDevices_Args da = {0};
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client = g_client;
  if (check(g_api->PJRT_Client_AddressableDevices(&da), "AddressableDevices")) return 1;
  if (da.num_addressable_devices == 0) {
    fprintf(stderr, "[pjrt_ffi] no addressable devices\n");
    return 1;
  }
  g_device = da.addressable_devices[0];
  g_client_refs = 1;

  {
    const char* e = getenv("PJRT_REPLICAS");
    g_replicas = e ? atoi(e) : 1;
    if (g_replicas < 1) g_replicas = 1;
    if ((size_t)g_replicas > da.num_addressable_devices) {
      fprintf(stderr,
              "[pjrt_ffi] PJRT_REPLICAS=%d but only %zu device(s) are addressable "
              "(is CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES restricting them?)\n",
              g_replicas, da.num_addressable_devices);
      return 1;
    }
    if (g_replicas > (int)(sizeof(g_devices)/sizeof(g_devices[0]))) {
      fprintf(stderr, "[pjrt_ffi] PJRT_REPLICAS=%d exceeds the compiled-in max\n",
              g_replicas);
      return 1;
    }
    for (int i = 0; i < g_replicas; i++) g_devices[i] = da.addressable_devices[i];
  }

  fprintf(stderr, "[pjrt_ffi] XLA backend: PJRT %d.%d, %zu device(s)\n",
          g_api->pjrt_api_version.major_version,
          g_api->pjrt_api_version.minor_version,
          da.num_addressable_devices);
  return 0;
}

// ─── session create: compile Lean-emitted StableHLO ────────────────────────

static char* slurp(const char* path, size_t* n) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "[pjrt_ffi] cannot open %s\n", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t got = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if (got != (size_t)sz) { free(buf); return NULL; }
  buf[sz] = 0;
  *n = got;
  return buf;
}

// Rewrite the single `func.func @<name>(` entry to `@main`, returning the new
// source and storing the original name. PJRT requires the entry to be @main;
// keeping the original lets `invoke_f32` verify the caller asked for this graph.
static char* rename_entry_to_main(const char* src, char** entry_out) {
  const char* p = strstr(src, "func.func @");
  if (!p) { fprintf(stderr, "[pjrt_ffi] no 'func.func @' in module\n"); return NULL; }
  const char* name = p + strlen("func.func @");
  const char* end = name;
  while (*end && (*end == '_' || *end == '.' ||
                  (*end >= '0' && *end <= '9') ||
                  (*end >= 'a' && *end <= 'z') ||
                  (*end >= 'A' && *end <= 'Z'))) end++;

  size_t namelen = (size_t)(end - name);
  char* entry = (char*)malloc(namelen + 1);
  memcpy(entry, name, namelen);
  entry[namelen] = 0;
  *entry_out = entry;

  // Refuse multi-function modules: we would silently compile the wrong one.
  if (strstr(end, "func.func @")) {
    fprintf(stderr, "[pjrt_ffi] module has >1 func.func; expected exactly one\n");
    free(entry); *entry_out = NULL;
    return NULL;
  }

  size_t pre = (size_t)(name - src);
  size_t tail = strlen(end);
  char* out = (char*)malloc(pre + 4 + tail + 1);
  memcpy(out, src, pre);
  memcpy(out + pre, "main", 4);
  memcpy(out + pre + 4, end, tail + 1);
  return out;
}

iree_ffi_session_t* iree_ffi_session_create(const char* path) {
  if (ensure_client()) return NULL;

  size_t n = 0;
  char* src = slurp(path, &n);
  if (!src) return NULL;

  char* entry = NULL;
  char* mlir = rename_entry_to_main(src, &entry);
  free(src);
  if (!mlir) return NULL;

  PJRT_Program prog = {0};
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = mlir;
  prog.code_size = strlen(mlir);
  prog.format = "mlir";
  prog.format_size = 4;

  PJRT_Client_Compile_Args cc = {0};
  cc.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  cc.client = g_client;
  cc.program = &prog;
  // Replica count is PER GRAPH, not per process. A module with no cross-replica
  // op computes the same thing at any replica count and is only ever invoked
  // single-device (the eval forward is exactly this), so compile it for one
  // replica — otherwise Execute rejects it with "Attempted to execute with 1
  // argument lists when local device count is 2".
  int reps = (g_replicas > 1 && strstr(mlir, "all_reduce")) ? g_replicas : 1;
  size_t optlen = 0;
  const unsigned char* optbuf = pjrt_compile_options_for(reps, &optlen);
  if (!optbuf) {
    fprintf(stderr, "[pjrt_ffi] no compile options for %d replicas "
                    "(regenerate ffi/pjrt_compile_options.h with that count)\n",
            reps);
    free(mlir); free(entry); return NULL;
  }
  cc.compile_options = (const char*)optbuf;
  cc.compile_options_size = optlen;
  // XLA JITs here — ONCE PER SESSION, i.e. once per process, not per step. The
  // training loop below only calls Execute on the result. IREE amortizes the
  // equivalent cost across runs via a cached .vmfb, so this shows up as a fixed
  // startup tax on every XLA run; at rung 3/4 graph sizes it is worth watching.
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  int bad = check(g_api->PJRT_Client_Compile(&cc), "Client_Compile");
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double compile_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
  free(mlir);
  if (bad) { free(entry); return NULL; }

  // Ask the compiled executable how many outputs it really has. This is what
  // makes the G4 guard in invoke_f32 possible.
  int num_outputs = -1;
  PJRT_LoadedExecutable_GetExecutable_Args ge = {0};
  ge.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ge.loaded_executable = cc.executable;
  if (!check(g_api->PJRT_LoadedExecutable_GetExecutable(&ge), "GetExecutable")) {
    PJRT_Executable_NumOutputs_Args no = {0};
    no.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    no.executable = ge.executable;
    if (!check(g_api->PJRT_Executable_NumOutputs(&no), "NumOutputs"))
      num_outputs = (int)no.num_outputs;
    PJRT_Executable_Destroy_Args ed = {0};
    ed.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
    ed.executable = ge.executable;
    g_api->PJRT_Executable_Destroy(&ed);
  }

  iree_ffi_session_t* s = (iree_ffi_session_t*)calloc(1, sizeof(*s));
  s->exe = cc.executable;
  s->entry = entry;
  s->num_outputs = num_outputs;
  s->replicas = reps;
  fprintf(stderr, "[pjrt_ffi] compiled %s (@%s, %d outputs, %d replica%s) in %.0f ms\n",
          path, entry, num_outputs, reps, reps == 1 ? "" : "s", compile_ms);
  return s;
}

void iree_ffi_session_release(iree_ffi_session_t* sess) {
  if (!sess) return;
  if (sess->exe) {
    PJRT_LoadedExecutable_Destroy_Args a = {0};
    a.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    a.executable = sess->exe;
    check(g_api->PJRT_LoadedExecutable_Destroy(&a), "LoadedExecutable_Destroy");
  }
  free(sess->entry);
  free(sess);
  if (timing_interval()) acct_report_all();
  if (--g_client_refs == 0 && g_client) {
    PJRT_Client_Destroy_Args a = {0};
    a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    a.client = g_client;
    check(g_api->PJRT_Client_Destroy(&a), "Client_Destroy");
    g_client = NULL;
  }
}

// ─── invoke ────────────────────────────────────────────────────────────────

// `fn_name` arrives as "<module>.<func>" (e.g. "m.linear_train_step"). PJRT has
// no notion of calling one function out of a module — the session IS one
// compiled graph — so rather than ignore the name, check it matches. A silent
// mismatch here would run the wrong graph at full speed and look like success.
static int entry_matches(const char* fn_name, const char* entry) {
  const char* dot = strrchr(fn_name, '.');
  const char* want = dot ? dot + 1 : fn_name;
  return strcmp(want, entry) == 0;
}

int iree_ffi_invoke_f32(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,
    const float* const* input_data,
    int n_outputs,
    const int64_t* output_totals,
    float* const* output_data) {

  if (!sess || !sess->exe) return 1;

  if (!entry_matches(fn_name, sess->entry)) {
    fprintf(stderr, "[pjrt_ffi] entry mismatch: session holds @%s, caller asked for '%s'\n",
            sess->entry, fn_name);
    return 1;
  }
  if (sess->replicas != 1) {
    fprintf(stderr,
            "[pjrt_ffi] @%s was compiled for %d replicas; use the data-parallel "
            "invoke, not the single-device one\n", sess->entry, sess->replicas);
    return 1;
  }

  // G4 — no dropped state. Every output the graph produces must have somewhere
  // to go. Discarding an output silently is exactly the failure mode that made a
  // full-speed non-learning R34 loop look identical to success.
  if (sess->num_outputs >= 0 && sess->num_outputs != n_outputs) {
    fprintf(stderr,
            "[pjrt_ffi] G4 VIOLATION: @%s returns %d outputs, caller supplied %d "
            "destinations — refusing to run\n",
            sess->entry, sess->num_outputs, n_outputs);
    return 1;
  }

  PJRT_Buffer** in = (PJRT_Buffer**)calloc((size_t)n_inputs, sizeof(*in));
  PJRT_Buffer** out = (PJRT_Buffer**)calloc((size_t)n_outputs, sizeof(*out));
  int rc = 0;

  const int tint = timing_interval();
  tacct_t* acct = NULL;
  if (tint) {
    acct = acct_for(sess->entry);
    acct_layout_once(sess->entry, n_inputs, input_dims_flat, input_ranks,
                     n_outputs, output_totals);
  }

  // Host → device for every input.
  //
  // Issue ALL transfers first, then await them, rather than awaiting each in
  // turn. At R34 scale this is 513 buffers totalling ~250 MB; awaiting inside the
  // loop serialises them into 513 round-trip latencies and was measured to
  // dominate the step (~76% of XLA's 256 ms/step). Deferring the awaits is safe
  // here: `kImmutableUntilTransferCompletes` only requires the host data to stay
  // alive until the event fires, and every `input_data[i]` is owned by the caller
  // for the whole call.
  {
    PJRT_Event** h2d = (PJRT_Event**)calloc((size_t)n_inputs, sizeof(*h2d));
    int off = 0, failed = 0;
    for (int i = 0; i < n_inputs; i++) {
      PJRT_Client_BufferFromHostBuffer_Args a = {0};
      a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      a.client = g_client;
      a.data = input_data[i];
      a.type = PJRT_Buffer_Type_F32;
      a.dims = &input_dims_flat[off];
      a.num_dims = (size_t)input_ranks[i];
      a.host_buffer_semantics =
          PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
      a.device = g_device;
      double t0 = acct ? now_ms() : 0;
      if (check(g_api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer")) {
        failed = 1; break;
      }
      if (acct) {
        double dt = now_ms() - t0;
        int64_t elems = 1;
        for (int k = 0; k < input_ranks[i]; k++) elems *= input_dims_flat[off + k];
        double mb = elems * 4.0 / 1048576.0;
        if (i == 0)                  { acct->h2d_issue_head  += dt; acct->h2d_head_mb  += mb; }
        else if (i == n_inputs - 1)  { acct->h2d_issue_tail  += dt; acct->h2d_tail_mb  += mb; }
        else                         { acct->h2d_issue_param += dt; acct->h2d_param_mb += mb; }
      }
      in[i] = a.buffer;
      h2d[i] = a.done_with_host_buffer;
      off += input_ranks[i];
    }
    double tw = acct ? now_ms() : 0;
    for (int i = 0; i < n_inputs; i++)
      if (h2d[i] && await_event(h2d[i], "h2d")) failed = 1;
    if (acct) acct->h2d_await += now_ms() - tw;
    free(h2d);
    if (failed) { rc = 2; goto cleanup; }
  }

  {
    PJRT_Buffer* const* arglist[1] = {in};
    PJRT_Buffer** outlist[1] = {out};

    PJRT_ExecuteOptions eo = {0};
    eo.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

    PJRT_LoadedExecutable_Execute_Args ea = {0};
    ea.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable = sess->exe;
    ea.options = &eo;
    ea.argument_lists = arglist;
    ea.num_devices = 1;
    ea.num_args = (size_t)n_inputs;
    ea.output_lists = outlist;
    // Execute is ASYNCHRONOUS: it returns as soon as the work is enqueued, so
    // without a completion event the GPU compute lands inside the d2h await and
    // the transfer share reads ~97% instead of the truth. Ask for the device
    // event and wait on it here, but ONLY when timing — the default path must
    // stay byte-identical. It costs no wall clock: this step's d2h depends on
    // this step's compute, so there was nothing to overlap.
    PJRT_Event* dce = NULL;
    if (acct) ea.device_complete_events = &dce;
    double t0 = acct ? now_ms() : 0;
    if (check(g_api->PJRT_LoadedExecutable_Execute(&ea), "Execute")) { rc = 3; goto cleanup; }
    if (acct) {
      acct->exec += now_ms() - t0;
      double t1 = now_ms();
      if (dce) await_event(dce, "device_complete");
      acct->exec_await += now_ms() - t1;
    }
  }

  // Device → host. Query the required size first and compare: a shape mismatch
  // between the graph and the caller's expectation is caught here rather than
  // silently truncating or reading garbage.
  // Same batching as the h2d path: size-check and issue all copies, then await
  // them together. The size query is synchronous and cheap (no transfer), so it
  // stays inline — it is the guard that catches a graph/caller shape mismatch
  // before any bytes move.
  {
    PJRT_Event** d2h = (PJRT_Event**)calloc((size_t)n_outputs, sizeof(*d2h));
    int failed = 0;

    // Optional pinned staging (§2d.3). One arena sized to the whole output set,
    // allocated once; each output DMAs to its own offset, and the memcpy out
    // happens after ALL the transfers are awaited so the two legs stay separable.
    // If the arena cannot be had, `pin` stays NULL and this is the direct path.
    char* pin = NULL;
    size_t pin_need = 0;
    if (pinned_enabled()) {
      for (int i = 0; i < n_outputs; i++) pin_need += (size_t)output_totals[i] * sizeof(float);
      pin = (char*)pin_arena(pin_need);
    }

    double td = acct ? now_ms() : 0;
    size_t pin_off = 0;
    for (int i = 0; i < n_outputs; i++) {
      size_t want = (size_t)output_totals[i] * sizeof(float);
      if (acct) acct->d2h_mb += want / 1048576.0;

      PJRT_Buffer_ToHostBuffer_Args q = {0};
      q.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      q.src = out[i];
      q.dst = NULL;
      if (check(g_api->PJRT_Buffer_ToHostBuffer(&q), "ToHostBuffer(size query)")) {
        failed = 1; break;
      }
      if (q.dst_size != want) {
        fprintf(stderr,
                "[pjrt_ffi] output %d size mismatch: graph produces %zu bytes, "
                "caller expects %zu — refusing to copy\n", i, q.dst_size, want);
        failed = 1; break;
      }

      PJRT_Buffer_ToHostBuffer_Args a = {0};
      a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      a.src = out[i];
      a.dst = pin ? (void*)(pin + pin_off) : (void*)output_data[i];
      a.dst_size = want;
      if (check(g_api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer")) { failed = 1; break; }
      d2h[i] = a.event;
      pin_off += want;
    }
    if (acct) { acct->d2h_issue += now_ms() - td; td = now_ms(); }
    for (int i = 0; i < n_outputs; i++)
      if (d2h[i] && await_event(d2h[i], "d2h")) failed = 1;
    if (acct) { acct->d2h_await += now_ms() - td; td = now_ms(); }
    if (pin && !failed) {
      size_t off2 = 0;
      for (int i = 0; i < n_outputs; i++) {
        size_t want = (size_t)output_totals[i] * sizeof(float);
        memcpy(output_data[i], pin + off2, want);
        off2 += want;
      }
    }
    if (acct && pin) acct->d2h_copy += now_ms() - td;
    if (fault_enabled() && !failed && n_outputs > 0 && output_totals[0] > 0) {
      union { float f; uint32_t u; } v;
      v.f = output_data[0][0];
      v.u ^= 1u;                       // 1 ULP, the smallest fault that is one
      output_data[0][0] = v.f;
    }
    free(d2h);
    if (failed) { rc = 4; goto cleanup; }
  }

  if (acct && ++acct->calls % tint == 0) acct_report(acct);

  if (trace_enabled())
    fprintf(stderr, "[pjrt_ffi] @%s ok (%d in, %d out)\n", sess->entry, n_inputs, n_outputs);

cleanup:
  for (int i = 0; i < n_inputs; i++)
    if (in[i]) {
      PJRT_Buffer_Destroy_Args d = {0};
      d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      d.buffer = in[i];
      g_api->PJRT_Buffer_Destroy(&d);
    }
  for (int i = 0; i < n_outputs; i++)
    if (out[i]) {
      PJRT_Buffer_Destroy_Args d = {0};
      d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      d.buffer = out[i];
      g_api->PJRT_Buffer_Destroy(&d);
    }
  free(in);
  free(out);
  return rc;
}

// ─── data-parallel invoke ──────────────────────────────────────────────────
//
// Exported ONLY by this shim. `iree_lean_ffi.c` reaches it through a WEAK
// reference, so the IREE build — which has no such symbol — simply never calls
// it, and `iree_ffi.h` stays unchanged.
//
// Contract: the caller passes ONE logical batch of size `n_replicas * b`.
// `shard_mask[i] != 0` means input i is split along its outermost dimension
// (replica r gets rows [r*b, (r+1)*b)); 0 means the buffer is replicated to every
// replica. Parameters are replicated, x and the labels are sharded.
//
// Outputs are read back from replica 0 ONLY. That is correct precisely because
// the emitted graph all-reduces every gradient before the optimizer consumes it
// (ViTRender.emitAdamVDP), so all replicas compute identical updated parameters.
// If that collective were ever missing, the replicas would silently diverge and
// this would quietly return replica 0's private answer — hence the check below
// that the executable really was compiled for `n_replicas`.
int pjrt_ffi_invoke_f32_dp(
    iree_ffi_session_t* sess,
    const char* fn_name,
    int n_replicas,
    int n_inputs,
    const int32_t* input_ranks,
    const int64_t* input_dims_flat,
    const float* const* input_data,
    const unsigned char* shard_mask,
    int n_outputs,
    const int64_t* output_totals,
    float* const* output_data) {

  if (!sess || !sess->exe) return 1;
  if (n_replicas < 1) return 1;

  if (n_replicas != sess->replicas) {
    fprintf(stderr,
            "[pjrt_ffi] DP invoke asked for %d replicas but @%s was compiled for "
            "%d (does the graph contain an all_reduce? is PJRT_REPLICAS set?)\n",
            n_replicas, sess->entry, sess->replicas);
    return 1;
  }
  if (!entry_matches(fn_name, sess->entry)) {
    fprintf(stderr, "[pjrt_ffi] entry mismatch: session holds @%s, caller asked '%s'\n",
            sess->entry, fn_name);
    return 1;
  }
  // G4, unchanged by data parallelism.
  if (sess->num_outputs >= 0 && sess->num_outputs != n_outputs) {
    fprintf(stderr,
            "[pjrt_ffi] G4 VIOLATION: @%s returns %d outputs, caller supplied %d\n",
            sess->entry, sess->num_outputs, n_outputs);
    return 1;
  }

  int rc = 0;
  PJRT_Buffer** in = (PJRT_Buffer**)calloc((size_t)n_replicas * n_inputs, sizeof(*in));
  PJRT_Buffer** out = (PJRT_Buffer**)calloc((size_t)n_replicas * n_outputs, sizeof(*out));
  PJRT_Buffer* const** arglists = (PJRT_Buffer* const**)calloc(n_replicas, sizeof(*arglists));
  PJRT_Buffer*** outlists = (PJRT_Buffer***)calloc(n_replicas, sizeof(*outlists));
  PJRT_Event** h2d = (PJRT_Event**)calloc((size_t)n_replicas * n_inputs, sizeof(*h2d));
  int64_t rdims[8];

  // §2d.3 accounting. Here `shard_mask` gives the classification exactly:
  // replicated == parameter. Note the param bytes are counted PER REPLICA,
  // because that is what actually crosses the link — the whole [theta|m|v] blob
  // is pushed to every device, every step, which is the diagnosed cause of the
  // 1.6-1.7x ceiling.
  const int tint = timing_interval();
  tacct_t* acct = NULL;
  if (tint) {
    acct = acct_for(sess->entry);
    acct_layout_once(sess->entry, n_inputs, input_dims_flat, input_ranks,
                     n_outputs, output_totals);
  }

  // Host -> device, per replica. All transfers issued before any is awaited.
  for (int rep = 0; rep < n_replicas && !rc; rep++) {
    int off = 0;
    for (int i = 0; i < n_inputs; i++) {
      int rank = input_ranks[i];
      const int64_t* d = &input_dims_flat[off];
      if (rank > (int)(sizeof(rdims)/sizeof(rdims[0]))) {
        fprintf(stderr, "[pjrt_ffi] input %d rank %d exceeds max\n", i, rank);
        rc = 2; break;
      }
      size_t elems = 1;
      for (int k = 0; k < rank; k++) { rdims[k] = d[k]; elems *= (size_t)d[k]; }

      const float* src = input_data[i];
      if (shard_mask && shard_mask[i]) {
        if (rank == 0 || d[0] % n_replicas != 0) {
          fprintf(stderr,
                  "[pjrt_ffi] input %d marked sharded but outer dim %lld is not "
                  "divisible by %d replicas\n",
                  i, rank ? (long long)d[0] : -1LL, n_replicas);
          rc = 2; break;
        }
        rdims[0] = d[0] / n_replicas;
        src = input_data[i] + (size_t)rep * (elems / (size_t)n_replicas);
      }

      PJRT_Client_BufferFromHostBuffer_Args a = {0};
      a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      a.client = g_client;
      a.data = src;
      a.type = PJRT_Buffer_Type_F32;
      a.dims = rdims;
      a.num_dims = (size_t)rank;
      a.host_buffer_semantics =
          PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
      a.device = g_devices[rep];
      double t0 = acct ? now_ms() : 0;
      if (check(g_api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer(dp)")) {
        rc = 2; break;
      }
      if (acct) {
        double dt = now_ms() - t0;
        size_t relems = elems;
        if (shard_mask && shard_mask[i]) relems = elems / (size_t)n_replicas;
        double mb = relems * 4.0 / 1048576.0;
        if (shard_mask && shard_mask[i]) { acct->h2d_issue_head  += dt; acct->h2d_head_mb  += mb; }
        else                            { acct->h2d_issue_param += dt; acct->h2d_param_mb += mb; }
      }
      in[(size_t)rep * n_inputs + i] = a.buffer;
      h2d[(size_t)rep * n_inputs + i] = a.done_with_host_buffer;
      off += rank;
    }
  }
  {
    double tw = acct ? now_ms() : 0;
    for (int k = 0; k < n_replicas * n_inputs; k++)
      if (h2d[k] && await_event(h2d[k], "h2d(dp)")) rc = 2;
    if (acct) acct->h2d_await += now_ms() - tw;
  }
  if (rc) goto cleanup;

  for (int rep = 0; rep < n_replicas; rep++) {
    arglists[rep] = &in[(size_t)rep * n_inputs];
    outlists[rep] = &out[(size_t)rep * n_outputs];
  }

  {
    PJRT_ExecuteOptions eo = {0};
    eo.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
    PJRT_LoadedExecutable_Execute_Args ea = {0};
    ea.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    ea.executable = sess->exe;
    ea.options = &eo;
    ea.argument_lists = arglists;
    ea.num_devices = (size_t)n_replicas;
    ea.num_args = (size_t)n_inputs;
    ea.output_lists = outlists;
    PJRT_Event* dce[8] = {0};   // see the single-device comment
    if (acct) ea.device_complete_events = dce;
    double t0 = acct ? now_ms() : 0;
    if (check(g_api->PJRT_LoadedExecutable_Execute(&ea), "Execute(dp)")) { rc = 3; goto cleanup; }
    if (acct) {
      acct->exec += now_ms() - t0;
      double t1 = now_ms();
      for (int r = 0; r < n_replicas && r < 8; r++)
        if (dce[r]) await_event(dce[r], "device_complete(dp)");
      acct->exec_await += now_ms() - t1;
    }
  }

  // Device -> host from replica 0 only (all replicas hold the same result).
  {
    PJRT_Event** d2h = (PJRT_Event**)calloc((size_t)n_outputs, sizeof(*d2h));
    double td = acct ? now_ms() : 0;
    for (int i = 0; i < n_outputs; i++) {
      size_t want = (size_t)output_totals[i] * sizeof(float);
      if (acct) acct->d2h_mb += want / 1048576.0;
      PJRT_Buffer_ToHostBuffer_Args q = {0};
      q.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      q.src = out[i];
      q.dst = NULL;
      if (check(g_api->PJRT_Buffer_ToHostBuffer(&q), "ToHostBuffer(dp size)")) { rc = 4; break; }
      if (q.dst_size != want) {
        fprintf(stderr,
                "[pjrt_ffi] output %d size mismatch: graph %zu bytes, caller %zu\n",
                i, q.dst_size, want);
        rc = 4; break;
      }
      PJRT_Buffer_ToHostBuffer_Args a = {0};
      a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      a.src = out[i];
      a.dst = output_data[i];
      a.dst_size = want;
      if (check(g_api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer(dp)")) { rc = 4; break; }
      d2h[i] = a.event;
    }
    if (acct) { acct->d2h_issue += now_ms() - td; td = now_ms(); }
    for (int i = 0; i < n_outputs; i++)
      if (d2h[i] && await_event(d2h[i], "d2h(dp)")) rc = 4;
    if (acct) acct->d2h_await += now_ms() - td;
    free(d2h);
  }

  if (acct && ++acct->calls % tint == 0) acct_report(acct);

  if (trace_enabled())
    fprintf(stderr, "[pjrt_ffi] @%s ok (%d replicas, %d in, %d out)\n",
            sess->entry, n_replicas, n_inputs, n_outputs);

cleanup:
  for (int k = 0; k < n_replicas * n_inputs; k++)
    if (in[k]) {
      PJRT_Buffer_Destroy_Args d = {0};
      d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      d.buffer = in[k];
      g_api->PJRT_Buffer_Destroy(&d);
    }
  for (int k = 0; k < n_replicas * n_outputs; k++)
    if (out[k]) {
      PJRT_Buffer_Destroy_Args d = {0};
      d.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      d.buffer = out[k];
      g_api->PJRT_Buffer_Destroy(&d);
    }
  free(in); free(out); free(arglists); free(outlists); free(h2d);
  return rc;
}

// ─── not yet ported ────────────────────────────────────────────────────────
// The packed-params / Adam / segmentation / DDPM / YOLO entry points are rungs
// 1-4 of planning/xla_pjrt_ladder.md. They are declared in iree_ffi.h, so they
// must exist for linking; each fails loudly rather than returning garbage.

static int not_ported(const char* who) {
  fprintf(stderr,
          "[pjrt_ffi] %s is not implemented on the XLA backend yet "
          "(see planning/xla_pjrt_ladder.md §2) — use the IREE build for this net\n",
          who);
  return 99;
}

int iree_ffi_train_step_mlp(
    iree_ffi_session_t* s, const char* f, int b,
    const float* a1, const float* a2, const float* a3, const float* a4,
    const float* a5, const float* a6, const float* a7, const int32_t* a8, float a9,
    float* o1, float* o2, float* o3, float* o4, float* o5, float* o6, float* o7) {
  (void)s;(void)f;(void)b;(void)a1;(void)a2;(void)a3;(void)a4;(void)a5;(void)a6;
  (void)a7;(void)a8;(void)a9;(void)o1;(void)o2;(void)o3;(void)o4;(void)o5;(void)o6;(void)o7;
  return not_ported("train_step_mlp");
}

int iree_ffi_train_step_generic(
    iree_ffi_session_t* s, const char* f, int b, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const int32_t* y, float lr,
    float* po, float* lo) {
  (void)s;(void)f;(void)b;(void)np;(void)pr;(void)pd;(void)ps;(void)pp;
  (void)xr;(void)xd;(void)x;(void)y;(void)lr;(void)po;(void)lo;
  return not_ported("train_step_generic");
}

int iree_ffi_train_step_adam(
    iree_ffi_session_t* s, const char* f, int b, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const int32_t* y, float lr, float t,
    float* po, float* lo, int nb, const int64_t* bs, float* bo) {
  (void)s;(void)f;(void)b;(void)np;(void)pr;(void)pd;(void)ps;(void)pp;
  (void)xr;(void)xd;(void)x;(void)y;(void)lr;(void)t;(void)po;(void)lo;
  (void)nb;(void)bs;(void)bo;
  return not_ported("train_step_adam");
}

int iree_ffi_train_step_adam_seg(
    iree_ffi_session_t* s, const char* f, int b, int H, int W, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const int32_t* y, float lr, float t,
    float* po, float* lo, int nb, const int64_t* bs, float* bo) {
  (void)s;(void)f;(void)b;(void)H;(void)W;(void)np;(void)pr;(void)pd;(void)ps;(void)pp;
  (void)xr;(void)xd;(void)x;(void)y;(void)lr;(void)t;(void)po;(void)lo;
  (void)nb;(void)bs;(void)bo;
  return not_ported("train_step_adam_seg");
}

int iree_ffi_train_step_adam_softlabel(
    iree_ffi_session_t* s, const char* f, int b, int nc, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const float* ys, float lr, float t,
    float* po, float* lo, int nb, const int64_t* bs, float* bo) {
  (void)s;(void)f;(void)b;(void)nc;(void)np;(void)pr;(void)pd;(void)ps;(void)pp;
  (void)xr;(void)xd;(void)x;(void)ys;(void)lr;(void)t;(void)po;(void)lo;
  (void)nb;(void)bs;(void)bo;
  return not_ported("train_step_adam_softlabel");
}

int iree_ffi_train_step_adam_ddpm(
    iree_ffi_session_t* s, const char* f, int b, int oC, int oH, int oW, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const float* yd, float lr, float t,
    float* po, float* lo, int nb, const int64_t* bs, float* bo) {
  (void)s;(void)f;(void)b;(void)oC;(void)oH;(void)oW;(void)np;(void)pr;(void)pd;
  (void)ps;(void)pp;(void)xr;(void)xd;(void)x;(void)yd;(void)lr;(void)t;(void)po;
  (void)lo;(void)nb;(void)bs;(void)bo;
  return not_ported("train_step_adam_ddpm");
}

int iree_ffi_train_step_adam_yolov1(
    iree_ffi_session_t* s, const char* f, int b, int gH, int gW, int pc, int np,
    const int32_t* pr, const int64_t* pd, const int64_t* ps, const float* pp,
    int xr, const int64_t* xd, const float* x, const float* yy, const float* my,
    float lr, float t,
    float* po, float* lo, int nb, const int64_t* bs, float* bo) {
  (void)s;(void)f;(void)b;(void)gH;(void)gW;(void)pc;(void)np;(void)pr;(void)pd;
  (void)ps;(void)pp;(void)xr;(void)xd;(void)x;(void)yy;(void)my;(void)lr;(void)t;
  (void)po;(void)lo;(void)nb;(void)bs;(void)bo;
  return not_ported("train_step_adam_yolov1");
}
