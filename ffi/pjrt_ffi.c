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

#define DEFAULT_PLUGIN                                                        \
  "/home/skoonce/lean/claude_max/lean4-jax/.venv/lib/python3.12/"             \
  "site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so"

// ─── process-global PJRT client ────────────────────────────────────────────
// One client per process, refcounted across sessions. This is not an
// optimization: each StreamExecutor GPU client reserves ~19 GB for its BFC
// allocator, so two clients on one device fight over memory. The trainers make
// two sessions (train step + forward), which must share.

static const PJRT_Api* g_api = NULL;
static PJRT_Client* g_client = NULL;
static PJRT_Device* g_device = NULL;
static int g_client_refs = 0;

struct iree_ffi_session_t {
  PJRT_LoadedExecutable* exe;
  char* entry;      // original func name, before the @main rename
  int num_outputs;  // from the compiled executable — used by the G4 guard
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
  if (!plugin) plugin = DEFAULT_PLUGIN;
  void* h = dlopen(plugin, RTLD_LAZY | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "[pjrt_ffi] dlopen(%s): %s\n", plugin, dlerror());
    fprintf(stderr, "[pjrt_ffi] set $PJRT_PLUGIN to the PJRT plugin .so\n");
    return 1;
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
  cc.compile_options = (const char*)kPjrtCompileOptions;
  cc.compile_options_size = kPjrtCompileOptionsSize;
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
  fprintf(stderr, "[pjrt_ffi] compiled %s (@%s, %d outputs) in %.0f ms\n",
          path, entry, num_outputs, compile_ms);
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
      if (check(g_api->PJRT_Client_BufferFromHostBuffer(&a), "BufferFromHostBuffer")) {
        failed = 1; break;
      }
      in[i] = a.buffer;
      h2d[i] = a.done_with_host_buffer;
      off += input_ranks[i];
    }
    for (int i = 0; i < n_inputs; i++)
      if (h2d[i] && await_event(h2d[i], "h2d")) failed = 1;
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
    if (check(g_api->PJRT_LoadedExecutable_Execute(&ea), "Execute")) { rc = 3; goto cleanup; }
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
    for (int i = 0; i < n_outputs; i++) {
      size_t want = (size_t)output_totals[i] * sizeof(float);

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
      a.dst = output_data[i];
      a.dst_size = want;
      if (check(g_api->PJRT_Buffer_ToHostBuffer(&a), "ToHostBuffer")) { failed = 1; break; }
      d2h[i] = a.event;
    }
    for (int i = 0; i < n_outputs; i++)
      if (d2h[i] && await_event(d2h[i], "d2h")) failed = 1;
    free(d2h);
    if (failed) { rc = 4; goto cleanup; }
  }

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
