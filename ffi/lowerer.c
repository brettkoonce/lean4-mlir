// Runtime lowerer dispatch. See lowerer.h for why this exists.
//
// Selection, highest precedence first:
//   $LEAN_MLIR_LOWERER_SO  explicit path to a shim .so   (one attempt, no fallback)
//   $LEAN_MLIR_LOWERER     "xla"|"pjrt" or "iree"        (one choice, no fallback)
//   default                try XLA, then IREE
//
// An explicit request never silently falls back: running the wrong backend under
// a name the user chose is worse than failing. That rule is inherited from
// pjrt_ffi.c's own $PJRT_PLUGIN handling, and for the same reason.

#include "lowerer.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Defined via __typeof__ of the header's own declarations, so a signature can
// never drift between the two files -- there is only one copy of each type.
#define DEFINE_PTR(name) __typeof__(name) name = NULL

DEFINE_PTR(lowerer_session_release);
DEFINE_PTR(lowerer_invoke_f32);
DEFINE_PTR(lowerer_train_step_mlp);
DEFINE_PTR(lowerer_train_step_generic);
DEFINE_PTR(lowerer_train_step_adam);
DEFINE_PTR(lowerer_train_step_adam_seg);
DEFINE_PTR(lowerer_train_step_adam_softlabel);
DEFINE_PTR(lowerer_train_step_adam_ddpm);
DEFINE_PTR(lowerer_train_step_adam_yolov1);

DEFINE_PTR(lowerer_pjrt_marker);
DEFINE_PTR(lowerer_pjrt_invoke_f32_resident_v2);
DEFINE_PTR(lowerer_pjrt_resident_read);
DEFINE_PTR(lowerer_pjrt_invoke_f32_dp);

static const char* g_active = "none";
static int load_once(void);

// Forces the load. The Lean driver asks which backend it has BEFORE opening a
// session -- it has to, since the answer decides whether to iree-compile to a
// .vmfb or hand the .mlir over directly. Answering "iree" merely because nothing
// had loaded yet sent a .vmfb to the XLA shim, which rejected it with
// "no 'func.func @' in module". So the question itself must resolve the backend.
const char* lowerer_active_name(void) { load_once(); return g_active; }

// ⚠⚠ THE BARE SONAME MUST COME FIRST, and the order is the whole point.
//
// `dlopen("name.so")` consults $LD_LIBRARY_PATH; `dlopen("./ffi/name.so")` does
// not -- it is a path, so the loader takes it literally. With the relative entry
// first, the repo copy ALWAYS won and LD_LIBRARY_PATH could never be honoured,
// which silently disabled every gate that injects a purpose-built shim that way:
// `tests/prefetch_tie.sh` and `scripts/residency_gate_all.sh` both point
// LD_LIBRARY_PATH at a deterministic, autotuning-off build (`scripts/det_shim.sh`)
// and neither was getting it. Measured 2026-08-11: prefetch_tie's A1-vs-A2
// control failed at ~145 M differing bytes -- the documented signature of "the
// deterministic shim did not take" -- and passed at 0 with the order fixed.
// It fails loudly rather than passing wrongly, but the gates were unusable.
//
// This regressed when the lowerer moved from link-time to dlopen: a LINKED
// binary resolves its DT_NEEDED soname through LD_LIBRARY_PATH, so the gates
// worked by construction before and quietly stopped after.
//
// Bare-first costs nothing in the normal case: no LD_LIBRARY_PATH entry means
// the bare dlopen simply fails and the relative entry answers, which is how
// every ordinary run already resolves. Relative entries resolve against the CWD,
// which for every runner in this repo is the repo root (the trainers already
// read `verified_mlir/` and `data/` that way).
static const char* const kXlaPaths[]  = { "libpjrt_ffi.so", "./ffi/libpjrt_ffi.so" };
static const char* const kIreePaths[] = { "libiree_ffi.so", "./ffi/libiree_ffi.so" };

static void* try_paths(const char* const* paths, int n) {
  for (int i = 0; i < n; i++) {
    void* h = dlopen(paths[i], RTLD_LAZY | RTLD_LOCAL);
    if (h) return h;
  }
  return NULL;
}

// Required: absence is a broken shim, not a configuration choice.
#define REQ(field, sym)                                                     \
  do {                                                                      \
    *(void**)(&field) = dlsym(h, sym);                                      \
    if (!field) {                                                           \
      fprintf(stderr, "[lowerer] %s is missing '%s' -- not a usable shim\n", \
              g_active, sym);                                               \
      return 1;                                                             \
    }                                                                       \
  } while (0)

// Optional: NULL here means exactly what a NULL weak symbol used to mean.
#define OPT(field, sym) do { *(void**)(&field) = dlsym(h, sym); } while (0)

static int bind_all(void* h) {
  REQ(lowerer_session_release,           "iree_ffi_session_release");
  REQ(lowerer_invoke_f32,                "iree_ffi_invoke_f32");
  REQ(lowerer_train_step_mlp,            "iree_ffi_train_step_mlp");
  REQ(lowerer_train_step_generic,        "iree_ffi_train_step_generic");
  REQ(lowerer_train_step_adam,           "iree_ffi_train_step_adam");
  REQ(lowerer_train_step_adam_seg,       "iree_ffi_train_step_adam_seg");
  REQ(lowerer_train_step_adam_softlabel, "iree_ffi_train_step_adam_softlabel");
  REQ(lowerer_train_step_adam_ddpm,      "iree_ffi_train_step_adam_ddpm");
  REQ(lowerer_train_step_adam_yolov1,    "iree_ffi_train_step_adam_yolov1");

  OPT(lowerer_pjrt_marker,                 "pjrt_ffi_marker");
  OPT(lowerer_pjrt_invoke_f32_resident_v2, "pjrt_ffi_invoke_f32_resident_v2");
  OPT(lowerer_pjrt_resident_read,          "pjrt_ffi_resident_read");
  OPT(lowerer_pjrt_invoke_f32_dp,          "pjrt_ffi_invoke_f32_dp");
  return 0;
}

// The session ctor is resolved separately because it is the only symbol needed
// before `g_active` is meaningful.
static iree_ffi_session_t* (*s_session_create)(const char*) = NULL;

static int load_once(void) {
  static int done = 0;
  if (done) return s_session_create ? 0 : 1;
  done = 1;

  const char* so   = getenv("LEAN_MLIR_LOWERER_SO");
  const char* want = getenv("LEAN_MLIR_LOWERER");
  void* h = NULL;

  if (so && *so) {
    h = dlopen(so, RTLD_LAZY | RTLD_LOCAL);
    if (!h) {
      fprintf(stderr, "[lowerer] $LEAN_MLIR_LOWERER_SO=%s did not load: %s\n", so, dlerror());
      return 1;
    }
    g_active = "custom";
  } else if (want && (!strcmp(want, "xla") || !strcmp(want, "pjrt"))) {
    h = try_paths(kXlaPaths, 2);
    g_active = "xla";
  } else if (want && !strcmp(want, "iree")) {
    h = try_paths(kIreePaths, 2);
    g_active = "iree";
  } else if (want && *want) {
    fprintf(stderr, "[lowerer] $LEAN_MLIR_LOWERER=%s is not one of: xla, iree\n", want);
    return 1;
  } else {
    h = try_paths(kXlaPaths, 2);
    g_active = "xla";
    if (!h) { h = try_paths(kIreePaths, 2); g_active = "iree"; }
  }

  if (!h) {
    fprintf(stderr,
            "[lowerer] no lowerer shim could be loaded (wanted: %s).\n"
            "  build one:  gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so\n"
            "  or select:  LEAN_MLIR_LOWERER=iree, or $LEAN_MLIR_LOWERER_SO=<path>\n",
            g_active);
    g_active = "none";
    return 1;
  }

  // Report what actually loaded, not what was asked for: the marker is defined
  // only by the PJRT shim, so it is the ground truth `lean_iree_backend_name`
  // has always relied on.
  //
  // ⚠ THIS USED TO BE GUARDED BY `if (!(so && *so))`, which exempted the one path
  // that most needs the ground truth. $LEAN_MLIR_LOWERER_SO left `g_active` at
  // "custom", the Lean driver matches that against "xla" and gets no, so pointing
  // the override at a PJRT shim ran the IREE path and died in `iree-compile` --
  // even with $LEAN_MLIR_LOWERER=xla also set, since the branch above never ran.
  // The guard contradicted the comment it sat under: "custom" is what was ASKED,
  // and the marker is what LOADED. Unconditional now, so an explicit .so is
  // classified by what it exports, exactly like an implicitly-found one.
  if (bind_all(h)) { g_active = "none"; return 1; }
  g_active = lowerer_pjrt_marker ? "xla" : "iree";

  *(void**)(&s_session_create) = dlsym(h, "iree_ffi_session_create");
  if (!s_session_create) {
    fprintf(stderr, "[lowerer] shim is missing 'iree_ffi_session_create'\n");
    g_active = "none";
    return 1;
  }
  return 0;
}

iree_ffi_session_t* lowerer_session_create(const char* path) {
  if (load_once()) return NULL;
  return s_session_create(path);
}
