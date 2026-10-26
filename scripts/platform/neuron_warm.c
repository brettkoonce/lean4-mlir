// Compile renders through the shim and nothing else, for scripts/platform/neuron_warm.sh: on an
// x86 box with no Trainium, Neuron's runtime under NEURON_RT_FAKE_INSTANCE_TYPE gives the plugin
// a client, the plugin compiles each graph with neuronx-cc into its compile cache, and only the
// load onto the fake NeuronCore fails. Through iree_ffi_session_create — the trainers' own entry
// point — so the client options, compile options and entry renaming, and with them the cache
// key, are the ones a trainer produces on the instance.
//
//   neuron_warm <shim.so> <render.mlir>...      exit 0 if every session was created or failed
//                                               only after the compile (see neuron_warm.sh)
#include <dlfcn.h>
#include <stdio.h>

typedef struct iree_ffi_session_t iree_ffi_session_t;
typedef iree_ffi_session_t* (*create_fn)(const char*);
typedef void (*release_fn)(iree_ffi_session_t*);

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s <shim.so> <render.mlir>...\n", argv[0]); return 2; }
  void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "FAIL load shim: %s\n", dlerror()); return 1; }
  create_fn create = (create_fn)dlsym(h, "iree_ffi_session_create");
  release_fn release = (release_fn)dlsym(h, "iree_ffi_session_release");
  if (!create || !release) { fprintf(stderr, "FAIL shim symbols: %s\n", dlerror()); return 1; }
  for (int i = 2; i < argc; i++) {
    iree_ffi_session_t* s = create(argv[i]);
    // On a fake device the session never exists: the plugin loads the NEFF inside Compile, and
    // that load is what fails. Whether the NEFF reached the cache is neuron_warm.sh's question.
    printf("%s %s\n", s ? "CREATED" : "NO-SESSION", argv[i]);
    fflush(stdout);
    if (s) release(s);
  }
  return 0;
}
