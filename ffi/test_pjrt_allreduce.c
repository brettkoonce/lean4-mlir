// pjrt_allreduce.c — known-answer test for a 2-replica collective through the
// SAME PJRT C-API path the verified trainers use. No JAX, no Python.
//
// This is the multi-GPU analogue of the rung-0 spike: prove the mechanism on a
// case whose answer can be read off the screen, before touching the emitter.
//
// replica 0 gets [1,2,3,4], replica 1 gets [10,20,30,40];
// after all_reduce(add) BOTH replicas must hold [11,22,33,44].
//
// build:
//   gcc -O2 -I<ffi> pjrt_allreduce.c -ldl -o pjrt_allreduce
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include "pjrt_c_api.h"

static const PJRT_Api* api = NULL;

static void die(PJRT_Error* err, const char* what) {
  if (!err) return;
  PJRT_Error_Message_Args m = {0};
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = err;
  api->PJRT_Error_Message(&m);
  fprintf(stderr, "FAIL %s: %.*s\n", what, (int)m.message_size, m.message);
  exit(1);
}

static void await(PJRT_Event* ev, const char* what) {
  PJRT_Event_Await_Args a = {0};
  a.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  a.event = ev;
  die(api->PJRT_Event_Await(&a), what);
  PJRT_Event_Destroy_Args d = {0};
  d.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  d.event = ev;
  api->PJRT_Event_Destroy(&d);
}

static char* slurp(const char* p, size_t* n) {
  FILE* f = fopen(p, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
  char* b = malloc(s + 1);
  if (fread(b, 1, s, f) != (size_t)s) { fprintf(stderr, "short read\n"); exit(1); }
  b[s] = 0; fclose(f); *n = s; return b;
}

#define NREP 2
#define N 4

int main(int argc, char** argv) {
  const char* plugin = argv[1];
  const char* mlirp  = argv[2];
  const char* optsp  = argv[3];

  void* h = dlopen(plugin, RTLD_LAZY | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  const PJRT_Api* (*GetPjrtApi)(void) = dlsym(h, "GetPjrtApi");
  api = GetPjrtApi();
  printf("PJRT %d.%d\n", api->pjrt_api_version.major_version,
         api->pjrt_api_version.minor_version);

  PJRT_Client_Create_Args ca = {0};
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  die(api->PJRT_Client_Create(&ca), "Client_Create");
  PJRT_Client* client = ca.client;

  PJRT_Client_AddressableDevices_Args da = {0};
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client = client;
  die(api->PJRT_Client_AddressableDevices(&da), "AddressableDevices");
  printf("addressable devices: %zu\n", da.num_addressable_devices);
  if (da.num_addressable_devices < NREP) {
    fprintf(stderr, "need %d devices, have %zu — do NOT set HIP_VISIBLE_DEVICES\n",
            NREP, da.num_addressable_devices);
    return 1;
  }

  size_t mn, on;
  char* mlir = slurp(mlirp, &mn);
  char* opts = slurp(optsp, &on);

  PJRT_Program prog = {0};
  prog.struct_size = PJRT_Program_STRUCT_SIZE;
  prog.code = mlir; prog.code_size = strlen(mlir);
  prog.format = "mlir"; prog.format_size = 4;

  PJRT_Client_Compile_Args cc = {0};
  cc.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  cc.client = client;
  cc.program = &prog;
  cc.compile_options = opts;
  cc.compile_options_size = on;
  die(api->PJRT_Client_Compile(&cc), "Client_Compile (num_replicas=2)");
  printf("compiled the collective OK\n");

  // Per-replica inputs: replica r holds (r ? 10x : 1x) * [1,2,3,4].
  float host[NREP][N];
  for (int r = 0; r < NREP; r++)
    for (int i = 0; i < N; i++) host[r][i] = (r ? 10.0f : 1.0f) * (i + 1);

  const int64_t dims[1] = {N};
  PJRT_Buffer* in[NREP][1];
  for (int r = 0; r < NREP; r++) {
    PJRT_Client_BufferFromHostBuffer_Args a = {0};
    a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    a.client = client;
    a.data = host[r];
    a.type = PJRT_Buffer_Type_F32;
    a.dims = dims; a.num_dims = 1;
    a.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
    a.device = da.addressable_devices[r];      // <-- replica r on device r
    die(api->PJRT_Client_BufferFromHostBuffer(&a), "h2d");
    await(a.done_with_host_buffer, "h2d await");
    in[r][0] = a.buffer;
  }

  PJRT_Buffer* const* arglists[NREP] = { in[0], in[1] };
  PJRT_Buffer* outbuf[NREP][1] = {{0},{0}};
  PJRT_Buffer** outlists[NREP] = { outbuf[0], outbuf[1] };

  PJRT_ExecuteOptions eo = {0};
  eo.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

  PJRT_LoadedExecutable_Execute_Args ea = {0};
  ea.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  ea.executable = cc.executable;
  ea.options = &eo;
  ea.argument_lists = arglists;
  ea.num_devices = NREP;                       // <-- the multi-device call
  ea.num_args = 1;
  ea.output_lists = outlists;
  die(api->PJRT_LoadedExecutable_Execute(&ea), "Execute(num_devices=2)");
  printf("executed across %d replicas\n", NREP);

  int bad = 0;
  for (int r = 0; r < NREP; r++) {
    float got[N] = {0};
    PJRT_Buffer_ToHostBuffer_Args a = {0};
    a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    a.src = outbuf[r][0];
    a.dst = got; a.dst_size = sizeof(got);
    die(api->PJRT_Buffer_ToHostBuffer(&a), "d2h");
    await(a.event, "d2h await");
    printf("  replica %d -> [%.1f %.1f %.1f %.1f]\n", r,
           got[0], got[1], got[2], got[3]);
    for (int i = 0; i < N; i++) {
      float want = 11.0f * (i + 1);            // 1x + 10x
      if (got[i] != want) bad = 1;
    }
  }
  printf("\nexpected both replicas -> [11.0 22.0 33.0 44.0]\n");
  printf("%s\n", bad ? "*** COLLECTIVE WRONG ***"
                     : "ALL-REDUCE CORRECT across 2 GPUs via the PJRT C API");
  return bad;
}
