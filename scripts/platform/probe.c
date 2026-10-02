// probe.c — tier 0 of the platform suite (planning/platform_integration.md §3).
//
// Talks to the PJRT plugin directly, with no shim in between, so a failure here is the
// driver, the plugin or the install and never our runtime. Prints `key=value` lines that
// scripts/platform/manifest.py files into manifest.json.
//
//   probe <plugin.so>                        dlopen, API version, client, devices
//   probe --dump-options <n> <out.pb>        write the shim's n-replica CompileOptions blob
//
// The second form exists because ffi/test_pjrt_allreduce.c and test_pjrt_compile_check.c
// take the options as a file, and the blob the shim really compiles with lives only in
// ffi/pjrt_compile_options.h.
//
// build:  gcc -O2 -Iffi scripts/platform/probe.c -ldl -o <out>/probe
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pjrt_c_api.h"
#include "pjrt_compile_options.h"

static const PJRT_Api* api;

static int fail(PJRT_Error* e, const char* what) {
  if (!e) return 0;
  PJRT_Error_Message_Args m = {0};
  m.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  m.error = e;
  api->PJRT_Error_Message(&m);
  fprintf(stderr, "FAIL %s: %.*s\n", what, (int)m.message_size, m.message);
  return 1;
}

static int dump_options(int n, const char* out) {
  size_t len = 0;
  const unsigned char* buf = pjrt_compile_options_for(n, 0, &len);
  if (!buf) { fprintf(stderr, "no compile options for %d replicas\n", n); return 1; }
  FILE* f = fopen(out, "wb");
  if (!f || fwrite(buf, 1, len, f) != len) { perror(out); return 1; }
  fclose(f);
  return 0;
}

int main(int argc, char** argv) {
  if (argc == 4 && !strcmp(argv[1], "--dump-options"))
    return dump_options(atoi(argv[2]), argv[3]);
  if (argc != 2) {
    fprintf(stderr, "usage: probe <plugin.so> | probe --dump-options <n> <out.pb>\n");
    return 2;
  }

  void* h = dlopen(argv[1], RTLD_LAZY | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "FAIL dlopen: %s\n", dlerror()); return 1; }
  const PJRT_Api* (*get)(void) = (const PJRT_Api* (*)(void))dlsym(h, "GetPjrtApi");
  if (!get) { fprintf(stderr, "FAIL dlsym(GetPjrtApi): %s\n", dlerror()); return 1; }
  api = get();
  int maj = api->pjrt_api_version.major_version, min = api->pjrt_api_version.minor_version;
  printf("pjrt_api_header=%d.%d\n", PJRT_API_MAJOR, PJRT_API_MINOR);
  printf("pjrt_api_plugin=%d.%d\n", maj, min);
  // The C API promises compatibility within a major version; a minor skew either way is
  // reported, not failed — tier 1 is what says whether our calls still work.
  if (maj != PJRT_API_MAJOR) {
    fprintf(stderr, "FAIL PJRT major version: plugin %d, header %d\n", maj, PJRT_API_MAJOR);
    return 1;
  }

  PJRT_Plugin_Initialize_Args pi = {0};
  pi.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  if (fail(api->PJRT_Plugin_Initialize(&pi), "Plugin_Initialize")) return 1;

  PJRT_Client_Create_Args ca = {0};
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  if (fail(api->PJRT_Client_Create(&ca), "Client_Create")) return 1;

  PJRT_Client_PlatformName_Args pn = {0};
  pn.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  pn.client = ca.client;
  if (fail(api->PJRT_Client_PlatformName(&pn), "PlatformName")) return 1;
  printf("platform_name=%.*s\n", (int)pn.platform_name_size, pn.platform_name);

  PJRT_Client_PlatformVersion_Args pv = {0};
  pv.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  pv.client = ca.client;
  if (fail(api->PJRT_Client_PlatformVersion(&pv), "PlatformVersion")) return 1;
  printf("platform_version=%.*s\n", (int)pv.platform_version_size, pv.platform_version);

  PJRT_Client_AddressableDevices_Args da = {0};
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client = ca.client;
  if (fail(api->PJRT_Client_AddressableDevices(&da), "AddressableDevices")) return 1;
  printf("devices=%zu\n", da.num_addressable_devices);
  for (size_t i = 0; i < da.num_addressable_devices; i++) {
    PJRT_Device_GetDescription_Args gd = {0};
    gd.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    gd.device = da.addressable_devices[i];
    if (fail(api->PJRT_Device_GetDescription(&gd), "GetDescription")) return 1;
    PJRT_DeviceDescription_Kind_Args dk = {0};
    dk.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    dk.device_description = gd.device_description;
    if (fail(api->PJRT_DeviceDescription_Kind(&dk), "DeviceKind")) return 1;
    printf("device_%zu=%.*s\n", i, (int)dk.device_kind_size, dk.device_kind);
  }
  if (da.num_addressable_devices == 0) { fprintf(stderr, "FAIL no devices\n"); return 1; }

  PJRT_Client_Destroy_Args cd = {0};
  cd.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  cd.client = ca.client;
  fail(api->PJRT_Client_Destroy(&cd), "Client_Destroy");
  return 0;
}
