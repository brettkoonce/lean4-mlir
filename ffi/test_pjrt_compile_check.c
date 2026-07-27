// Compile-only check: does a Lean-emitted module compile under PJRT with the
// given CompileOptions (e.g. num_replicas=2)? Reports outputs on success.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "pjrt_c_api.h"
static const PJRT_Api* api;
static int check(PJRT_Error* e, const char* w){ if(!e) return 0;
  PJRT_Error_Message_Args m={0}; m.struct_size=PJRT_Error_Message_Args_STRUCT_SIZE; m.error=e;
  api->PJRT_Error_Message(&m); fprintf(stderr,"FAIL %s: %.*s\n",w,(int)m.message_size,m.message); return 1; }
static char* slurp(const char*p,size_t*n){ FILE*f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
  fseek(f,0,SEEK_END); long s=ftell(f); fseek(f,0,SEEK_SET); char*b=malloc(s+1);
  if(fread(b,1,s,f)!=(size_t)s){fprintf(stderr,"short read\n");exit(1);} b[s]=0; fclose(f); *n=s; return b; }
static char* rename_main(const char* src){
  const char* p=strstr(src,"func.func @"); if(!p){fprintf(stderr,"no entry\n");exit(1);}
  const char* nm=p+strlen("func.func @"); const char* e=nm;
  while(*e && (*e=='_'||*e=='.'||(*e>='0'&&*e<='9')||(*e>='a'&&*e<='z')||(*e>='A'&&*e<='Z'))) e++;
  size_t pre=nm-src, tail=strlen(e); char* out=malloc(pre+4+tail+1);
  memcpy(out,src,pre); memcpy(out+pre,"main",4); memcpy(out+pre+4,e,tail+1); return out; }
int main(int argc,char**argv){
  void* h=dlopen(argv[1],RTLD_LAZY|RTLD_LOCAL); if(!h){fprintf(stderr,"dlopen: %s\n",dlerror());return 1;}
  const PJRT_Api*(*G)(void)=dlsym(h,"GetPjrtApi"); api=G();
  PJRT_Client_Create_Args ca={0}; ca.struct_size=PJRT_Client_Create_Args_STRUCT_SIZE;
  if(check(api->PJRT_Client_Create(&ca),"Client_Create")) return 1;
  PJRT_Client_AddressableDevices_Args da={0}; da.struct_size=PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  da.client=ca.client; check(api->PJRT_Client_AddressableDevices(&da),"devices");
  printf("devices: %zu\n", da.num_addressable_devices);
  size_t mn,on; char* raw=slurp(argv[2],&mn); char* mlir=rename_main(raw); char* opts=slurp(argv[3],&on);
  PJRT_Program pr={0}; pr.struct_size=PJRT_Program_STRUCT_SIZE;
  pr.code=mlir; pr.code_size=strlen(mlir); pr.format="mlir"; pr.format_size=4;
  PJRT_Client_Compile_Args cc={0}; cc.struct_size=PJRT_Client_Compile_Args_STRUCT_SIZE;
  cc.client=ca.client; cc.program=&pr; cc.compile_options=opts; cc.compile_options_size=on;
  if(check(api->PJRT_Client_Compile(&cc),"Client_Compile")) return 1;
  PJRT_LoadedExecutable_GetExecutable_Args ge={0};
  ge.struct_size=PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE; ge.loaded_executable=cc.executable;
  if(!check(api->PJRT_LoadedExecutable_GetExecutable(&ge),"GetExecutable")){
    PJRT_Executable_NumOutputs_Args no={0}; no.struct_size=PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
    no.executable=ge.executable;
    if(!check(api->PJRT_Executable_NumOutputs(&no),"NumOutputs"))
      printf("COMPILED OK — %zu outputs\n", no.num_outputs);
  }
  return 0; }
