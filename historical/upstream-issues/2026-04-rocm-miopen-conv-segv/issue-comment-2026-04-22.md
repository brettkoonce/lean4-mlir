Some additional diagnostic info from further bisection:

**MIOpen debug log pinpoints the crash to the compile-from-source fallback.**
Running the repro with `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6`, the
last line before the SIGSEGV is consistently:

```
MIOpen(HIP): Info2 [Prepare] SELECT kernel_blob, kernel_hash,
   uncompressed_size FROM kern_db WHERE (kernel_name = 'naive_conv.cpp.o')
   AND (kernel_args = ' -DMIOPEN_USE_FP16=0 ... -mcpu=gfx1100');
MIOpen(HIP): Info2 [Measure] Db::FindRecord time: 0.022635 ms
MIOpen(HIP): Info2 [LoadBinary] Unable to load binary for: "naive_conv.cpp.o";
   args: -DMIOPEN_USE_FP16=0 -mcpu=gfx1100
[SIGSEGV]
```

So the kernel-DB query itself completes cleanly — the row comes back
without a usable blob, `LoadBinary` returns failure, and MIOpen falls
through to source-compile via `GetKernelSrc` → `HIPOCProgramImpl::BuildCodeObject`,
which is where the crash actually happens. Falling through to
compile-from-source is expected behavior; crashing there is the bug.

**Env-var matrix — nothing sidesteps it.** For anyone else debugging this,
I tried each of these against the 6-line repro; all SIGSEGV:

- `HIP_VISIBLE_DEVICES=0`, `=1`, `ROCR_VISIBLE_DEVICES=0`
- Deleting `~/.cache/miopen/` entirely (forces DB rebuild; same crash first run)
- `ROCPROFILER_DISABLE_TOOL=1`, `ROCP_METRICS=none`,
  `ROCPROFILER_REGISTER_DISABLE=1`
- `HSA_ENABLE_INTERRUPT=0`
- `MIOPEN_FIND_MODE=1` (fast fallback)
- `MIOPEN_DEBUG_DISABLE_FIND_DB=1`
- `MIOPEN_USER_DB_PATH=/tmp/...` (redirect DB to fresh location)

Crash is deterministic and independent of profiler state, cache state,
and GPU visibility. Only workaround is `JAX_PLATFORMS=cpu`.
