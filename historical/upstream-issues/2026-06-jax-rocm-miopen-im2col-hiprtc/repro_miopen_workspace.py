#!/usr/bin/env python3
"""Direct-MIOpen probe of the forward-conv *immediate* API on gfx1100 (no JAX/XLA).

Drives MIOpen's forward-convolution immediate API by hand (ctypes) for the ViT-Tiny
patch-embed conv, with NO JAX/XLA in the loop. Built to chase a reported im2col/hiprtc
`get_global_id` compile crash — which, on re-verification (2026-06-24, ROCm 7.2.0 /
MIOpen 3.5.1.70200), does NOT reproduce. See README.md. What this script actually shows:

  --workspace full  (control)  give GemmFwdRest the workspace it asks for.
        -> RESULT: OK. With the kernel cache moved aside, the cold-compile log shows
           MIOpenIm2d2Col.cpp.o building cleanly via HIPRTC v.9.0 (no get_global_id error).

  --workspace zero             give GemmFwdRest a 0-byte workspace.
        -> miopenStatusUnknownError (7): "Not enough workspace for GemmFwdRest
           (0 provided, N required)" at gemm.cpp:957. A clean refusal — GemmFwdRest has
           no inline no-workspace fallback in this build; it does NOT crash in hiprtc.

Run with MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6 to see the solver/compile breadcrumbs.
"""
import ctypes as C
import sys

WS_MODE = "zero"
for i, a in enumerate(sys.argv):
    if a == "--workspace" and i + 1 < len(sys.argv):
        WS_MODE = sys.argv[i + 1]

hip = C.CDLL("libamdhip64.so", mode=C.RTLD_GLOBAL)
mi = C.CDLL("libMIOpen.so", mode=C.RTLD_GLOBAL)

# ── HIP ───────────────────────────────────────────────────────────────────────
def hipchk(code, what):
    if code != 0:
        raise RuntimeError(f"HIP error {code} in {what}")

def dmalloc(nbytes):
    p = C.c_void_p()
    hipchk(hip.hipMalloc(C.byref(p), C.c_size_t(nbytes)), "hipMalloc")
    hipchk(hip.hipMemset(p, 0, C.c_size_t(nbytes)), "hipMemset")
    return p

# ── MIOpen enums ──────────────────────────────────────────────────────────────
miopenBFloat16 = 5
miopenConvolution = 0
STATUS = {0: "Success", 1: "NotInitialized", 2: "InvalidValue", 3: "BadParm",
          4: "AllocFailed", 5: "InternalError", 6: "NotImplemented",
          7: "UnknownError", 8: "UnsupportedOp", 9: "GpuOperationsSkipped",
          10: "VersionMismatch"}

def mchk(code, what):
    name = STATUS.get(code, str(code))
    print(f"    [{what}] -> miopenStatus{name} ({code})", flush=True)
    return code

class ConvSolution(C.Structure):
    _fields_ = [("time", C.c_float),
                ("workspace_size", C.c_size_t),
                ("solution_id", C.c_uint64),
                ("algorithm", C.c_int)]

# ── build the ViT-Tiny patch-embed conv problem ───────────────────────────────
# x:(N,3,224,224) w:(192,3,16,16) stride16 pad0 -> y:(N,192,14,14), bf16, NCHW.
N = 1
handle = C.c_void_p()
mchk(mi.miopenCreate(C.byref(handle)), "miopenCreate")

def tensor4d(n, c, h, w):
    t = C.c_void_p()
    mi.miopenCreateTensorDescriptor(C.byref(t))
    mi.miopenSet4dTensorDescriptor(t, miopenBFloat16, n, c, h, w)
    return t

xDesc = tensor4d(N, 3, 224, 224)
wDesc = tensor4d(192, 3, 16, 16)
yDesc = tensor4d(N, 192, 14, 14)

convDesc = C.c_void_p()
mi.miopenCreateConvolutionDescriptor(C.byref(convDesc))
# miopenInitConvolutionDescriptor(c, mode, pad_h, pad_w, str_h, str_w, dil_h, dil_w)
mi.miopenInitConvolutionDescriptor(convDesc, miopenConvolution, 0, 0, 16, 16, 1, 1)

# device buffers (zeroed; values irrelevant — crash is at kernel compile)
xb = dmalloc(N * 3 * 224 * 224 * 2)
wb = dmalloc(192 * 3 * 16 * 16 * 2)
yb = dmalloc(N * 192 * 14 * 14 * 2)

# ── workspace MIOpen *wants* ──────────────────────────────────────────────────
ws_needed = C.c_size_t(0)
mi.miopenConvolutionForwardGetWorkSpaceSize(
    handle, wDesc, xDesc, convDesc, yDesc, C.byref(ws_needed))
print(f"  GetWorkSpaceSize wants: {ws_needed.value} bytes", flush=True)

# ── enumerate immediate-mode solutions ────────────────────────────────────────
cnt = C.c_size_t(0)
mi.miopenConvolutionForwardGetSolutionCount(
    handle, wDesc, xDesc, convDesc, yDesc, C.byref(cnt))
print(f"  solution count: {cnt.value}", flush=True)

nsol = max(1, cnt.value)
sols = (ConvSolution * nsol)()
got = C.c_size_t(0)
mi.miopenConvolutionForwardGetSolution(
    handle, wDesc, xDesc, convDesc, yDesc, nsol, C.byref(got), sols)
print(f"  solutions returned: {got.value}", flush=True)
for i in range(got.value):
    s = sols[i]
    print(f"    sol[{i}] id={s.solution_id} algo={s.algorithm} "
          f"ws={s.workspace_size} time={s.time:.2f}", flush=True)

# pick the GEMM solution (algorithm 0 == miopenConvolutionFwdAlgoGEMM)
chosen = None
for i in range(got.value):
    if sols[i].algorithm == 0:
        chosen = sols[i]; break
if chosen is None and got.value:
    chosen = sols[0]
print(f"  chosen solution_id={chosen.solution_id} (ws={chosen.workspace_size})", flush=True)

# ── compile + run the chosen solution ─────────────────────────────────────────
print("  [CompileSolution] …", flush=True)
mchk(mi.miopenConvolutionForwardCompileSolution(
    handle, wDesc, xDesc, convDesc, yDesc, C.c_uint64(chosen.solution_id)),
    "CompileSolution")

if WS_MODE == "full":
    wsb = dmalloc(max(1, chosen.workspace_size))
    ws_ptr, ws_sz = wsb, C.c_size_t(chosen.workspace_size)
    print(f"  running with FULL workspace ({chosen.workspace_size} B) — control", flush=True)
else:
    ws_ptr, ws_sz = C.c_void_p(0), C.c_size_t(0)
    print("  running with ZERO workspace — forces inline im2col", flush=True)

print("  [ConvolutionForwardImmediate] …", flush=True)
rc = mi.miopenConvolutionForwardImmediate(
    handle,
    wDesc, wb,
    xDesc, xb,
    convDesc,
    yDesc, yb,
    ws_ptr, ws_sz,
    C.c_uint64(chosen.solution_id))
mchk(rc, "ConvolutionForwardImmediate")
hip.hipDeviceSynchronize()
print(f"  RESULT: {'OK' if rc == 0 else 'FAILED'} (rc={rc})", flush=True)
sys.exit(0 if rc == 0 else 7)
