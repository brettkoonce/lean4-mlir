# MIOpen conv on gfx1100 (ROCm 7.2.0): the im2col/hiprtc crash no longer reproduces — what remains

**Status: NOT FILED — headline bug not reproducible.** Re-verified 2026-06-24.
The `get_global_id` hiprtc compile crash this doc was opened to chase does **not**
reproduce on the current stack. Conv nets run; the residue is a *performance*
fallback (ResNet-34 ~2× the reference ms/step) plus a latent *no-workspace*
limitation in MIOpen's GEMM solver. Nothing here is worth an upstream filing as-is.
The original mid-investigation hypothesis is preserved at the bottom for history.

---

## ⭐ UPDATE 2026-06-24 — re-verified; the crash is gone

Picked this back up to build the minimal reproducer (handoff step 3). Building it
**disproved the premise**. Findings on this exact box (ROCm 7.2.0 / MIOpen 3.5.1.70200,
jax 0.10.0 / jax-rocm7 0.9.1.post4, 2× RX 7900 XTX / gfx1100):

1. **The im2col HIP kernel compiles cleanly via hiprtc.** Forced a *cold* compile of
   `MIOpenIm2d2Col.cpp.o` (cache moved aside) for bf16 / `-mcpu=gfx1100`. It builds via
   **HIPRTC v.9.0**, saves the binary, and runs kernel `Im2d2Col_v2` → `RESULT: OK`.
   No `use of undeclared identifier 'get_global_id'`. The OpenCL→HIP builtin shim **is**
   in scope. The kernel db now caches `MIOpenIm2d2Col.cpp.o` across many flag variants,
   all compiled successfully. Whatever produced the original error was a transient
   find-db/cache state or was quietly fixed in this point build.

2. **What *does* reproduce deterministically is a different failure** — and it's a clean
   refusal, not a compile crash. Driving MIOpen's immediate conv API directly (no XLA,
   `repro_miopen_workspace.py`) with a **0-byte workspace** on `GemmFwdRest`:
   ```
   MIOpen Error: .../src/solver/conv/gemm.cpp:957:
       Not enough workspace for GemmFwdRest (0 provided, 301056 required)
   -> miopenStatusUnknownError (7)
   ```
   So `GemmFwdRest` (the only no-workspace-eligible forward GEMM solver) has **no inline
   no-workspace fallback** in this build — given 0 workspace it bails with
   `miopenStatusUnknownError` rather than running. That status code is what the original
   ViT patch-embed conv surfaced; the *cause* was the workspace refusal, not a hiprtc
   `get_global_id` error.

3. **ResNet-34 runs — just slow.** The bench completes end-to-end now:
   - ResNet-34: **283 ms/step**, 903 img/s, ~23.6 min/epoch (vs ~139 ms reference → ≈2×).
   - ViT-Tiny (conv-free): **185 ms/step**, 2770 img/s, clean.
   The 2× conv slowdown is a *solver-selection / fallback* issue (MIOpen lands on a slow
   path), **not** a crash. This matches the original "limped at ~278 ms/step" note — it
   was already the state then; nothing regressed or improved on r34.

**Net:** the bug as framed (hiprtc `get_global_id` blocks all conv nets) is stale. Conv
nets are *usable*, ~2× under peak. No upstream filing made — you can't file a crash you
can't reproduce. The two real, lower-severity residues to revisit if conv throughput
matters: **(a)** why XLA/MIOpen picks the slow conv solver on gfx1100, and **(b)** the
`GemmFwdRest` no-workspace-fallback gap.

### Reproducing the re-verification

```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH HIP_VISIBLE_DEVICES=0
export MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6

# (1) the deterministic "Not enough workspace" failure (immediate API, 0 workspace):
python3 repro_miopen_workspace.py --workspace zero      # -> miopenStatusUnknownError (7)
python3 repro_miopen_workspace.py --workspace full      # control -> RESULT: OK

# (2) cold im2col compile is clean (move the cache aside to force a fresh build):
mv ~/.cache/miopen ~/.cache/miopen.aside
python3 repro_miopen_workspace.py --workspace full      # logs HIPRTC v.9.0 compile -> OK
rm -rf ~/.cache/miopen && mv ~/.cache/miopen.aside ~/.cache/miopen

# (3) the real-world picture — bench runs, r34 ~2× slow, no crash:
/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python scripts/jax_imagenet_bench.py
```

---

## Environment

- Box `mars`: 2× Radeon RX 7900 XTX (gfx1100, RDNA3), ROCm **7.2.0**, MIOpen
  **3.5.1.70200** (miopen-hip), HIP runtime 7.2.70200, hiprtc **v9.0**.
- JAX venv: `/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python` —
  jax/jaxlib **0.10.0**, jax-rocm7-{plugin,pjrt} **0.9.1.post4**.
  ⚠ **Deliberately pinned** — it's the fix for #3955. A 7.2.0→7.2.4 *point* upgrade stays
  on ROCm 7.x and should keep the plugin happy (verify it still imports +
  `jax.device_count()==2` after); a ROCm *major* bump may not have a matching plugin.

## Next steps (if conv throughput becomes worth it)

1. **Diagnose the 2× conv slowdown.** With `MIOPEN_ENABLE_LOGGING=1 MIOPEN_LOG_LEVEL=6`,
   capture which solver each r34 conv lands on and the workspace XLA hands it. Hypothesis:
   XLA's conv scratch allocator is starved under the full multi-GPU graph → MIOpen takes a
   no/low-workspace slow solver instead of Winograd/implicit-GEMM. Check whether forcing a
   solver (`MIOPEN_DEBUG_CONV_WINOGRAD` / `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM`) or a bigger
   scratch budget restores ~139 ms/step.
2. **ROCm 7.2.0 → 7.2.4** (point upgrade, low risk) — check the MIOpen changelog for
   gfx1100 conv solver-selection / im2col fixes, then re-bench. May fix (1) outright.
3. **Bench policy:** ViT stays conv-free (works). For conv nets, the numbers are *real but
   ~2× slow*; cite them with that caveat, or cite the `jax/runs/*/RESULTS.md` reference
   numbers until the solver path is fixed.

## The Bench (feature context)

- **UNCOMMITTED:** `scripts/jax_imagenet_bench.py` — JAX/ImageNet training-time ETA, the
  per-chapter "sample" idea (like `lake run benchmark`) for the phase-2 (Lean→JAX) ImageNet
  path. Synthetic input (no dataset on disk), bf16, multi-GPU via `jax.device_count()` +
  a data-parallel mesh. Registry seeded with **ResNet-34 @ 90ep** and **ViT-Tiny @ 300ep**
  (DeiT recipe); extensible to MNv2/ENet/ConvNeXt.
- Built on `scripts/jax_r34_imagenet_bench.py` (the original r34-only synth-throughput ETA).
- Eventual home could be `lake run jax-benchmark`, mirroring `lake run benchmark`.

## Reference anchors (from `jax/runs/*_imagenet_*/RESULTS.md`, bf16)

| net | ms/step (ref) | min/epoch | full run | devices |
|---|---|---|---|---|
| ResNet-34 | ~139 | ~10.2 | ~15h / 90ep | 2 |
| MobileNetV2 | ~108 | ~9 | ~14h | 2 |
| EfficientNet-B0 | ~90 | ~8.0 | ~10.5h | 2 |
| ConvNeXt-T | ~143 | ~12.6 | ~15.5h | 6 |
| ViT-Tiny | ~185 | ~7.7 | ~11h / 80ep | ? |

(Heterogeneous device counts — normalize to images/sec/GPU, or re-measure all at a fixed
N once the conv solver path is fast again. Current bench measures r34 at ~283 ms/step,
≈2× the ~139 reference, on 2 GPUs.)

## Related

- `upstream-issues/2026-04-rocm-miopen-conv-segv/` — MIOpen#3955 (conv SIGSEGV, **fixed**).
- `upstream-issues/2026-04-jax-jit-conv-backward-segv/` — adjacent JAX/ROCm conv issue.

---

## Original mid-investigation hypothesis (2026-06, superseded by the 06-24 update above)

> Retained for history. The chain below was the working theory before the re-verification;
> step 2 (the hiprtc compile failure) does **not** reproduce on the current stack.

We were building a JAX/ImageNet training-time ETA bench (above). It appeared to expose a
MIOpen convolution bug: the im2col/GEMM conv solver's HIP kernel was reported to fail
hiprtc compilation on gfx1100 — `error: use of undeclared identifier 'get_global_id'`.
ViT was reworked to be **conv-free** (patch-embed = reshape + matmul) and runs clean —
**keep this**, it's leaner on RDNA3 regardless.

Hypothesised chain, bottom to top:

1. **No-workspace fallback.** Log showed `workspace required: 25690112, provided ptr: 0
   size: 0` — XLA hands MIOpen a 0-byte workspace, so MIOpen falls back to a no-workspace
   forward conv path (`GemmFwdRest`, source `MIOpenIm2d2Col.cpp`).
2. **That kernel was thought not to build.** MIOpen JIT-compiles it via hiprtc for
   gfx1100; its source uses OpenCL work-item builtins `get_global_id(0)` /
   `get_global_size(0)`. The theory was the OpenCL→HIP shim that `#define`s them wasn't in
   scope → `undeclared identifier`. **Re-verification 06-24: it *is* in scope; the kernel
   compiles fine cold via HIPRTC v9.0.** So this step did not hold up.
3. **Observed symptoms:** ViT's patch-embed conv threw `miopenStatusUnknownError` (now
   traced to the workspace refusal, not a compile error); ResNet-34 limped via a slow
   fallback at ~278 ms/step (≈2× the 139 ms reference — still the case).

### What was tried during the original investigation

- ✅ **ViT made conv-free** → no MIOpen, runs clean. Kept.
- ❌ **`MIOPEN_DEBUG_CONV_GEMM=0`** (disable the GEMM solver) → MIOpen hung in solver
  search (compiling winograd/implicit-gemm), no clean result. Killed it.

**Distinct from #3955:** that was a *SIGSEGV loading a precompiled kernel* (fixed by the
plugin bump). This doc was opened for a *source-compile error of the im2col HIP kernel* —
which, on re-verification, does not reproduce.
