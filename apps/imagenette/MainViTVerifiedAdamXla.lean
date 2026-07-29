import apps.imagenette.ViTAdamCommon

/-! # `vit-verified-adam-xla` — the ViT-Tiny AdamW trainer on XLA/PJRT

The same program as `vit-verified-adam` (shared body in `apps/imagenette/ViTAdamCommon.lean`),
linked against `ffi/libpjrt_ffi.so`.

## ⛔ This binary DOES NOT RUN on this box, and it compiling is not evidence that it does

The plumbing is here because it is 30 lines and because the DP render (`LEAN_MLIR_VARIANT=adamdp`)
is unreachable without it — collectives live only on the PJRT path. But every ViT graph carrying a
backward dies at **execution** on this box with `miopenStatusUnknownError`, *"Failed to enqueue
convolution on stream"*, in the patch-embed weight-gradient convolution. It **compiles fine** — XLA
accepts the module in ~29 s with both devices visible — so a successful `lake build` and a
successful compile say nothing about whether it works. Handoff §2h: **do not describe a ViT `-xla`
target as working on the strength of it compiling.**

What is established about the blocker, so nobody re-does it:

* **Not the collective and not the replica count.** It dies on the single-device step, before the DP
  invoke is reached. Two GPUs cannot help.
* **Not AdamW.** Measured 2026-07-28: the **SGD** `@vit_train_step` fails identically. It is in
  every ViT graph with a backward, since both renders contain the same two convolutions — which
  makes a minimal repro much cheaper, starting from `vit_train_step.mlir` (202 in / 200 out, two
  convolutions) rather than from anything AdamW-shaped.
* **Not the shape.** A minimal JAX repro of that exact `(3,32,224,224)×(192,32,209,209) →
  (3,192,16,16)` convolution (`scratchpad/miopen_repro.py`) runs fine and returns the right shape.
* **Not memory.** `scripts/miopen_mem_probe.py` succeeds with 12 GiB of ballast on top of its own
  1.02 GiB, and a genuine OOM gives a clean `RESOURCE_EXHAUSTED` instead.
* **Not strided weight gradients as a class.** ConvNeXt's structurally similar 4×4/s4 patchify weight
  gradient runs fine on this box (§2h), which narrows it to ViT's one 16×16/s16 shape.
* **Leading unconfirmed hypothesis:** XLA fuses the `stablehlo.pad` (`interior = [0,0,15,15]`) into
  the convolution as `rhs_dilation = 16`, a different MIOpen call than either probe makes. To settle
  it, dump post-optimisation HLO and read the `convolution` descriptor — **`XLA_FLAGS` is inert on
  this path**, so use §4's regenerate-a-throwaway-shim recipe.

**No bug has been filed and none should be** until there is a self-contained repro; the isolated
convolution does not reproduce it, so a report today would point maintainers at the wrong thing.

The same graph runs fine under IREE, which localises this to the XLA/MIOpen path rather than to the
render. So: use `vit-verified-adam` here, and run this one on ares.

**This does not move the verification tier** (§9).

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build vit-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/vit-verified-adam-xla data   # ⛔ miopenStatusUnknownError here
```
-/

def main (argv : List String) : IO Unit := runViTAdam argv
