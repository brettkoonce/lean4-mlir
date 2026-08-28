import LeanMlir.VerifiedNets

/-! # `vit-b-imagenet-verified` — ViT-**Base** (DeiT-B) on full ImageNet-1k, verified → XLA

The third width off one renderer, and the widest. ViT-B is ViT-Ti widened twice over:
`D = 768 = 12 heads × 64` against Small's `384 = 6 × 64` and Tiny's `192 = 3 × 64`, MLP 3072
against 1536 and 768, and everything else identical — same depth 12, same 16×16 patch grid, same
block structure, same drop-path ramp. 86,567,656 parameters — DeiT-B's published 86.57M — in the
SAME 200 tensors as Ti and S.

⭐⭐ **The proof side needed nothing.** `Proofs.vitForwardKV_has_vjp` is already
`∀ heads d_head mlpDim k`, and it is a GLOBAL `HasVJP` rather than the pointwise `_at` form the
relu-family nets carry, because GELU/softmax/LayerNorm have no kink. The same theorem covers Tiny,
Small and Base at different arguments. What changed was the RENDERER: `ViTRenderB.lean`'s six
`private def` width constants became a `VitDims` record threaded as a trailing defaulted
parameter, so one renderer serves all three sizes. Every ViT-Tiny artifact re-renders byte-identical,
which is the gate that says the parameterisation was inert.

⭐⭐ **GLOBAL 512 — DeiT's OWN BATCH — REACHES THIS BOX, AND THE ACCUMULATION LOOP IS NOT NEEDED**
(2026-08-27, `runs/2026-08-27-vitb-global512/`). This paragraph used to say the opposite, in these
words: *"ViT-B at 4×128 was rendered and run: it OOMs, asking 11.90 GiB against the 11.68 GiB the
BFC allocator gets on a 16 GB card … Reaching the recipe's global 512 needs GRADIENT ACCUMULATION."*

⚠⚠ **The OOM was real and its explanation was not.** 11.68 GiB is NOT what "the BFC allocator gets
on a 16 GB card" — it is what it gets at the CUDA plugin's `memory_fraction = 0.75` DEFAULT, which
the verified path had no way to raise until `ffi/pjrt_ffi.c` began passing create options.
`LEAN_MLIR_MEM_FRACTION=0.97` gives **15.11 GiB**, and `vitbin_adamdp128x4wxclipdrop` then executes
on four cards at **13.99 GiB, 93 %** of it. The reproduction is one environment variable: unset,
the same bytes on the same four devices still die *"Out of memory while trying to allocate
11.96GiB"*.

  | variant                        | per-dev | global | peak    | 11.68 GiB | 15.11 GiB |
  |--------------------------------|---------|--------|---------|-----------|-----------|
  | `adamdp128x4wxclipdrop`        |     128 |  **512** | 13.99 G | ⛔ OOM    | ✅ 93 %   |
  | `adamdp128x4wxclipdropbf16`    |     128 |  **512** | 12.61 G | ✅ 93 %   | ✅ 83 %   |

⭐ **And it is FASTER, not merely more faithful.** Trainer ms/step over 40 steps on four 4060 Ti,
2,502 steps/epoch at global 512 against 10,009 at 128 — **291 h fp32 and 178 h bf16** for 300
epochs, against 322 and 228 at the old batch. Same per-invoke-overhead amortisation R50's `4×128`
showed over `8×64`: a quarter of the steps, each less than four times the cost.

▶ **What this does to the LR paragraph below.** `baseLR = 5e-4` is DeiT's **batch-512** rate. At
32×4 it was the reference's number applied to a batch four times too small — a deviation the old
text recorded as "the LR here is the batch-512 rate, un-rescaled". At 128×4 it is simply correct,
and nothing needs rescaling. So global 512 removes BOTH halves of "no run of this driver is
comparable to DeiT-B's 81.8%".

⚠ **Only 4-replica variants are rendered for the DeiT batches**, so unlike the other ImageNet
drivers this one has no single-device default to fall back to. (`vitbin_adam128wxclipdrop` and its
bf16 peer exist at 1 replica, but they are a memory CONTROL — a graph with no collective in it, so
the all-reduce could be priced by difference — and they render global 128, not 512.)

⚠ ViT has no BatchNorm, so there is no running-stats eval forward. `vitbin_drop_fwd` plus the
train step is the complete artifact set for this net.

⚠ **Nothing has been trained.** The artifacts render, the shapes tie to `VLayer.toSpecs`, and the
parameter count is `#guard`ed. No accuracy has been measured and no wall clock has been probed.

⭐ **Run it through the job config**, which is where the required options live and which refuses
to start without them:
```
scripts/supervise.sh vitb-default-g512-4gpu
DRY_RUN=1 scripts/supervise.sh vitb-default-g512-4gpu   # print the plan, run nothing
```

By hand, at DeiT's global 512 (4 GPUs — and BOTH replica knobs are required, see
`planning/mnv4_convm_ties_todo.md`):
```
CUDA_VISIBLE_DEVICES=0,2,3,4 PJRT_REPLICAS=4 LEAN_MLIR_REPLICAS=4 \
  LEAN_MLIR_MEM_FRACTION=0.97 \
  LEAN_MLIR_VARIANT=adamdp128x4wxclipdrop LEAN_MLIR_BATCH=128 \
  PJRT_FFI_RESIDENT=1 SHIM_WORKERS=8 \
  .lake/build/bin/vit-b-imagenet-verified data
```
⚠⚠ `LEAN_MLIR_MEM_FRACTION` is not optional in that command and its absence is not a slowdown —
the first step raises `RESOURCE_EXHAUSTED` on all four devices. ⚠ And it only does anything if
`ffi/libpjrt_ffi.so` was rebuilt after the allocator-options change: that shim is **not a lake
target**, so `lake build` reports SUCCESS without recompiling it and the variable is then read by
nothing. `strings ffi/libpjrt_ffi.so | grep memory_fraction` is the check; the job config runs it.

⚠⚠ **THE `32×4` PAIR IS GONE** (2026-08-27). Global 512 is DeiT's batch, it is faster per epoch,
and it makes the LR below correct — `32×4` lost on every axis it had, exactly as R50's `8×64` did.
The renders are deleted; the measurements that licensed the deletion are in
`runs/2026-08-27-vitb-global512/`. ▶ So there is no small-batch fallback and the allocator option
is not optional for THIS net: `LEAN_MLIR_MEM_FRACTION` is now checked before the first step, and
`runViTBImagenet` refuses with a sentence rather than letting XLA raise `RESOURCE_EXHAUSTED`.
-/

/-- 300 epochs and 128 per device — the DeiT schedule length at DeiT's own batch, since four
    replicas of 128 is the reference's global 512. ⭐ That is what makes `baseLR` below correct
    rather than merely inherited: 5e-4 is the batch-512 rate, and this is batch 512. -/
def vitBImagenetConfig : VerifiedConfig where
  epochs    := 300
  batchSize := 128

/-- Entry point. ⚠ Defaults to the FOUR-REPLICA variant, unlike every other ImageNet driver here,
    because it is the only one rendered for this net. A plain invocation therefore needs
    `PJRT_REPLICAS=4` and `LEAN_MLIR_REPLICAS=4` or it fails at the first step on a replica-count
    refusal. That is the honest failure: the alternative is a single-device default that names an
    artifact which does not exist.

    ⚠⚠ **AND IT REFUSES WITHOUT `LEAN_MLIR_MEM_FRACTION`.** This is the only driver in the tree
    that does, because it is the only one whose sole fp32 render does not fit in the allocator's
    DEFAULT arena: the graph peaks at 13.99 GiB and PJRT's CUDA plugin hands out 11.68 unless
    asked. ▶ The refusal exists because the failure it replaces is *misleading*, not merely
    unhelpful — `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 11.96GiB` reads as
    "this card is too small for ViT-B", and that reading is precisely what pinned this net at a
    quarter of its recipe's batch for a month. A sentence naming the variable costs one `getEnv`.

    ⚠ The bf16 twin DOES fit at the default (10.88 GiB, measured), so the check exempts it rather
    than refusing a run that would have worked. ⚠ And the option only does anything if
    `ffi/libpjrt_ffi.so` was rebuilt after the allocator-options change — that shim is not a lake
    target, so `lake build` reports SUCCESS without recompiling it. This driver cannot see that
    from Lean; `scripts/jobs/vitb-default-g512-4gpu.conf` greps the binary for the string. -/
def runViTBImagenet (argv : List String) : IO Unit := do
  let variant := (← IO.getEnv "LEAN_MLIR_VARIANT").getD "adamdp128x4wxclipdrop"
  -- ⚠ `bf16` is a SUFFIX test and not a substring one: every variant this net renders ends in the
  -- marker or does not carry it, and a substring test would exempt a hypothetical `bf16`-prefixed
  -- spelling that is still fp32 in its optimizer tail. `TestVariantPredicates`' own lesson.
  if !(variant.endsWith "bf16") then
    if (← IO.getEnv "LEAN_MLIR_MEM_FRACTION").isNone then
      throw <| IO.userError <|
        "LEAN_MLIR_MEM_FRACTION is unset, and this net's fp32 render does not fit without it.
" ++
        "  The graph peaks at 13.99 GiB. PJRT's CUDA plugin defaults its BFC arena to
" ++
        "  memory_fraction = 0.75, i.e. 11.68 GiB of a 16 GB card, and the first step would
" ++
        "  die RESOURCE_EXHAUSTED on every device — which reads as a card being too small.
" ++
        "  It is not. Set LEAN_MLIR_MEM_FRACTION=0.97 (15.11 GiB; the graph then uses 93%).
" ++
        "  Or run the bf16 twin, which fits at the default:
" ++
        "    LEAN_MLIR_VARIANT=adamdp128x4wxclipdropbf16
" ++
        "  Or use the job config, which sets both this and the shim check:
" ++
        "    scripts/supervise.sh vitb-default-g512-4gpu"
  let bs := ((← IO.getEnv "LEAN_MLIR_BATCH").bind (·.toNat?)).getD vitBImagenetConfig.batchSize
  let baseLR := match (← IO.getEnv "LEAN_MLIR_BASE_LR_U").bind (·.toNat?) with
    | some u => u.toFloat * 1e-6
    | none   => 0.0005   -- the DeiT batch-512 rate, as the Tiny driver uses. ⚠ NOT retuned for S:
                         -- DeiT-S uses the same 5e-4 at batch 512, so this matches the reference
                         -- rather than being an untuned carry-over.
  let epochs := ((← IO.getEnv "LEAN_MLIR_EPOCHS").bind (·.toNat?)).getD vitBImagenetConfig.epochs
  vitBImagenetVerified.toNet.trainAdamSched
    { vitBImagenetConfig with batchSize := bs, epochs := epochs }
    (argv.head?.getD "data") baseLR 0.9 0.999 5 variant

def main (argv : List String) : IO Unit := runViTBImagenet argv
