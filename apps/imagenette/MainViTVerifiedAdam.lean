import apps.imagenette.ViTAdamCommon

/-! # `vit-verified-adam` — train ViT-Tiny with the VERIFIED-rendered **AdamW** step

Phase 3c of `planning/vit_train_to_vit_verified.md`: the SGD `vit-verified` with its optimizer
swapped for AdamW. The packed train step `@vit_adam_train_step` is `pretty(provenGraph)` out of
`LeanMlir/Proofs/Codegen/ViTRender.lean`'s `vitAdamTrainStepFaithful` — gradients un-fused and
handed to the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple — then driven by
`VerifiedNet.trainAdamSched`, which threads `[θ|m|v]` as a single packed param blob plus the runtime
`lr`/`bc₁`/`bc₂` scalars through the generic FFI (`n_params = 3k`; the moments ride in the params
slot, so the prebuilt `.so` is unchanged).

**The artifact became `pretty(provenGraph)` on 2026-07-28** (handoff §2a). Before that this driver
emitted the hand-written render itself at startup; the swap was licensed by `vit-adam-tie` —
gradient norm-rel 1e-6, `%loss` **bit-exact**, 0/200 params disagreeing, on all 16,579,041 returned
floats. `%loss` is the load-bearing check on this net rather than a footnote: ViT has no BN, so
nothing else in the output depends on the forward alone.

Recipe matches `MainVitTrain.lean`'s `vitTinyConfig`: AdamW lr 3e-4 / wd 1e-4, cosine + 5-epoch
warmup, label smoothing 0.1, augment, 80 epochs, bs 32 — the schedule differs from the other four
nets, which is why the shared body keeps each net's own numbers.

Run (GPU): `IREE_BACKEND=rocm .lake/build/bin/vit-verified-adam data`. ⛔ The XLA peer
`vit-verified-adam-xla` exists but **does not run on this box** — the graph dies in MIOpen; see that
file. The shared body is `apps/imagenette/ViTAdamCommon.lean`.
-/

def main (argv : List String) : IO Unit := runViTAdam argv
