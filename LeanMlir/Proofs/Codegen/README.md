# Codegen/ — the emitted-graph AST, its semantics, and the artifact writers

Every file in `verified_mlir/` is written by an `#eval` in one of the `*Render*` files here, and
its text is `pretty` of an `SHlo` term. The theorems about what that term *means* are stated about
`den`, mostly in `Nets/`.

| file | role |
|---|---|
| `StableHLO.lean` | the `SHlo` AST, `den` (its ℝ semantics), the per-op `*_faithful` lemmas, the printer (`skel` → `Tok` → `emitTok` → `pretty`), the chapter 1–3 graphs. Header has a table of contents |
| `StableHLOLex.lean`, `StableHLOParse.lean` | the syntactic round-trip (`parse (lex (pretty g)) = some (skel g)`) |
| `SyncBnSites.lean` | the one writer of the sync-BatchNorm text, shared by every net's data-parallel render |
| `RenderKit.lean` | the renderers' shared optimizer tail: `PGrad` and the per-parameter steps `adamOne`, `rmsOne`, `adamOneEma` (ResNet's multi-optimizer `optOne` stays in `ResNet34RenderB`) |
| `MlpRender`, `CnnRender` | chapter 2–4 train steps (MLP, MNIST CNN, CIFAR, the cifar8 family) |
| `MlpArtifacts`, `CnnArtifacts` | their `#eval` artifact writers — leaf modules, imported by nothing, so building a proof never rewrites `verified_mlir/` |
| `ResNet34RenderB`, `ResNet50RenderB`, `MobileNetV2RenderB`, `MobileNetV4RenderB`, `EfficientNetRender`, `ConvNeXtRender(B)`, `ViTRender(B)` | per-net ImageNet/Imagenette train steps (batched index, batch BN) |
| `EfficientNetRenderPC`, `EfficientNetRenderPCEval` | EfficientNet's batched block forwards + typed graphs, and their eval-mode twins — ⚠ they write no artifact (the batched stages other nets use, `cbsB`, `projB`, …, are in `Foundation/BatchedStages`; MobileNetV2's per-channel stages are `Nets/MobileNet/MobileNetV2StagesPC`) |
| `IRPrint.lean` | a scratch-only execution oracle for the small-net `IR`, not an artifact writer |
| `LambTriple.lean` | the LAMB `(θ', m', v')` triple's faithfulness — the ℝ optimizer specs the optimizer ops denote are in `Training/Optim/` (`AdamStep`, `SgdMomentumStep`, `RmsPropStep`, `Lamb`, `GradClip`), drop-path in `Training/DropPath` |

## From a net to its theorems

| net | renderer | T2: forward graph = forward | T3: train step = certified step | T6: backward graph = VJP |
|---|---|---|---|---|
| ResNet-34 | `ResNet34RenderB` | `resnet34FwdGraphB_full_faithful` | `r34_net_tiedB` (sync-BN DP: `r34_net_syncTiedB`) | `r34InputGradB_eq_r34B_full_vjp` |
| ResNet-50 | `ResNet50RenderB` | `resnet50FwdGraphB_full_faithful` | `r50_net_tiedB` | `r50InputGradB_eq_r34B_full_vjp` |
| MobileNetV2 | `MobileNetV2RenderB` | `mobilenetv2FwdGraphB_full_faithful` | `mnv2_net_tiedB` | `mnv2InputGradB_eq_mobilenetv2B_full_vjp` |
| MobileNetV4 | `MobileNetV4RenderB` | `mnv4FwdGraphB_full_faithful` | `mnv4_net_tiedB` | `mnv4InputGradB_eq_mnv4B_full_vjp` |
| EfficientNet-B0 | `EfficientNetRender` | `efficientnetFwdGraphB_full_faithful` | `efficientnet_net_tiedG` | `efficientnetInputGradB_full_eq_efficientnetForwardB_full_vjp` |
| ConvNeXt-T | `ConvNeXtRenderB` | `convNextFwdGraphTCh_faithful` | `cnx_net_tiedGB` | `convnextInputGradB_eq_convNextForwardTChB_vjp` |
| ViT-Tiny | `ViTRenderB` | `vitFwdGraphKMHV_faithful` | `vit_net_tiedGB` | `vitInputGradKB_eq_vitKVB_vjp` |

The emitted forward and the T2 graph are separate definitions: the renderer builds its forward
inline, and the T2 theorem is about `*FwdGraph*` at the renderer's tokens. The CI drift guard
(`proofs.yml`) re-elaborates every renderer and byte-checks `verified_mlir/`.

## `SHlo` constructor suffixes

`F` — a forward op, but also the optimizer ops (`adamMNextF`, …). `B` and `Batched` — two
spellings of the batched index. `Grad` / `GradB` — the raw gradient node every optimizer tail
consumes. `Sgd` / `SgdB` — the fused `θ − lr·g` node (SGD-inline renders only). `Bf16` / `F8` —
reduced-precision twins (their nodes are folded in `Foundation/Bf16GradNodes.lean`). `Xla` —
XLA-`SAME` (asymmetric) padding. Suffixes stack: `convWeightGradBBf16`.
