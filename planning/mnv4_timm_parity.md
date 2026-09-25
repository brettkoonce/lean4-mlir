# MobileNetV4-Conv-M → timm's `mobilenetv4_conv_medium`

Correctness audit 2026-09-24, finding 4: the net the repo calls MobileNetV4-Conv-M is not timm 1.0.28's
`mobilenetv4_conv_medium`. The block table agrees on `(ic, oc, expand, preDWk, postDWk, h, stride2)`,
and so does the parameter count (9,715,512), but five things that change the function do not.
Owner decisions (2026-09-24): make timm the reference, keep 224 px, replace the old variant (no
archive copy), and queue the 100-epoch ImageNet rerun for later.

## The five deviations (timm walked in `.venv-timm`)

| # | site | timm | repo today |
|---|---|---|---|
| D1 | UIB stride carrier | `dw_mid` when both DWs exist, else `dw_start` | always the pre-DW (`# stride consumed by pre-DW`) |
| D2 | UIB `dw_start` activation | none (`BatchNormAct2d` act = Identity) | relu |
| D3 | head order | GAP → `conv_head` 960→1280 → `norm_head` BN → ReLU → Linear | conv 960→1280 → BN → relu at 7×7 → GAP → dense |
| D4 | stage 0 `EdgeResidual` activation | ReLU | swish |
| D5 | stride-2 padding (stem; fused conv) | symmetric `(k-1)//2` | stem XLA-`SAME` (0,1); fused already symmetric |

All three strided rows (1, 3, 11) have both DWs, so D1 moves each of their strides onto the
post-DW at the EXPANDED width, and the expand 1×1 runs at the input resolution. D2 removes 17 relu
sites (one per present pre-DW). D3's BN normalises the pooled `[N, 1280, 1, 1]` over N only.
Unchanged: channel table, expand ratios, kernels, skips, BN ε = 1e-5, no conv biases, the
`256→960` 1×1 conv-BN-ReLU at 7×7 before the pool.

## Phases

**P0: pin timm properly.** `scripts/mnv4_timm_spec.py` (`.venv-timm`) dumps, for every layer, the
kernel size, stride, padding, groups, and whether an activation is present, plus the
stem/stage-0/head order. The Lean `#guard`s read that tuple, not a `stride2` Bool. A numeric gate,
`scripts/parity/mnv4_timm_parity.py`, copies a random-init timm `state_dict` into the JAX param layout and
compares logits in train-mode BN (and eval mode with running stats) to ≤1e-4. This is the check
that makes "timm is the spec" mean something, since a structural dump cannot see the op order
inside a block.

**P1: JAX reference.**
- `Layer.fusedMbConv` gains `(act := .swish)`; EfficientNetV2 is unchanged.
- `uib_block` (running and non-running forms) moves the stride per D1 and drops the pre-DW relu (D2).
- `MainMobilenetV4{,Imagenet}.lean`: `.fusedMbConv … (act := .relu)`, `convPadStyle := .symmetric`,
  and the head reordered to GAP before `conv_head`.
- The BN-stat slot order in the running-BN path follows the new op order.
- Regenerate `jax/generated/generated_mobilenet_v4*.py`, and check `VjpOracleNets`' `.uib` net.

**P2: verified side.**
- `VerifiedNetsCore` `mobilenetv4Verified` / `mnv4in`, with the `bnChannels` order.
- `MobileNetV4Spec` (stride-carrier rule, guards from P0).
- `MobileNetV4RenderB` forward and backward: pre-DW BN without relu; strided symmetric depthwise at
  the post-DW; fused relu (`selectPosB`, not `swishBackB`); symmetric stem; head
  GAP → conv → BN(N) → relu → dense.
- The bf16 twins.
- Regenerate the nine `mnv4*.mlir`, then run `mnv4_forward_tie.py` and `grad_tie.py --net mnv4`
  against the P1 reference on shared weights.

**P3: proofs.** In order: `MobileNetV4BackB0`, `FullB` / `FullBVJP`, `StepTieB`, `WholeBackCertifiedTieB`,
`SyncB`, `SyncStepTieB`, `FullBSeal`, then the Float bridge, the comparator tier, `AuditAxioms`,
`formalization.yaml` and the blueprint (the "fifty-four clauses" text). D4 removes swish from the
net entirely, which should simplify the seal: the grid-constant `swishGap` base existed only
because of that site. D3 changes the carrier's path through the head.

**P4: docs + rerun (queued).** Book ch 6's 75.51% and its curve become the pre-fix variant until
the 100-epoch rerun lands (~15 h, 4 cards). Mark them as such, and don't delete them until the
replacement exists.

## Status
- [x] P0: `scripts/parity/mnv4_timm_parity.py` (+ `_mnv4_timm_dump.py`). Train 1.5e-5 / 1.8e-5, eval 6.5e-7;
  the pre-fix reference scores 0.69. The structural dump was not built, since the numeric gate
  subsumes it; the stride-carrier rule is a renderer `#guard`.
- [x] P1: JAX refs regenerated; vjp_oracle `uib` passes.
- [x] P2: nine `mnv4*.mlir` re-rendered (byte-reproducible). `mnv4_forward_tie.py` 1.8e-5.
  `grad_tie.py --net mnv4` now runs at B=8 (at B=2 the 1×1 head BN is batch-degenerate) and passes
  `--nokink`. `%h1bt`/`%hW` are exempt there: rank-1 head-BN cancellation, precision-limited;
  default mode covers them.
- [x] P3: BackB0 → Seal all rebuilt. The seal is now 38 clauses, with no swish machinery (kit §13–14,
  `hasDerivAt_mul_of_zero`, `bnIstd_le_one`, `swishScalar_lt`/`Deriv_pos` deleted). AuditAxioms
  pins renamed. Comparator, docstring-checkrefs and blueprint `\uses` are green.
- [ ] P4: docs marked (book §`mnv4_side_quest`, README). The 100-epoch rerun is QUEUED, not launched.
