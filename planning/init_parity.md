# Init parity — verified trainers vs their JAX references

**Opened 2026-10-01.** Status: AUDITED, nothing switched. One default-off flag (`dwFanK2`) written
and compiled, uncommitted. No run launched. Decisions for Brett are in §5.

## 0. Why this exists

The MobileNetV2 350-epoch pair landed with a steady gap: JAX `mnv2-full-jax-4gpu` 71.634 against
verified `mnv2-default-4gpu` ~71.1, about −0.5 from epoch ~126 on. The verified *train* loss is
+0.022–0.027 higher in every 25-epoch window from epoch 26, so the gap is in TRAINING, not eval.

Everything structural was checked and matches (`runs/2026-09-27-mnv2-verified-bf16-350ep`, audit
notes): the RMSProp update op for op, loss scale and all-reduce means, sync-BN forward AND
backward, XLA-SAME padding, BN ε, the LR staircase, the shim, dropout (the generator is
statistically clean), the θ|m|v slot threading. A one-step gradient tie on shared weights
(`tests/mnv2_bf16_grad_tie.py`, CPU) put verified-bf16 vs JAX-bf16 at its own noise floor.

What that test *cannot* see is where each run **starts**, because it feeds both sides the same θ.
The one input that differs by construction is the initial weights, and the curve fits an init
story: verified's train loss is **lower** for epochs 1–25 (−0.08), then higher from epoch 26 on.

`mkParam`'s docstring (`LeanMlir/Verified/Train.lean`) says it mirrors `jax/Jax/Codegen.lean`.
Until this audit nothing checked that per tensor, per net.

⚠ **Counter-evidence, recorded up front.** The OLD MNv2 pair (`content.tex` :7789–7790, 71.90 /
71.91) carried the SAME depthwise mismatch. Its verified arm LED early (51.36 vs 46.69 at e10) and
it ended tied. That pair differed in four other ways (per-replica BN, no dropout, ls 0.1, ReLU
stem/head), so it is confounded. But it means init is not shown to make the gap on its own; an
interaction with the newly matched recipe (sync-BN at 256, TF-slim BN, staircase) is as possible.
Only a matched-init run can settle it.

## 1. The audit

`tests/init_parity_audit.py` reads both sides from the code that runs:

* **JAX:** imports the generated reference and CALLS its own `init_params`, then measures each
  tensor's empirical mean and variance. Every emitter form is covered by measurement, not parsing.
* **verified:** the net's `VerifiedNetSpec.toSpecs` (dims + init kind, func-arg order), dumped from
  Lean, with `mkParam`'s variance rule applied, including the flags each driver sets
  (`cnxInit`, `vitInit`, `dwFanK2`).
* Tensors are aligned by order, with shape asserted (a dense may be stored transposed, and
  EfficientNet's SE FCs are rank-2 `[in,out]` on one side and `[out,in,1,1]` on the other).
* Only the first two moments are compared. Distribution (verified Bates-3, JAX one uniform) is a
  documented, deliberate gap.

Spec dump, scratch file, existing oleans, rebuilds nothing:

```lean
import LeanMlir.Verified.NetsCore
def dumpOne (s : VerifiedNetSpec) : String :=
  let items := s.toSpecs.toList.map (fun (d, k) => s!"[{d.toList}, {k}]")
  "{\"slug\": \"" ++ s.slug ++ "\", \"specs\": [" ++ ", ".intercalate items ++ "]}"
#eval IO.FS.writeFile "<out>/specs.jsonl" (String.intercalate "\n"
  [dumpOne resnet34ImagenetVerified, dumpOne resnet50Imagenet2018Verified, …])
```

then `JAX_PLATFORMS=cpu python3 tests/init_parity_audit.py specs.jsonl [--all]`. That takes
seconds on CPU.

### Results (2026-10-01)

| net (pair reference) | tensors that start differently | what |
|---|---|---|
| ResNet-34 (`resnet34_imagenet`) | **0 / 110** | ✅ |
| ConvNeXt-T (`convnext_tiny_imagenet_full`, cnxInit) | **0 / 182** | ✅ |
| **MobileNetV2** (`mobilenet_v2_imagenet_full`) | **17 / 158** | every depthwise: verified std **0.03–0.18×** |
| **MobileNetV4** (`mobilenet_v4_imagenet`) | **30 / 233** | every depthwise (3×3 and 5×5): **0.03–0.15×** |
| **EfficientNet-B0** (`efficientnet_b0_imagenet_full`) | **33 / 213** | 16 depthwise **0.03–0.18×** · 16 SE-reduce FCs **~0.2×** · 1 SE-expand 0.91× |
| ResNet-50, all recipes (`_2018`, `_rsbfaithful`) | 16 / 161 | last BN γ of every bottleneck: JAX **0** (zero-γ), verified **1** → ✅ **0 / 161** since init kind 4 (2026-10-05, `zeroGammaInit` on in `MainResnet50Imagenet`; `LEAN_MLIR_ZERO_GAMMA=0` restores γ = 1) |
| ViT-Ti (`vit_tiny_imagenet`, the `default` recipe = DeiT init, vitInit) | 2 / 200 | CLS token + position embedding: JAX σ 0.02, verified **0** → ✅ **0 / 200** since init kind 5 (2026-10-01, `planning/vit_parity_todo.md` P3) |

## 2. The mismatches, one by one

### 2a. Depthwise fan: MNv2, MNv4, ENet (63 tensors)

* **JAX** (`jax/Jax/Codegen.lean`, every depthwise emitter: `dwFanOut := 9`, `kSize * kSize`):
  `U(±√(6/k²))`, variance **2/k²**. This is TF's convention: a depthwise kernel `[k,k,C,1]` has
  fan_out = k²·1.
* **verified** (`mkParam`, rank-4 default): He fan-OUT **2/(dims[0]·k²) = 2/(C·k²)**. This is
  PyTorch's `_calculate_fan_in_and_fan_out`, which ignores `groups`. torchvision's MNv2
  (`kaiming_normal_(mode='fan_out')`) lands here too.
* So both are real conventions. The verified side is just not the one its reference uses.
  The std ratio is 1/√C: 0.18 at C=32, 0.03 at C=1152.
* **Why it can matter under BN:** BN normalizes the forward pass, but a BN-followed weight's
  gradient scales as 1/‖W‖. The optimizer then decides what that does to the *relative* step:

  | net | optimizer | relative step vs ‖W‖ | pair gap |
  |---|---|---|---|
  | MNv2 | TF-RMSProp, **ε = 1.0** (step ≈ g) | ∝ 1/‖W‖² | **−0.5** |
  | ENet-B0 | TF-RMSProp, ε = 1e-3 | Adam-like once the mean-square adapts; early steps ∝ 1/‖W‖² | −0.27 |
  | MNv4 | AdamW (step ≈ sign g) | ∝ 1/‖W‖ | tie |

  All three carry the same mismatch. The gap ranks with how scale-sensitive the optimizer is.
  That is suggestive at n = 3, not proof. ENet's gap was earlier attributed to EMA-over-BN
  buffers; this audit adds a second candidate.
* **The imprint is in the trajectory, not the endpoint.** At MNv2 epoch 279 the depthwise weight
  RMS has converged (verified/JAX 0.87–1.04, from 0.03–0.18 at init). Weight decay equilibrated
  the norms; the path there differed.

### 2b. EfficientNet SE FCs (17 tensors)

* **JAX** emits the SE squeeze/excite as 1×1 convs: reduce `U(±√(6/seMid))` (fan_out = the
  reduced width), expand `U(±√(6/mid))`.
* **verified** carries them as rank-2 dense, so `mkParam` gives Glorot **2/(in+out)**. For reduce
  that is ~0.2× std (e.g. `[480,20]`: 2/500 vs 2/20). For expand it is ~equal, except block 1
  (`[8,32]`, 0.91×).
* Unlike BN-followed convs, the SE FCs are **not** scale-invariant: they feed a sigmoid gate. A
  0.2× reduce FC starts every SE gate near σ(0) = 0.5, almost uniform. This changes the forward
  pass at init, not only the step size.

### 2c. ResNet-50 zero-γ (16 tensors)

* **JAX** (`emitConvBnInit … (zeroGamma := true)` on each bottleneck's last 1×1): γ = 0, so every
  residual branch starts as the identity. This is the torchvision `zero_init_residual` / RSB / timm
  standard.
* **verified**: kind 1 → γ = 1.
* Both R50 pairs **tied** anyway (2018 +0.12, A3 tie). Lower priority, but it is still a
  difference in the net's start, and the RSB-A3 recipe is explicitly written around zero-γ.

### 2d. ViT CLS token and position embedding (2 tensors)

* **JAX** (deit-init): σ = 0.02, timm `trunc_normal_`.
* **verified**: kind 2 → **0**. `vitInit` changes the weight rule, but these two are kind 2
  ("bias") in the ViT spec, so they never reach it. The `vitInit` docstring says *"the CLS token
  and positional embedding are already 0.02 on the JAX side"* without checking the verified side.
* The verified ViT run **beat** its reference by +0.04. Lowest priority.

## 3. What parity would look like

The principle is the one the padding work used: **the render (verified) side moves; the
references and their numbers stay.** The JAX references have landed, published numbers, and
re-running them to match verified would move every one.

### 3a. Mechanism: init kinds, not more booleans

`dwFanK2` (written, §4) is a boolean that keys off shape (`dims[1] = 1 ∧ dims[0] > 1`). That is
fine for depthwise, but it cannot express the other three:

* an SE FC and the classifier head are both rank-2 dense;
* a zero-γ and an ordinary γ are both rank-1 kind 1;
* a CLS token and a bias are both kind 2.

So parity wants the **init kind** to carry the role, set where the layer is known
(`VLayer.toSpecs`). Proposal:

| kind | meaning | value |
|---|---|---|
| 0 | weight, default rule | unchanged |
| 1 | BN/LN γ | 1 |
| 2 | bias / β | 0 |
| 3 | layer scale | 1e-6 |
| **4** | **zero-γ** (residual-closing BN) | **0** |
| **5** | **embedding** (CLS, pos) | **σ 0.02** |
| **6** | **depthwise kernel** | **2/k²** |
| **7** | **SE FC, conv-fan** | **2/out** (fan_out of the 1×1 it stands for) |

Kinds 4–7 are emitted only by the VLayers that own them, so every other net's spec is
byte-identical. Kind 6 retires `dwFanK2` and its shape heuristic.

⚠ **The specs are pinned.** `#guard <net>Verified.toSpecs == <Net>Layout.specs` (NetsCore.lean) for
R34/MNv2/ENet/ConvNeXt/ViT pins the derived list against an audited hand-list. A kind change moves
both, in lockstep, in one commit. That is the guard doing its job, not a cost to route around.

⚠ **Kinds are host-side only.** The render's signature carries dims, not kinds, so no
`verified_mlir/` artifact moves and no proof is touched. Confirm with `regen_verified_mlir.sh
check` after the change.

### 3b. Opt-in per net, like `cnxInit`

Each net's driver config gets an explicit `jaxInit := true` (or per-net flags) that turns its new
kinds on. Off by default, so **every landed verified number still reproduces from its seed**. The
ConvNeXt history (`runs/2026-09-17-cnx-verified-300ep/RESULTS.md` §7.0, a pair run that diverged
over init) is the precedent for making it a per-net, announced, logged switch.

### 3c. Gate: the audit becomes a ratchet

`tests/init_parity_audit.py` grows an expected-mismatch table (net → count + reason) and exits
non-zero if any pair has a mismatch NOT in the table, or if a listed one disappears without the
table changing. It belongs next to the convention audit (`planning/…convention audit`, which
already ratchets render-vs-reference divergences). After §3a lands, the table should read 0 for
every net whose flag is on.

## 4. What is already done (uncommitted)

* `VerifiedConfig.dwFanK2 : Bool := false` + `LEAN_MLIR_DW_FAN_K2=1` env override + the `mkParam`
  branch + a banner line (`▸ INIT: depthwise fan = k² …`). Only the main ImageNet driver path is
  plumbed. `lake build LeanMlir.Verified.Train` passes; the trainer exe was NOT rebuilt (the MNv2
  run was in flight).
* `tests/init_parity_audit.py`: the §1 audit.
* `tests/mnv2_bf16_grad_tie.py`: the one-step gradient tie (CPU result above; GPU arms owed).

## 5. Decisions for Brett

**Taken 2026-10-07 ("flip on the constructors"):** `dwFanK2 := true` in `mobilenetv2ImagenetConfig`,
`mnv4ImagenetConfig` and `efficientnetImagenetConfig` (verified moves, as recommended in 1.);
`LEAN_MLIR_DW_FAN_K2` gained a `=0` off-switch (the zero-γ pattern) so every landed run still
reproduces from its seed. The audit, re-run on the flipped flags: MNv2 0/158, MNv4 0/233,
EfficientNet-B0 17/213 — the 17 are §2b's SE FCs (`imagenet_parity.md` D5), the one depthwise-net
init item left. No run yet; the MNv2 / MNv4 `full` / B0 reruns carry it (`imagenet_parity.md` §7 R4/R5,
side_quest_runs.md batch 1).

**D5 taken the same day ("d5 please"):** §3a's kind 7, exactly as proposed — `VLayer.mbConvSE` /
`mbConvSENB` emit the two SE FCs at kind 7, `EfficientNetLayout.specs` moves in lockstep (the
`#guard` holds), `mkParam` gives kind 7 variance 2/out (`dims[1]` of the `[in,out]` dense, the
reference's `variance_scaling(2, fan_out)` on its 1×1 conv) under `VerifiedConfig.seFanOutInit`,
which the ImageNet B0 driver sets; off, kind 7 falls through to Glorot, so the Imagenette B0 and every
gate are byte-identical to their kind-0 past. `LEAN_MLIR_SE_FAN_OUT=0|1` overrides at launch.
`E4M3Quant.quantPackedParams` treats kind 7 as a weight (it keyed weights off kind 0). Kind 6
(depthwise) is NOT introduced: `dwFanK2`'s shape test covers those 63 tensors and is already on.
Expected audit after the rebuild: B0 0/213. No run yet.

1. **Which side moves.** Recommend verified → JAX, per the padding precedent.
2. **Kinds (§3a) vs flags.** Recommend kinds. `dwFanK2` alone covers 63 of 68 MNv2/MNv4/ENet
   tensors, but not ENet's SE, R50's zero-γ or ViT's embeddings.
3. **Which pairs get re-run, if any.** Cost on the 4× 3060 box (verified side only; references
   stay):

   | pair | current | mismatch | expected effect | re-run |
   |---|---|---|---|---|
   | MNv2 | −0.5 | 17 DW | largest (ε = 1) | ~56 h |
   | ENet-B0 | −0.27 | 16 DW + 17 SE | moderate, and SE moves the forward | ~73 h |
   | MNv4 | tie | 30 DW | small (AdamW) | ~18 h |
   | R50 2018 / A3 | tie / tie | 16 zero-γ | small | ~31 h / ~22 h |
   | ViT-Ti | +0.04 | 2 emb | negligible | ~48 h |

   A cheaper first step that is NOT a long run: an early-curve check (~30 epochs, ~4.8 h on MNv2)
   with the flag on. If init is the cause, verified's epoch 1–25 train-loss lead should vanish.
   Deferred 2026-10-01 ("no long test").

4. **Book.** Any re-run changes a `content.tex` number. Results/runs only until Brett says
   otherwise.

## 6. Known limits of this audit

* Moments only, not distributions (Bates-3 vs uniform stays a documented gap).
* It checks the reference module the pair's JAX run used (from `scripts/jobs/*jax*.conf`, or the
  run's RESULTS.md). If a historic JAX run trained an older emit, this audits today's emit, not
  that one (see the stale-emit history in memory/`stale-emit-ate-a-published-number`).
* R50's three recipes share one init (all show the same 16 zero-γ); one row covers them.
* It does not cover stochastic-depth/dropout seeds, which are per-step inputs, not init.
