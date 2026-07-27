# xla_pjrt_ladder.md — Lean4 → StableHLO → PJRT → XLA → GPU

**Written 2026-07-27.** A validation ladder for running the **already-emitted,
already-proof-tied** StableHLO through XLA instead of IREE. Companion to
`visdrone_detector.md` (the demo this unblocks) and memory
`rocm-is-the-transformer-box` (the IREE measurements that motivated it).

---

## 0. Why — the measurement that started this

IREE's convolution codegen was measured at **~1% of hardware peak** on both an RX
7900 XTX and a Jetson Orin, against 24–47% for a tuned vendor library. It is not a
layout problem (IREE already emits `linalg.conv_2d_nhwc_hwcf`), not a tuning-flag
problem (igemm / tile-and-fuse / igemm-pad all within 1 ms), not graph structure
(an isolated hand-written 3×3 conv shows the same 121× gap), and not
target-specific.

**XLA compiles the identical Lean-emitted files and is 20–40× faster,** because
XLA:GPU dispatches convolutions to MIOpen/cuDNN.

| graph (unchanged Lean output) | IREE | XLA | speedup |
|---|---|---|---|
| `resnet34_fwd` (b32 @224) | 292 ms | **7.4 ms** | 39.7× |
| `resnet34_adam_train_step` | 1237 ms | **61.3 ms** | 20.2× |
| FPN detector train step (b8 @448) | 1650 ms | **46.5 ms** | **35.5×** |

At 46.5 ms/step a 12-epoch VisDrone run is **7.5 minutes instead of 4.4 hours**.
That is the entire reason this doc exists: it changes how fast every other
experiment in the repo can be run.

**This is not a new codegen path.** Same `NetSpec`, same emitter, same
`verified_mlir/*.mlir` artifacts, same §1a ties. Only the consumer changes — from
one unverified trusted lowerer (IREE) to another (XLA). Nothing about the
verification tier moves. Do not describe this as "more" or "less" verified.

---

## 1. What is established vs assumed

**✅ Established by measurement (2026-07-27):**
- XLA compiles Lean-emitted StableHLO with **one edit**: the entry function must be
  renamed to `@main`. Nothing else.
- `resnet34_fwd` under XLA vs IREE on identical inputs: **rel 8.4e-06** — the two
  backends compute the same function.
- The R34 Adam train step and the FPN detector train step both compile and execute
  under XLA at the speeds above.

**✅ Established by measurement (2026-07-27, later) — rung 0 complete, no Python:**
- **PJRT is reachable from C**, so the host loop stays in Lean. The JAX-shipped
  plugin exports `GetPjrtApi`; `PJRT_Program.format = "mlir"` takes the StableHLO
  **text directly** (no protobuf for the program). See §8.
- **All four gates pass on `linear_train_step`** — details in §8. G2 is the one that
  did not exist before: after 5,616 steps the two backends' parameters agree to
  **rel L2 5.3e-07**.

**✅ Established by measurement — rungs 1, 1b, 2 (2026-07-27):**
- Depth, multiple params, He init, **convolutions**, **Adam moments**, and
  **rank-0 runtime scalars** all work, with **no new shim code** at any rung: every
  one of them rides through `iree_ffi_invoke_f32`. Details and gate results in §8.
- **G2 had to be re-specified twice** — once for step count (§3, rung 1) and once
  to measure the gradient rather than θ on Adam nets (§8, rung 2). Both times the
  original wording would have condemned a correct backend.

**✅ Established by measurement — rung 3, R34 on Imagenette (2026-07-27):**
- **The speedup is real at scale: 10.5× steady state** (1,702 → 162 ms/step after
  the memcpy fix), 52.5 s/epoch vs IREE's ~549. 80-epoch projection ~12.2 h → ~1.2 h.
- **The verified path is within 1.04× of hand-written JAX per step** (162 vs 156 ms,
  matched bs32) — see §8.
- **BN running stats thread fine, and `train_step_adam*` was never needed.**
  `trainAdamSched` packs them into the params blob, so all 513 in / 513 out ride
  `iree_ffi_invoke_f32`. **No rung has required new shim code.**
- **The R34 loop converges from Lean** — 47.3% val at epoch 1. §4's init blocker is
  a property of the Python prototype, not the path: `mkParam` owns init.

- **R34's gradients are correct** (§8). The raw G2 number (5.20e-02) looked
  alarming, but the gate was mis-specified: R34's gradient does not reproduce to
  better than ~6e-3 *against the same backend* under a sub-ULP nudge, so a 1e-4
  threshold fails XLA-vs-XLA by 60×. Correctness is established at layer
  granularity instead — all 6 R34 layer families tie to JAX autodiff at ≤6.4e-06
  via `tests/vjp_oracle/run.sh`. No defect.

**⚠️ Assumed, NOT yet established:**
- **Rung 4** (the FPN detector) is untouched. It is unblocked, and the 35.5×
  headline is the one still unverified end-to-end.
- **Whether anything material is left in the step.** Two rounds are done: batching
  the PJRT transfers (256 → 205 ms) and removing the per-step host memcpys
  (205 → 162 ms). At 162 vs native JAX's 156 the remaining headroom against a
  hand-written baseline is ~4%; device-resident parameters would target that, but
  the payoff is now small. **Do not expect the graph-level 20.2× end to end** —
  that number excluded the data pipeline, which JAX also pays.
- **That an executable cache is worth it.** `PJRT_Executable_Serialize` /
  `DeserializeAndLoad` exist; the JIT tax is 5–14 s per process (§8). Matters for
  short experiments, rounds to nothing on a multi-hour run.

---

## 2. The ladder

Climb in order. **The rung where it breaks is the signal** — do not skip ahead to
the detector because the small nets are "obviously fine".

| rung | graph | inputs → outputs | what it introduces | state |
|---|---|---|---|---|
| **0** | `linear_train_step` | 4 → 2 | the bare loop: x, W, b, onehot. Hand-auditable. | **✅ done** (§8) |
| **1** | `mlp_train_step` | 8 → 6 | depth, multiple params, **He init** | **✅ done** (§8) |
| **1b** | `cnn_train_step` | 12 → 10 | **convolutions** + maxpool, 4-D params | **✅ done** (§8) |
| **2** | `cifar8_bn_adam_train_step` | 117 → 117 | Adam moments + **rank-0 scalars** | **✅ done** (§8) |
| **3** | `resnet34_adam_train_step` | 513 → 513 | full scale; **BN running stats**; the 20× number | **✅ done** — 10.5×, ~JAX parity, VJPs verified (§8) |
| **4** | FPN detector train step | 355 → — | the payoff; the 35× number | unblocked — next |

Rung 0 is deliberately trivial: `linear_train_step` has no Adam moments, no BN
stats, no bias correction. If the plumbing is wrong there, it is wrong in a form
you can read off the screen. Every subsequent rung adds exactly one class of state.

---

## 3. The gates — non-negotiable, and they are the point

Today's session produced a driver that ran at full speed, updated parameters, and
learned **nothing**, because 295 of 513 outputs were silently discarded. It looked
identical to success. Every rung must pass all four:

- **G1 — forward tie.** XLA output vs IREE output on byte-identical inputs,
  relative error < 1e-4. (Done for `resnet34_fwd` at 8.4e-06.)
- **G2 — step tie.** From byte-identical initial params, run **N steps under both
  IREE and XLA** and diff every returned tensor. This is the real gate. Without it,
  "XLA trains our net" is unproven. Dump params with `LEAN_MLIR_DUMP_PARAMS=<path>`
  on both builds and diff; equal accuracy is a summary statistic, not a tie.

  **N must be small — this was underspecified and rung 1 exposed it.** Any net with
  ReLU is chaotic: a branch flip caused by a 1-ULP difference changes the gradient
  materially, and the gap compounds. Measured on the MNIST MLP:

  | steps | rel L2 | params still bit-identical |
  |---|---|---|
  | 1 | 5.82e-09 | 97.6% |
  | 10 | 1.86e-08 | 88.6% |
  | 100 | 7.74e-08 | 58.2% |
  | 5,616 (12 ep) | **1.32e-02** | 6.8% |

  Through 100 steps the difference sits at the f32 rounding floor; the blow-up is
  later and super-linear. So **G2 is a 1–100 step gate**, and a full-run diff proves
  nothing either way. Cap steps with `LEAN_MLIR_MAX_STEPS`. Long-run agreement is
  what **G3** is for, and it should be judged on accuracy, not parameters.

  **Run the determinism control before blaming reassociation — and run it at the
  scale you care about.** XLA is bit-identical run-to-run *for a single step*, and
  that does **not** extend to a full epoch. Measured on R34, three runs of the same
  binary at the same seed gave epoch-1 val accuracy of **43.21% / 46.80% / 47.29%**
  — a ~4-point spread from nothing but run-to-run nondeterminism (GPU reduction
  order / autotuned kernel selection). A one-step determinism check is necessary,
  not sufficient. Establish a backend's own variance before attributing any
  cross-backend difference to the backend.
- **G3 — it learns.** Train loss falls and val accuracy rises above chance. A speed
  number from a loop that does not converge is worthless.
- **G4 — no dropped state.** Assert **every** return value maps to an input buffer;
  raise on any unmapped name. Do not `continue` past an unrecognised output.

G4 is a three-line assertion that would have saved the entire debugging detour
described in §4. Write it first. It is implemented for the XLA shim in
`iree_ffi_invoke_f32` (compare `PJRT_Executable_NumOutputs` against the caller's
destination count) — and, because a guard nobody has seen fire is not known to
work, `ffi/test_pjrt_guards.c` violates it on purpose and checks the refusal.

---

## 4. The known blocker: initialization

`scripts/xla_train_r34_imagenette.py` exists and is correct about everything
structural — signature parsing, data formats, and the full output→input mapping.
It **does not converge**: logits at init have std ≈ 12 (want ≈ 1), and Adam then
diverges to NaN.

Cause: the script **reimplements He-init by guessing at parameter names**. The
naming was decoded empirically (`g` = BN gamma — setting it to 0 zeroes the
network; `bt` = beta, survives as an additive term; `b` = conv bias), but the
resulting scale is wrong, most likely at the final dense layer.

**Do not keep reverse-engineering this.** The fix is to stop re-deriving init:

> Have Lean dump its own initial parameters once (`heInitLayer` already does this
> correctly), write them as a flat `params.bin`, and have the driver load that.

This kills two birds: it removes the guesswork, and it gives **G2 for free** —
both backends start from a byte-identical state, so their trajectories are
directly comparable.

---

## 5. Facts the next session should not have to rediscover

**PJRT invocation** — Python, via the JAX-shipped plugin. Kept for quick
experiments only; the shipped path is the C one in §8, which needs no Python:
```python
from jax._src.lib import xla_client as xc
dev = jax.devices()[0]
exe = dev.client.compile_and_load(src, xc.DeviceList((dev,)), xc.CompileOptions())
out = exe.execute(list_of_device_buffers)      # positional, in signature order
```
`jax 0.10.0`, ROCm plugin, `runtime_type = pjrt_ifrt`. The entry function **must**
be `@main`.

**R34 train step layout** (derive at runtime, never hardcode):
```
arg 0        %x        [32, 150528]
args 1..438  params, then Adam m, then Adam v
args 439-441 %lr %bc1 %bc2   (rank-0 scalars)
args 442..513 BN running stats  (%...mui, %...vari)
arg 514      %onehot   [32, 10]
```
Outputs (513) map back **by name**:
`adnew<X>`→`<X>`, `admn<X>`→`<X>m`, `advn<X>`→`<X>v`,
`<X>bnmu`→`<X>mui`, `<X>bnvar`→`<X>vari`, plus `loss`, `bc1`, `bc2`.
**The train step returns `loss` directly** — you do not need an eval graph to see
whether it is learning.

**Adam bias correction** is `bc1 = 1 − β1^t`, `bc2 = 1 − β2^t`
(`VerifiedTrain.lean:420`) — matches the obvious convention.

**Data formats** (all verified by reading headers, not assumed):
- `data/imagenette/train.bin` — 9,469 records × 196,609 B = **leading label byte +
  3×256×256 uint8** (stored at 256 for random cropping).
- `data/imagenette/val.bin` — 3,925 records × 150,529 B = **leading label byte +
  3×224×224** (already centre-cropped). *Different stride from train — assuming
  they match reads garbage.*
- MNIST is raw IDX at the repo root (`data/train-images-idx3-ubyte`, 60,000 @
  28×28), **not** a `.bin`. `data/mnist16` is a 16-image fixture, not a dataset.

**Eval graphs**: `resnet34_fwd` (147 in, batch stats) and `resnet34_fwd_eval`
(219 in, BN running stats named `...mu`/`...var` — i.e. the train step's
`...mui`/`...vari` minus the trailing `i`).

---

## 6. Recipes

**Time any Lean-emitted graph under XLA:**
```
/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/python3 - <<'EOF'
# rename entry to @main, compile_and_load, splat inputs from the signature, execute
EOF
```
**The prototype driver** (structurally correct, does not converge — §4):
```
HIP_VISIBLE_DEVICES=0 <jax-venv>/python3 scripts/xla_train_r34_imagenette.py \
    --steps 300 --log-every 50
```

---

## 7. Decisions that are yours

1. ~~**Does the host loop move to Lean?**~~ **Answered: yes, and it did** (§8). The
   C FFI integration was indeed the same shape as the IREE runtime binding — close
   enough that `ffi/pjrt_ffi.c` implements the *identical* `iree_ffi.h` surface and
   nothing above the shim changed. Python is gone from the run-time path.
2. **Does IREE stay?** It should: it is the **portability** story (five backends,
   two vendors, bit-exact, cross-compiled). XLA is the **speed** story and is
   NVIDIA/AMD-datacenter shaped. They are complementary, and the edge deployment
   question (`visdrone_detector.md` §13) is separate from both.
3. **Does `jax/` still need to exist?** Its main justification was speed. If
   StableHLO→XLA is JAX-speed — and rung 3 measured 20× — then the *proof-tied*
   emitter can be the fast one, and the JAX transcription stops being load-bearing.

---

## 8. Rung 0 — done, Lean-native, no Python (2026-07-27)

The host loop **stays in Lean**. Python appears nowhere at run time.

### How it plugs in

`ffi/pjrt_ffi.c` implements the **same C interface as `ffi/iree_ffi.c`** (the
`iree_ffi.h` surface — verified symbol-for-symbol identical with `nm -D`). So
`iree_lean_ffi.c` and every Lean trainer above it are unchanged; you choose a
backend by choosing which shim gets linked:

```
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build mnist-linear-verified-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/mnist-linear-verified-xla data
```

The plugin is `dlopen`ed at run time (`$PJRT_PLUGIN` overrides the path) — no
link-time dependency on XLA, JAX, or Python.

One behavioural difference: on XLA a session is created from the **`.mlir`
source**, not a precompiled `.vmfb`; XLA compiles in-process, so there is no
`iree-compile` step. `VerifiedNet.mkSession` handles this, branching on
`IreeSession.backendName`, which is resolved by a **weak reference** to a symbol
only the PJRT shim defines. That beats an env var: the binary cannot disagree
with itself about which library it linked.

### Gate results — `linear_train_step`, MNIST, 12 epochs, bs 128

| gate | result |
|---|---|
| **G1** forward tie | inherited from `resnet34_fwd` (rel 8.4e-06) |
| **G2** step tie | **rel L2 5.31e-07**, max abs 5.96e-07, after 5,616 steps |
| **G3** it learns | 89.77% → **92.10%**, identical to IREE at *every* epoch |
| **G4** no dropped state | implemented + **observed to fire** (negative tests) |

G2 is **not bit-exact** — 1,366 of 7,850 parameters match exactly, the rest differ
at the f32 rounding floor. That is dot-product reassociation from different GEMM
tiling, which is expected; the doc's threshold is rel < 1e-4 and this is 200× under
it. Do not report this as bit-exactness; IREE's bit-exact-across-backends property
is a separate claim that does not extend to XLA.

`ffi/test_pjrt_guards.c` deliberately violates each guard and checks it is
rejected — asking for 1 of 2 outputs (G4), calling a graph the session does not
hold, and declaring a wrong output size. All three are refused:

```
gcc -O2 -Iffi ffi/test_pjrt_guards.c -Lffi -lpjrt_ffi -ldl -Wl,-rpath,./ffi -o /tmp/tg && /tmp/tg
```

### Speed — and why this number is not the point

**342 ms/epoch (XLA) vs 549 ms/epoch (IREE) — 1.6×.** MNIST-linear is a
784→10 dense op: no convolutions, so none of IREE's conv weakness is in play, and
per-step host↔device copies dominate. Rung 0 exists to establish *correctness
plumbing*, not speed. The 20–40× lives at rungs 3–4 where convs dominate.

Both backends still round-trip parameters host↔device every step because that is
what `iree_ffi_invoke_f32`'s signature does. Keeping it identical was deliberate —
it makes the G2 diff apples-to-apples. Device-resident parameters with donated
buffers are the obvious next optimization and will need a wider interface.

### Rung 1 — `mlp_train_step`, done (2026-07-27)

`mnist-mlp-verified-xla`. 784→512→512→10, 6 param tensors, 669,706 floats,
packed-params path.

**It needed zero new shim code.** `lean_iree_mlp_train_step_v` and
`lean_iree_forward_f32` both route through `iree_ffi_invoke_f32`, which rung 0
already implemented. The only change was swapping `VerifiedNet.train`'s two
session-creation sites to `mkSession` — the same two lines as `trainLinear`.

**§4's init problem did not appear, because it is already solved on this path.**
He init runs *in Lean* (`mkParam`, seeded by `LEAN_MLIR_SEED`, default 1), so both
backends start byte-identical with nothing to reverse-engineer. This is precisely
what §4 prescribes for R34 — the machinery exists, it just is not wired into the
Python R34 prototype. Rung 3 should reuse `mkParam`, not re-derive init.

| gate | result |
|---|---|
| **G2** step tie | **rel L2 5.82e-09 @ 1 step**, 7.74e-08 @ 100 — passes |
| **G2** full run | 1.32e-02 @ 5,616 steps — **chaos, not a defect**; see §3 |
| **G3** it learns | **97.80%** (XLA) vs 97.76% (IREE), 12 epochs |
| **G4** no dropped state | 6 outputs asserted against 6 destinations |
| determinism control | XLA twice → **bit-identical** |

**1,030 ms/epoch (XLA) vs 2,970 ms (IREE) — 2.9×.** Better than rung 0's 1.6×
because there is more compute per step to amortize the host↔device round-trip,
and still nowhere near the 20–40× — there are no convolutions here, which is
where IREE actually falls down.

### Rung 1b — `cnn_train_step`, the first CONVOLUTIONAL graph (2026-07-27)

`mnist-cnn-verified-xla`. `conv 1→32 → conv 32→32 → maxpool → dense 6272→512 →
512 → 10`, 10 param tensors, 3,489,130 floats. Not a numbered rung, but it is the
first graph where the *reason for this whole doc* — IREE's conv codegen — applies.
Zero new shim code again; `VerifiedNet.train` does not care that params are 4-D.

Full 10-epoch runs on both backends:

| gate | result |
|---|---|
| **G2** step tie | **rel L2 1.19e-08 @ 1 step** — passes |
| **G2** full run | 4.67e-02 @ 4,680 steps — chaos (maxpool adds a second discrete branch) |
| **G3** it learns | **98.89%** (XLA) vs 98.99% (IREE), 10 epochs |
| **G4** no dropped state | 10 outputs asserted against 10 destinations |

**The 0.10pp accuracy gap is noise, and that was checked rather than assumed.**
Paired runs at three init seeds (`LEAN_MLIR_SEED`), final test accuracy:

| seed | IREE | XLA | Δ |
|---|---|---|---|
| 1 | 98.99% | 98.89% | IREE +0.10 |
| 2 | 98.98% | 98.94% | IREE +0.04 |
| 3 | 99.00% | **99.01%** | XLA +0.01 |

The sign flips, so there is no systematic accuracy penalty from XLA — which is what
the mechanism predicts, since chaotic divergence from ULP-level reassociation gives
neither backend a reason to land higher. Two same-direction samples looked
suggestive; the third is why you take three.

**~5.0 s/epoch (XLA) vs ~16.3 s/epoch (IREE) — 3.1–3.3×.** XLA was stable at
4.9–5.2 s across every run; IREE ranged 15.5–20.2 s and drifted *upward* over the
session (thermal or GPU state, not a code change), so the conservative end of that
range is the honest number. The trend across the three graphs is the point:

| graph | convs | IREE | XLA | speedup |
|---|---|---|---|---|
| linear | 0 | 549 ms | 342 ms | 1.6× |
| MLP | 0 | 2,970 ms | 1,030 ms | 2.9× |
| CNN | 2 | ~16,300 ms | ~5,000 ms | **~3.2×** |

Still far from the 20–40× at R34/FPN scale, and that is expected: two small
convolutions at 28×28 are not where a 121× conv gap dominates a step. The number
to watch is rung 3.

### Rung 2 — `cifar8_bn_adam_train_step`, done (2026-07-27)

`cifar8-bn-verified-adam-xla`. 8× conv→BN→relu [16,16,32,32], 4 pools, 128→64→64→10.
38 params → `[θ|m|v]` = 114 tensors + 3 rank-0 scalars = **117 in / 117 out**.

**Still no new shim code.** The Adam moments and the runtime `lr`/`bc₁`/`bc₂` all
ride inside the packed params blob (`adamShapes` appends three rank-0 shapes), so
this goes through `mlpTrainStepV` → `iree_ffi_invoke_f32` like every earlier rung.
The one genuinely new capability was **rank-0 tensor inputs**
(`BufferFromHostBuffer` with `num_dims = 0`) — PJRT accepts them as-is.

**A checkpoint hazard was found and fixed.** `trainAdamSched` auto-resumes from
`.lake/build/<slug>_<variant>_ckpt.bin`, which was **not backend-scoped**. An XLA
run would happily resume from an IREE checkpoint, fusing two trajectories into one
while printing nothing unusual. The path now carries an `_xla` suffix. Any future
driver with resume needs the same treatment — this is the §3 failure mode exactly:
looks like success, is not.

| gate | result |
|---|---|
| **G1** forward tie | step-0 loss **2.921757** (XLA) vs 2.921755 (IREE), rel ~7e-7 |
| **G2** gradient tie | **8.20e-05 @ 1 step** (see below — measure `m`, not θ) |
| **G3** it learns | 60.87% vs 61.45% @ 5 epochs; losses 1.1283 vs 1.1289 |
| **G4** no dropped state | 117 outputs asserted against 117 destinations |

#### G2 must be measured on the GRADIENT for Adam nets, not on θ

Rung 2's θ tie came in at **1.005e-04 @ 1 step** — at the 1e-4 threshold, four
orders of magnitude looser than rung 1. That looks like a defect. It is not, and
the controlled experiment says why. `cifar8-verified-adam-xla` is the **same
driver, same hyperparameters, same seed, minus BatchNorm**:

| | gradient (`m` @ 1 step) | θ @ 1 step |
|---|---|---|
| no BN | **1.28e-06** | 1.23e-04 |
| with BN | **8.20e-05** | 1.01e-04 |

θ diverges by ~1e-4 **whether or not BN is present**, even though the gradients
differ by 64×. So θ's divergence is not coming from the graph at all — it is
Adam's update being *scale-free*. With moments zero-init, step 1 gives
`m̂/(√v̂+ε) = g/(|g|+ε) ≈ sign(g)`, so the step is ±lr **regardless of |g|**. For
any parameter whose gradient is near zero, a 1-ULP disagreement flips the sign and
produces a full-magnitude ±lr difference. No backend pair can tie tighter than
that on Adam, at any N.

**So for Adam nets, gate G2 on `m` after one step** — with zero-init moments,
`m = (1−β₁)·g` exactly, making it a direct read of the gradient. Both nets pass
(1.28e-06 and 8.20e-05). Gating on θ would have condemned a correct backend.

**BN amplifies backward reassociation ~64×** (1.28e-06 → 8.20e-05 on the same
architecture). That is a real effect and it is the number to watch at rung 3:
**R34 has ~36 BN layers against this net's 8**, so its gradient tie could plausibly
land near or over 1e-4 without anything being wrong. Measure it before judging it.

#### Speed — the weakest rung so far, and worth understanding

**25 ms/step (XLA) vs 33 ms/step (IREE) — 1.32×** steady state (`LEAN_MLIR_MAX_STEPS`
probe, median of 112 steps, warmup excluded). Whole-run wall clock over 5 epochs was
60.4 s vs 68.4 s (1.13×), but that figure is unfair to XLA: it compiles in-process
every run while IREE reuses a cached `.vmfb`.

1.32× is *lower* than rung 1b's 3.2× despite having 4× more convolutions, which
kills the tempting "more convs → bigger speedup" story. These convs are tiny —
16–32 channels at 32×32 — so both backends are launch/overhead-bound rather than
compute-bound, and MIOpen's advantage needs real arithmetic intensity to show.
The 20–40× measurements are at 224×224 with 64–512 channels. Rung 3 is still the
number that decides this.

### Rung 3 — `resnet34_adam_train_step` on Imagenette (2026-07-27)

`resnet34-verified-adam-xla`. 22,674,570 params, bs 32 @ 224², **513 in / 513 out**,
**36 BN layers with running statistics** (17,024 stat floats, eval via
`@resnet34_fwd_eval`).

**Still no new shim code.** `trainAdamSched` appends the BN stat shapes to
`adamShapes` and threads them through the packed blob, so even running stats ride
`iree_ffi_invoke_f32`. The `train_step_adam*` family remains unported and stubbed;
nothing on the ladder has needed it. §1's earlier claim that rung 2 or 3 would
force it was wrong.

#### Speed — the number the whole exercise was for

| measure | IREE | XLA | speedup |
|---|---|---|---|
| steady state (probe, median of 32) | 1,702 ms/step | **205 ms/step** | **8.3×** |
| epoch 1 end-to-end (incl. XLA JIT) | 555.7 s | **77.7 s** | **7.2×** |
| **marginal epoch** (2-epoch minus 1-epoch) | ~549 s | **64.9 s** | **8.5×** |
| 80-epoch **projection** | ~12.2 h | **~1.5 h** | |

The marginal-epoch figure is the one to quote for planning: XLA 2 epochs took
142.6 s against 77.7 s for 1, so a steady epoch is **64.9 s** and the one-time
cost (data load + JIT) is ~12.8 s. **The 80-epoch numbers are projections — a full
run has not been done.** The driver checkpoints every epoch, so one is resumable.

**8.3× is real but it is not the doc's 20.2×**, and the gap is ours, not XLA's.
That 20.2× was *pure graph execution* (1237 vs 61.3 ms). At 205 ms/step against a
61 ms graph, **~70% of XLA's step is still host↔device round-trip**: `[θ|m|v]` is
272 MB, moved both ways every step, as 513 separate buffers.

Batching those transfers — issue all, then await, instead of awaiting each — took
**256 → 205 ms/step (20%)** for ~15 lines in `iree_ffi_invoke_f32`. The remaining
~144 ms is the transfer itself. **Device-resident parameters with donated buffers
is the next big win** and is worth more than everything else left on this ladder;
it should close most of the distance to 20×.

#### G2 FAILS here, and it is not certified as benign

The gradient tie (`m` @ 1 step) comes in at **5.20e-02** — 500× over threshold,
0.0% of parameters bit-identical.

| net | BN layers | gradient rel L2 @ 1 step |
|---|---|---|
| MNIST MLP | 0 | 5.8e-09 |
| cifar8 (no BN) | 0 | 1.28e-06 |
| cifar8-bn | 8 | 8.20e-05 |
| **R34** | **36** | **5.20e-02** |

Everything available points at numerical amplification rather than a defect:
- **The error profile is a smooth monotonic ramp with backprop depth** — 2.19e-03
  in the chunk nearest the loss, rising steadily to 5.64e-02 at the first conv. A
  mis-threaded tensor or swapped BN stat would spike in one chunk or show a
  discontinuity. This does neither.
- The **forward ties cleanly**: step-0 loss 2.486055 vs 2.486046 (rel ~4e-6).
- XLA is **bit-identical run-to-run for a single step**, so the 1-step gradient
  comparison above is measuring backends, not noise. (It is *not* deterministic
  over a full epoch — see §3.)
- **G4 passes**: 513 outputs against 513 destinations.
- Both backends learn: epoch-1 loss 1.9916 vs 1.9834.
- BN backward contains a catastrophic cancellation
  (`dy − mean(dy) − x̂·mean(dy·x̂)`); 36 stacked instances amplifying to 5e-2 is
  entirely consistent with the 0/8/36-layer series above.

Epoch-1 val accuracy came out **47.29% (XLA) vs 41.32% (IREE)**, 6 points apart —
which initially looked like a consequence worth worrying about. **It is not
evidence of anything: XLA alone spans 43.21–47.29% across three identical runs**
(§3). The cross-backend gap is inside one backend's own run-to-run variance, so
that comparison has no power. Judge G3 on the trained-out curve, not one epoch.

#### Resolved: the 1e-4 threshold is unachievable at R34 depth

A third backend was not needed. The question — "is a 5.20e-02 spread between two
valid f32 evaluations normal here?" — is answerable with **one** backend, by
measuring how sensitive the gradient is to a perturbation at f32 resolution.
`LEAN_MLIR_PERTURB_R` displaces θ along a random unit vector of exact L2 norm r
(`F32.perturbUnit`) before training; comparing XLA against **itself**:

| input perturbation (rel) | resulting gradient rel Δ |
|---|---|
| 1e-8 (≈0.09 ULP per component — most components unchanged) | **6.55e-03** |
| 1e-7 | **6.05e-03** |
| 1e-6 | **9.51e-03** |
| — *XLA vs IREE, same θ* — | *5.20e-02* |

The response is **saturated, not linear**: shrinking the nudge 100× does not
shrink the gradient change. That is an intrinsic noise floor. **R34's gradient
does not reproduce to better than ~6e-3 even against itself, under a sub-ULP
nudge.** So a 1e-4 gate fails XLA-vs-XLA by 60×; it is not a meaningful threshold
at this depth, and rung 3's 5.20e-02 is within ~9× of the floor.

The floor is itself a depth effect — measured the same way on the 8-BN-layer net:

| net | BN layers | XLA vs itself (sub-ULP) | XLA vs IREE | ratio |
|---|---|---|---|---|
| cifar8-bn | 8 | 8.24e-08 | 8.20e-05 | 996× |
| R34 | 36 | **6.05e-03** | 5.20e-02 | 8.6× |

The ratio is far larger on the shallow net because an input nudge is injected
**once**, whereas a backend difference is injected at **every one of thousands of
intermediate ops**. Both quantities explode with BN depth (floor by ~74,000×,
backend disagreement by ~634× across 8→36 layers).

**Revised G2 for deep nets: compare the backend disagreement against the net's own
measured noise floor, not against a fixed constant.** The threshold is
self-calibrating and the probe costs one extra run.

**Honest limit of that result on its own.** It shows the *gate* was wrong and that
the observed spread is what ill-conditioning predicts. It does **not** prove the
VJPs are correct — a systematic error could hide beneath a 6e-3 floor.

#### The correctness claim comes from the layer-level oracle, and it passes

`tests/vjp_oracle/run.sh` diffs phase 3 (Lean → MLIR → IREE, the **hand-derived**
VJPs) against phase 2 (Lean → JAX, **autodiff**) on step-2 loss — the first step
whose loss depends on the backward pass. At layer granularity conditioning is mild,
so a real tie is achievable and a wrong VJP has nowhere to hide. Every layer family
R34 is built from:

| family | step-1 Δ | step-2 Δ | tol |
|---|---|---|---|
| `conv` | 5.41e-07 | **4.62e-07** | 1e-5 |
| `convbn` | 3.65e-07 | **5.09e-07** | 1e-4 |
| `residual` | 1.01e-06 | **2.55e-06** | 1e-5 |
| `bottleneck` | 3.36e-06 | **6.40e-06** | 1e-4 |
| `dense` | 9.80e-09 | **2.73e-07** | 1e-5 |
| `global-avg-pool` | 7.84e-09 | **6.40e-07** | 1e-5 |

That is **every layer family R34 is built from** — stem conv, conv+BN, residual
blocks, GAP, dense head — all passing. `convbn`, the BN backward that was the prime
suspect, ties to **5.09e-07**.

**Verdict: rung 3 is correct.** The two results compose. Layer-level: the VJP math
is independently confirmed against JAX autodiff to ~1e-6. Whole-net: the
IREE/XLA spread sits within 9× of a *measured* noise floor that a single backend
cannot beat against itself. There is no defect; whole-net f32 agreement at R34
depth is simply not an instrument capable of resolving one.

**Running the oracle needs `.venv/bin/python3`** (the runner hardcodes it). This
repo's `.venv` is a gitignored symlink farm, and a *symlinked* interpreter derives
`sys.prefix` from the symlink's location — so it looks for site-packages inside
this `.venv` and cannot find jax. Use a wrapper script that `exec`s the real venv
interpreter instead. (Also note `python3 -c 'import jax'` from the repo root
imports the local `jax/` **directory**, not the package.)

### Baseline: how close is the verified path to hand-written JAX?

`jax/MainResnet.lean` is the same ResNet-34 on the same Imagenette, written
directly against JAX. Run on GPU 1 (`HIP_VISIBLE_DEVICES=1`) so it does not
contend with the verified trainer, 3 epochs, marginal epoch = ep3 − ep2:

| path | batch | s/epoch | ms/step |
|---|---|---|---|
| verified, before the memcpy fix | 32 | 64.9 | 205 |
| **verified, after the memcpy fix** | **32** | **52.5** | **162** |
| **native JAX** | **32** | **46.0** | **156** |
| native JAX | 192 | 45.0 | — |

**The proof-tied path is within 1.04× of hand-written JAX per step** (1.14× per
epoch — the residual is eval and data-pipeline differences, not the train step).
It was 1.41× before removing the two per-step host memcpys (§ below).

Two things fall out of this:
- **Batch size is not the lever it looked like.** JAX is 46.0 s/epoch at bs32 and
  45.0 at bs192 — essentially flat, because it keeps parameters on device and the
  total compute per epoch is the same either way. Re-rendering the verified graph
  at a larger batch would help *us* (our transfer bill is per-step) but it does not
  unlock some large headroom that JAX is enjoying. Deprioritise it.
- **The remaining gap was 49 ms/step, ~36 ms of it host memcpy — now fixed.**
  Removing it landed 205 → **162 ms/step**, slightly better than predicted, with no
  change to the emitter, the graph, or the proofs.

Also retires an earlier framing: §0's "61.3 ms pure graph" is not comparable to
these end-to-end step times (JAX's own real per-step, including its data pipeline,
is 156 ms). The honest headroom over native is 49 ms/step, not ~144.

### Where the 205 ms/step goes, and what is worth fixing

Measured / estimated split for the R34 step:

| component | ms | basis |
|---|---|---|
| graph execution | ~61 | §0 pure-graph measurement |
| **host memcpy (concat + extract)** | **36** | measured: 2×272 MB @ 15.0 GB/s |
| PCIe transfer, 544 MB both ways | ~30 | bandwidth estimate |
| ~1026 buffer create/destroy + PJRT | ~78 | remainder |

Ranked by value, not by ease:

1. **Device-resident parameters (donated buffers)** — targets the ~144 ms of
   non-graph time. Requires the session to hold device buffers across calls: a
   real interface change, not a tweak. Biggest single win available.
2. **Stop pre-concatenating the parameter blob — ~36 ms/step, genuinely easy.**
   Every step does `F32.concat #[thetamv, tail, runningBnStats]` and
   `out.extract 0 mvBytes`, each copying all 272 MB on the host. The C shim
   already walks parameters per-tensor via the shapes array, so it can take the
   pieces separately. ~18% off the step for no device-side work and no change in
   semantics: 205 → ~169 ms/step, 80 epochs 87 → ~72 min.
3. **Batch size — possibly the largest lever, but it is a training decision.**
   Parameter transfer is **per-step and independent of batch size**: 544 MB moves
   whether the step does 32 images or 192. The `jax/` reference config uses
   bs192; the verified trainer uses bs32. Re-rendering at a larger batch cuts
   steps/epoch (295 → 49 at 192) and the transfer bill with it — but it changes
   the LR schedule and BN statistics, so it is a different training run, not a
   free optimization.
4. **Executable cache** — see below. Worth ~nothing here.

### The XLA JIT is a one-off per process — but not a cheap one

Worth being unambiguous, because it is easy to misread as per-step overhead:
XLA compiles **once per session, i.e. twice per run** (train step + forward), at
startup. The training loop only calls `Execute` on the already-compiled
executable. The `[pjrt_ffi] compiled …` line appears exactly twice per run.

Measured (`ffi/pjrt_ffi.c` times `PJRT_Client_Compile` directly):

| graph | outputs | JIT |
|---|---|---|
| `mlp_train_step` | 6 | **14,094 ms** |
| `cifar8_bn_adam_train_step` | 117 | **7,000 ms** |
| `mlp_fwd` | 1 | 197 ms |
| `cifar8_bn_fwd` | 1 | 326 ms |

The MLP costs *more* than the far larger CIFAR graph — that is XLA autotuning its
big dense GEMMs (784×512, 512×512), not graph size.

**This does not affect the per-step / per-epoch numbers quoted above** (those are
measured inside the loop), but it does affect end-to-end wall clock, and IREE does
not pay it because `compileVmfb` caches the `.vmfb` across runs. A 12-epoch MNIST
MLP run is really 14.3 s JIT + 12.4 s training under XLA against 35.6 s under
IREE — **1.33× end-to-end, not the 2.9× steady-state figure.** For a 4-hour
VisDrone run it rounds to nothing; for the repo's many short experiments it eats
most of the gain. Quote the steady-state number for throughput claims and the
end-to-end number for "how long until I see a result".

**This is fixable and the API is already there.** `PJRT_Executable_Serialize` +
`PJRT_Executable_DeserializeAndLoad` are both in the C API (v0.90), so the shim can
cache compiled executables to disk keyed on the `.mlir` path + mtime — the exact
analogue of the `.vmfb` cache. Note the header's warning: a serialized executable
is only valid for "the same platform and library version", so the cache key must
include the plugin version and be treated as disposable.

**But be clear about what it buys, because it is not a training speedup:**

| run | total | JIT | cache saves |
|---|---|---|---|
| R34, 80 epochs (projected) | 5205 s | 6.5 s | **0.1%** |
| R34, 1 epoch | 77.7 s | 6.5 s | 8.3% |
| cifar8-bn, 5 epochs | 60.4 s | 7.3 s | 12.1% |
| MNIST-MLP, 12 epochs | 26.7 s | 14.3 s | **53.5%** |

It halves the MNIST-MLP demo and is worth ~nothing on an R34 training run. Build it
for the edit-run loop and for CI, not for throughput.

### What is deliberately not done

`iree_ffi_train_step_{mlp,generic,adam,adam_seg,adam_softlabel,adam_ddpm,adam_yolov1}`
are **stubs that fail loudly** with a pointer to §2. They must exist for linking;
they must not return plausible garbage. Every net above rung 0 still requires the
IREE build.

### Files

| file | what |
|---|---|
| `ffi/pjrt_ffi.c` | the shim — PJRT-backed implementation of `iree_ffi.h` |
| `ffi/pjrt_c_api.h` | vendored PJRT C API (v0.90; plugin reports 0.96 — append-only ABI, so older is the safe direction) |
| `ffi/pjrt_compile_options.h` | **generated**; see below |
| `ffi/test_pjrt_guards.c` | negative tests proving the guards fire |
| `apps/mnist/{Mlp,Cnn}VerifiedCommon.lean` + `Main*Xla.lean` | rungs 1 / 1b |
| `scripts/gen_pjrt_compile_options.py` | the only Python, run at **build** time |
| `apps/mnist/LinearVerifiedCommon.lean` | shared body of both executables |
| `apps/mnist/MainMnistLinearVerifiedXla.lean` | the XLA root |

The one wrinkle worth knowing: `PJRT_Client_Compile` requires `compile_options` as
a serialized `CompileOptionsProto`, and **passing zero bytes is not the same as
defaults** — proto3 zero-defaults give `replica_count = 0` and XLA aborts with
`Check failed: replica_count > 0`. So a 905-byte blob is generated once into a C
header. Capturing what the installed XLA actually constructs also keeps
`DebugOptions` identical to JAX's, which keeps timing comparisons honest.

---

## 9. What this does not change

The theorem is "emitted graph ≡ spec". XLA, IREE, TensorRT, and XNNPACK are all
**trusted, unverified lowerers** occupying the same tier. Swapping among them costs
nothing in verification and buys nothing in verification. Any writeup that implies
otherwise is wrong in both directions.

---

## 10. Scope: data-parallel multi-GPU via in-graph `all_reduce`

**Status: scoped, not built (2026-07-27).** Target box is *ares* — **4× RTX 4060 Ti**.

### 10.1 The seam exists, and it is small

`ViTRender.emitAdamV (θ g m v : String) (ds : List Nat) (t : String)` takes the
gradient **as a named SSA value** and emits the AdamW update from it. So true
data-parallelism does not need the train step restructured — it needs the
gradient replaced by its cross-replica mean at that one point:

```mlir
%gar<t> = "stablehlo.all_reduce"(%g<t>) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %s = stablehlo.add %a, %b : tensor<f32>
    stablehlo.return %s : tensor<f32>
}) { replica_groups = dense<[[0,1,2,3]]> : tensor<1x4xi64> } : (T) -> T
%gavg<t> = stablehlo.divide %gar<t>, %adN<t>   // N broadcast as a constant
```

Three call sites, all the same one-line lambda (`ViTRender.lean:582/633/707`).
A `replicas : Nat := 1` field on the render config plus a wrapper around
`emitAdamV` covers all of them. **No collective exists anywhere in the emitter
today** — this would be the first.

### 10.2 What else has to change

| layer | change | rough size |
|---|---|---|
| emitter | `all_reduce` + divide before `emitAdamV`; `replicas` in the render cfg | ~30 lines |
| compile options | `num_replicas = N` + device assignment (currently hardcoded 1 in `scripts/gen_pjrt_compile_options.py`) | ~10 lines |
| `ffi/pjrt_ffi.c` | `Execute` with `num_devices = N`: `argument_lists[N][num_args]`, `output_lists[N][num_outputs]`; params replicated, x/y sharded; read back from replica 0 | ~150 lines |
| driver | per-replica batch slicing; global batch = N × per-replica | ~30 lines |

The PJRT C API already supports this — `PJRT_LoadedExecutable_Execute` takes
`[num_devices][num_args]` — so no new API surface, just real work in the shim.

### 10.3 Two things that make this bigger than it looks

**(a) Device-resident parameters are a PREREQUISITE, not a companion.**
Today the shim pushes the whole `[θ|m|v]` blob host→device and back every step —
272 MB each way at R34. With 4 replicas that becomes **4× the PCIe traffic**, on
4060 Tis (PCIe x8, well under the 7900 XTX's bus). Data-parallel on top of the
current host-resident design would very plausibly be a net *loss*. Sequencing:
device-resident params first, `all_reduce` second.

**(b) BatchNorm — only a problem if you SPLIT the batch. Do not split; SCALE.**

Two different designs, and they behave completely differently:

| design | per-replica batch | BN group | consequence |
|---|---|---|---|
| **split** bs256 → 2×128 | 128 | **shrinks** | Ghost-BN: a real accuracy cost, needs sync-BN (a second collective) to undo |
| **scale** 2× bs256 → global 512 | 256 | **unchanged** | BN identical to the 1-GPU bs256 baseline. Nothing to fix. |

Under **scaling**, each replica evaluates *exactly the graph that is already tied*,
at exactly the batch size it was tied at. The collective then averages gradients of
the same function over two disjoint equal-size batches — which is the gradient of
the mean loss over their union, with no BN caveat, because no claim is being made
about equivalence to a single-device bs512 run. **The §10.4 BN exclusion is not
needed in this design.**

What you pay instead is a *training-recipe* cost, not a correctness one: the global
batch grows (512 on 2 GPUs, 1024 on 4), so the LR schedule needs the linear-scaling
rule and warmup. Memory `cross-box-throughput-anchors` measured compute-bound nets
(R34/R50/ViT) rewarding batch up to ~1–2k, so 4×256 sits comfortably inside the
useful range.

If you ever *do* need to split (memory-bound, batch will not fit),
`jax/scripts/bn_sharding_demo.py` already quantifies the synced-vs-sharded top-1
gap. Read it before choosing.

### 10.4 Verification implications — smaller than feared, for a specific reason

**There is no whole-graph `den(trainStep) = spec` theorem for R34 to redo.**
`verified_mlir/resnet34_adam_train_step.mlir` is built by string concatenation in
`tests/TestResnet34Train.lean` (with `ViTRender.emitAdamV` for the update tail).
`LeanMlir/ViTRender.lean` says so in its own header: *"NOT a single
`den(trainStep)` theorem — faithful PER-OP, validated by the Lean gradchecks and
by training."* What backs R34 is (a) per-op faithfulness, (b) the spec-level VJP
theorems, (c) the `pretty`-of-proven-graph drift guard in
`LeanMlir/Proofs/Codegen/ViTRender.lean`, and (d) the layer-family VJP oracle.
Adding a line to a hand-written string renderer does not invalidate any of those.

**The real obstacle is different, and worth stating precisely: the specification
language has no notion of replicas.** `BatchableOp : Nat → Nat → Type` indexes a
single-batch function ℝⁿ → ℝᵐ. There is no spec op that `all_reduce` could be
"faithful to", so adding it to the verified AST is not "add a constructor" — it
would mean teaching the spec language about multiple devices. That is a far bigger
change than the emitter edit, and it is the wrong place to spend the effort.

**Better design than either option I first listed.** Do not try to make
`all_reduce` a verified op. Instead prove the **batch-decomposition lemma at the
spec level**, where no MLIR and no devices appear:

> for a mean-reduced loss, the full-batch gradient over B equals the mean of the
> per-shard gradients over N shards of size B/N.

That is net-independent, provable in the existing spec language, and it makes
`all_reduce`+divide the *trusted implementation of a proven mathematical
identity* — exactly the tier structure the repo already uses (proven math /
trusted lowerer), rather than a new trusted hole.

**The BN caveat depends on the design chosen in §10.3b.** If the batch is *split*,
the lemma is false for BatchNorm (batch statistics do not decompose over shards)
and BN must be excluded from the statement. If the batch is *scaled* — each replica
keeps the tied batch size — the caveat disappears: every replica computes the tied
function unchanged, and the lemma is only needed in its easy form (mean of
gradients over disjoint equal batches = gradient of the mean loss over the union).
**Prefer scaling; it costs a proof obligation less.**

Whatever is chosen: **"verified multi-GPU" is not a claim any of these earns.**
The most that can be said is "the gradient averaging is a proven identity; the
collective implementing it is trusted, like the lowerer".

### 10.5 The measurement that should come first

**The JAX-parity result in §8 was measured in a data-bound regime and must not be
carried into ImageNet planning.** JAX on Imagenette is flat at 46.0 / 45.0 /
44.1 s per epoch for bs32 / bs192 / bs256 — ~215 img/s, against this box's
measured R34 capability of **~918 img/s at bs256** (memory
`cross-box-throughput-anchors`). Both paths are idling the GPU on data loading, so
"1.04× of JAX" is true of this workload and says little about a compute-bound run.

Per-image, the verified path is at 162 ms / 32 = **5.06 ms/img**, against the
anchor's 279 ms / 256 = **1.09 ms/img** — roughly **4.6× off** the box's
compute-bound rate. Some is bs32 under-utilisation, some is the per-step parameter
transfer (constant in batch, so it amortises 8× better at bs256).

**So the first move is a bs256 re-render and a step-time measurement, not
multi-GPU.** It is cheap, it is needed for ImageNet anyway, and it establishes the
per-image number every scaling estimate depends on.

### 10.6 Expected payoff, honestly

Measured (memory `cross-box-throughput-anchors`, 2026-07-11): ares 4×4060 Ti does
**R34 1,896 img/s** (bf16, bs256) against this box's **918**. So the whole 4-GPU
box is ≈ **2.1×** one 7900 XTX for this net — not 4×. Note the ares figure is
bf16 while the verified emitter is fp32, and bf16 buys ~nothing for conv on MIOpen
(it may differ on cuDNN — unmeasured for our path).

Ordering by value:
1. **bs256 re-render + measure** — establishes the real baseline; cheap. Also the
   unit the multi-GPU design scales in (§10.3b), so it is a prerequisite twice over.
2. **Device-resident parameters** — prerequisite for (3), and worth doing alone.
3. **`all_reduce` multi-GPU** — ~2× on ares, once (2) is in. Scale, do not split.

Note the renderer/trainer split when editing: the trainer is
`apps/imagenette/MainResnet34VerifiedAdam.lean` → `Resnet34AdamCommon.lean` →
`VerifiedNet.trainAdamSched` (it emits no MLIR), while the artifact it consumes is
written by `tests/TestResnet34Train.lean:467`. The collective goes in the latter.
(The **SGD** step `resnet34_train_step.mlir` additionally has a Proofs-tree
renderer, `LeanMlir/Proofs/Codegen/ResNet34Render.lean:306`; the **Adam** step used
for training does not.)


---

## 11. Multi-GPU de-risk: the collective works through our own stack (2026-07-27)

**Done, before touching the emitter.** `ffi/test_pjrt_allreduce.c` is the multi-GPU
analogue of the rung-0 spike: a known-answer 2-replica `all_reduce` driven through
the *same* PJRT C API the verified trainers use. No JAX, no Python at run time.

```
addressable devices: 2
compiled the collective OK
executed across 2 replicas
  replica 0 -> [11.0 22.0 33.0 44.0]      # got [1,2,3,4]
  replica 1 -> [11.0 22.0 33.0 44.0]      # got [10,20,30,40]
ALL-REDUCE CORRECT across 2 GPUs via the PJRT C API
```

Everything the design depends on is now proven except the emitter edit itself:
`num_replicas = 2` compile options, multi-device `Execute`
(`argument_lists[2][…]`, `output_lists[2][…]`), replica-r-to-device-r placement,
and RCCL reachable from C rather than only from JAX.

**The exact StableHLO the emitter must produce** (this parses and runs):

```mlir
%0 = "stablehlo.all_reduce"(%x) ({
^bb0(%a: tensor<f32>, %b: tensor<f32>):
  %s = stablehlo.add %a, %b : tensor<f32>
  stablehlo.return %s : tensor<f32>
}) { replica_groups = dense<[[0, 1]]> : tensor<1x2xi64> } : (T) -> T
```

**Gotcha, cost one round trip:** adding `use_global_device_ids` (which looks
natural) fails compilation with *"channel_id must be positive when
useGlobalDeviceIds is set but got: 0"*. For a plain cross-replica reduce, omit it —
it is only for the global-device-id addressing mode, which needs a channel handle.

**Run it with no `HIP_VISIBLE_DEVICES` set**, or the client sees one device and the
test refuses (it says so rather than silently running single-replica):

```
gcc -O2 -Iffi ffi/test_pjrt_allreduce.c -ldl -o /tmp/ar && /tmp/ar \
  <plugin.so> <allreduce.mlir> <compile_options_r2.pb>
```

The 2-replica options blob is the same generator as the 1-replica one with
`num_replicas = 2` (`scripts/gen_pjrt_compile_options.py`, 905 bytes either way).

### What is left

1. Emitter: `all_reduce` + divide-by-N before `emitAdamV` (§10.1) — ~30 lines.
2. Shim: generalise `iree_ffi_invoke_f32` to N replicas — ~150 lines.
3. Driver: N batches per step — ~30 lines.
4. Validate on a **non-BN** net (`cifar8-verified-adam`), where 2×k sharded must
   equal 1×2k single-device *exactly* (to the noise floor). Only then R34, where
   BN makes it inexact by design (§10.3b).

Prior concern retired: the bench script's "known multi-GPU sharding hang" warning
(`jax/scripts/jax_r34_bf16_bench.py`) refers to a bug fixed some time ago.
