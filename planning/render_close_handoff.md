# Handoff — closing the CNN render, then CIFAR (render + close)

This continues `planning/verified_train_step.md`. The linear and MLP train steps are
**closed both ways** (text rendered from proven forward graphs *and* every parameter
output proven to denote `θ − lr·certified`). The CNN train step has the **close** (all
ten param outputs certified) but not the name-threaded **render**. This doc hands off:
(1) the CNN render, then (2) the same treatment — render + close — for CIFAR.

All work targets the `ℝ`-level / per-op-template trusted base unchanged (see
`verified_train_step.md` §"What stays trusted"): ℝ→Float32, `iree-compile`, the op
templates (link 4). "Closed" means text↔denotation↔certified-math within that base.

---

## 0. The two reusable patterns (established for linear + MLP)

**Render (text = render of proven graphs).** See `LeanMlir/Proofs/MlpRender.lean`
(`mlpTrainStepStructured`). Recipe:
- Render each forward piece from its proven `SHlo` graph (`denseF`/`reluF`/… — each
  `*_faithful`) via `pretty B graph` in a `StateM Nat` do-block, capturing the fresh
  result SSA name.
- Operand *values* are placeholders (`fun _ => 0`); `pretty` renders names only and
  never reads an operand's value, so the renderer stays **computable** (`#eval`-able).
- Emit the backward + param-grad + SGD ops as templates referencing the **captured
  names** (so the relu-back `compare GT` reads the rendered pre-activation, the param
  grads read the rendered activations).
- Validate: `#eval` to a file, `iree-compile`, swap into `verified_mlir/<net>_train_step.mlir`,
  run the `*-verified` exe, confirm accuracy, restore the committed file (see §3).

**Close (param outputs denote certified).** See `mlp_render_{W,b}*_certified`
(`LeanMlir/Proofs/MlpTrainStep.lean`) and `cnn_render_conv{W,b}_certified`
(`CnnTrainStep.lean`). Recipe: each rendered SGD output `θ − lr·(emitted grad)` equals
`θ − lr·(certified Jacobian · cotangent)` by `rw [<the layer bridge>]`. The bridges are
generic in the cotangent, so one theorem covers all layers of a kind.
- dense weight/bias → `IR.weight_grad_bridge` / `IR.bias_grad_bridge`
- conv  weight/bias → `conv_weight_grad_bridge` / `conv_bias_grad_bridge`
- BN input grad     → `StableHLO.bnBack_faithful` (= `bn_input_grad_correct`, under `0<ε`)

---

## 1. CNN render — the remaining piece

**Done:** the close. `cnn_render_conv{W,b}_certified` (conv) + the M2 dense bridges
cover all ten params of `cnnTrainStepText`. The hand-written `cnnTrainStepText`
(`StableHLO.lean:2026`) already GPU-trains to ~99% (`mnist-cnn-verified`).

**To do:** `cnnTrainStepStructured`, the analogue of `mlpTrainStepStructured`.

**The one real difficulty — the flat↔4-D boundary.** The MLP was all-flat, so
`pretty`'s flat-typed SHlo result names fed the templates directly. The CNN's conv
param grads (`convWGrad %dW2 %ac1 %dhc2`, `convWGrad %dW1 %xr %dhc1`) need the **4-D
NCHW** activations `%xr`/`%ac1` and 4-D cotangents `%dhc1`/`%dhc2`, which are *not* the
flat results `pretty(flatConvF)` returns.

**Step 1 — investigate the conv rendering.** Read `emitTok` for `.flatConvF`
(`StableHLO.lean:1439`) and `.maxPoolF` (`:1452`). Determine: does `pretty` of the conv
chain keep the tensor 4-D between convs (with reshapes only at the `%x`→NCHW entry and
the maxpool→flat exit, as `cnnTrainStepText` does at `:2112`,`:2116`), or does it reshape
per conv? This decides whether the 4-D activation `%ac1` is a capturable intermediate or
must be hand-emitted.

**Step 2 — render the forward, capturing the names the tail needs.** Mirror
`mlpTrainStepStructured`, but the captured-name set is larger: the conv pre-acts/acts
(`%hc1,%ac1,%hc2,%ac2`, 4-D), `%pool`, `%flat`, then the dense pre-acts/acts
(`%h3,%a3,%h4,%a4`) and `%logits`, plus the cotangent `%dy`. Two viable tactics:
- **(a) Pretty the forward as `cnnFwdGraph` pieces**, capturing names — works for the
  dense tail directly; for the conv part, capture the 4-D intermediate names exposed by
  `emitTok` (per Step 1).
- **(b) Hand-emit the conv forward** (reshape + conv + relu, controlled names) paired
  with its den via `flatConvF_faithful`/`reluF_faithful`/`maxPoolF_faithful`, and pretty
  only the all-flat dense tail. This sidesteps the 4-D-capture question and is likely the
  faster path; the conv op text already exists in `cnnTrainStepText`'s `convFwd`/`maxpoolFwd`
  helpers.

**Step 3 — emit the tail templates** (backward + param grads + SGD) from
`cnnTrainStepText:2126-2148` (`convBack`/`convWGrad`/`convBiasGrad`/`scatter`/`selMask4`
/`dg`/`reduce0`/`sgd`), substituting the captured names. The select-and-scatter
(maxpool back) and transpose-trick conv grads are verbatim-reusable.

**Step 4 — validate** (see §3): swap-train `mnist-cnn-verified`, expect ~98-99%, restore.

**Optional polish — upgrade the conv close to the chain.** `cnn_render_conv*_certified`
is generic in the cotangent. To pin the cotangent to the *actual* CNN backward chain
(the analogue of `mlpCotOut0/1`), build the `Back3` cotangent subgraphs
(`convBackDenote`/`maxPoolBackDenote` + `denote_subst3`, `IR.lean:447-554`) crossing the
flatten/maxpool boundary, and instantiate the bridges at them. Not required for "closed."

---

## 2. CIFAR — same treatment (non-BN, then BN)

CIFAR is two variants. Forwards: `cifarCnnForward` (`CifarCNN.lean:45`, 7 weight layers
W₁..W₇ — 4 conv across 2 stages + 3 dense, two maxpools) and `cifarCnnBnForward`
(`:448`, + per-example BN after each conv). Train steps: `cifarTrainStepText`
(`StableHLO.lean:2169`), `cifarBnTrainStepText` (`:2331`); exes `cifar-verified`,
`cifar-bn-verified`; files `verified_mlir/cifar{,_bn}_{fwd,train_step}.mlir`.

### 2a. CIFAR non-BN
- **Close — mostly free.** `cnn_render_conv{W,b}_certified` are **generic in dims**, so
  they already cover CIFAR's four conv layers; the three dense layers reuse the M2 bridges.
  So the CIFAR-non-BN close is: instantiate the same theorems (optionally add a thin
  `cifar_render_*_certified` wrapper for discoverability). Nearly nothing new.
- **Render.** Identical difficulty to the CNN render (flat↔4-D), just two conv stages and
  a second maxpool. Once `cnnTrainStepStructured` exists, CIFAR is a re-parameterization
  (more layers, same op kinds). Reuse `cifarTrainStepText`'s templates.

### 2b. CIFAR BN (the one genuinely-new ingredient)
- **Close — add the BN param/input grads.** Per-example BN inserts after each conv:
  forward `bnForward` (reduce mean/var, scalar γ/β); its input-VJP is the consolidated
  3-term gradient, proven faithful by `StableHLO.bnBack_faithful` (`:546`, =
  `bn_input_grad_correct`, under `0<ε`). The BN **parameter** grads (∂L/∂γ, ∂L/∂β) are the
  remaining bridges to state (the γ/β reductions) — check whether `bn_input_grad_correct`'s
  neighbours already supply them; if not, they're the BN analogue of `bias_grad_bridge`
  (a reduce), provable from the BN `pdiv` lemmas (`pdiv_bnAffine` et al.). The conv/dense
  closes are unchanged.
- **Render.** Add the BN forward op (reduce+normalize+affine) and the BN-back op
  (`bnBack`, the renderable 3-term gradient) to the name-threaded render, between conv and
  relu. `bnBack_faithful` is the den-faithfulness; the op text is in `cifarBnTrainStepText`.
  Carry the `0<ε` side-condition.

---

## 3. Validation recipe (GPU, reused all session)

Infra is under `/home/skoonce/lean/claude_max` (see memory
`running-verified-trainers-locally`):
- FFI: `cp claude_max/lean4-jax/ffi/libiree_ffi.so verify-v2/ffi/` (gitignored; run from
  repo root for the `-rpath ./ffi`).
- `iree-compile`: `claude_max/lean4-jax/.venv/bin` on PATH.
- data: `claude_max/mnist-lean4/data` (MNIST); CIFAR-10 lives at
  `claude_max/mnist-lean4/data/cifar-10` (point the exe's data arg there).
- GPU: RX 7900 XTX = **gfx1100**, `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0`. (CPU
  `llvm-cpu` FAILS — the prebuilt FFI defaults to the HIP device.)

Per render: `#eval` the renderer to `/tmp/x.mlir`; `iree-compile … --iree-rocm-target=gfx1100
--iree-codegen-llvmgpu-use-reduction-vector-distribution=false -o /tmp/x.vmfb` (expect exit 0);
then **swap-train-restore**: back up `verified_mlir/<net>_train_step.mlir`, copy the render
over it, `rm .lake/build/<slug>_ts_v.vmfb`, run the `*-verified` exe on the data dir, confirm
epoch-by-epoch parity with the committed hand-written renderer, then restore the committed
file (and confirm `diff` byte-identical). The committed `.mlir` must never change.

Each new theorem: add `#print axioms` to `tests/AuditAxioms.lean` and confirm the closure
stays `[propext, Classical.choice, Quot.sound]` (CI gate `proofs.yml`).

---

## 4. Suggested order

1. CNN render Step 1 (emitTok investigation) — decides the approach. Smallest, highest-leverage.
2. CNN render Steps 2-4 — the one real new piece this whole handoff turns on.
3. CIFAR non-BN: re-parameterize the CNN render; close is ~free (generic conv bridges).
4. CIFAR BN: add the BN forward/back ops to the render + the BN param-grad bridges to the close.
5. (Optional) upgrade all conv closes from generic-cotangent to the `Back3` chain.

Everything after Step 2 is re-parameterization + the single BN ingredient; the conv
render is the only genuinely-new rendering work left in the train-step program.
