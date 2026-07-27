# xla_pjrt_handoff.md — where the XLA/PJRT work stands, and what to do next

**Written 2026-07-27.** Handoff for a fresh session. The full history, gate
definitions, and every measurement live in `planning/xla_pjrt_ladder.md`; this
file is the short version — state, next moves, and the things that cost time to
learn the first time.

Branch **`xla-pjrt-backend`**, four commits on top of `cfbdccd`:

| commit | what |
|---|---|
| `b44caaa` | PJRT C-API backend; ladder rungs 0–3 |
| `20dfd29` | emit cross-replica `all_reduce` for data-parallel AdamW |
| `9a7957a` | `String.dropEnd` deprecation fix |
| `b5b843e` | N-replica execute; data-parallel validated against 1×N |

---

## 1. What works today

**A second trusted lowerer, with no Python at run time.** `ffi/pjrt_ffi.c`
implements the same C surface as `ffi/iree_ffi.c` (symbol-identical under
`nm -D`), so nothing above the shim changed; the backend is whichever `.so` is
linked. Backend detection is a **weak symbol**, so a binary cannot disagree with
the library it linked.

**Rungs 0–3 complete** — linear, MLP, CNN, cifar8-bn+Adam, ResNet-34/Imagenette
(513 in / 513 out, 36 BN layers). **No rung ever needed new shim code**: BN
running stats and rank-0 scalars all ride `iree_ffi_invoke_f32`. The
`train_step_adam*` family in `iree_ffi.h` is still stubbed and was never reached.

**Data-parallel multi-GPU**, validated on cifar8 (no BN) where the
batch-decomposition identity is exact: 1×256 vs 2×128+all_reduce agree on the
gradient to **1.015e-06**.

**Speed, R34/Imagenette bs32:** IREE 1702 → XLA **162 ms/step** (10.5×);
52.5 s/epoch. Within **1.04×** of hand-written JAX per step at bs32, but see §3.

### Running it

```bash
gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so
lake build resnet34-verified-adam-xla
HIP_VISIBLE_DEVICES=0 .lake/build/bin/resnet34-verified-adam-xla data

# multi-GPU: render at REPLICAS=N first (tests/TestResnet34Train.lean), then
unset HIP_VISIBLE_DEVICES
LEAN_MLIR_REPLICAS=2 PJRT_REPLICAS=2 .lake/build/bin/resnet34-verified-adam-xla data
```

Env knobs added by this work: `PJRT_REPLICAS`, `PJRT_PLUGIN`, `PJRT_FFI_TRACE`,
`LEAN_MLIR_REPLICAS`, `LEAN_MLIR_SKIP_EVAL`, `LEAN_MLIR_G2_STEPS`,
`LEAN_MLIR_DUMP_PARAMS`, `LEAN_MLIR_PERTURB_R`.

---

## 2. Next up

### 2a. Move the imagenette `_fwd` renders from `tests/` into `Proofs/` ▶ IN FLIGHT

The provenance audit (`xla_pjrt_ladder.md` §8, and the tables below) found the
repo splits cleanly along an axis nobody chose deliberately:

| artifact class | rendered by |
|---|---|
| mnist/cifar `_fwd` and `_train_step` (SGD) | **`Proofs/Codegen/`** — `pretty(emit g)` of a proven graph |
| imagenette `_train_step` (SGD) | **`Proofs/Codegen/`** |
| imagenette `_fwd`, `_fwd_eval` | `tests/` |
| **every** `_adam_train_step`, all nets | `tests/` |

The dividing line is **Adam**, not the net: `emitAdamV` lives in
`LeanMlir/ViTRender.lean` (a hand-written string emitter) and is only ever called
from `tests/`. The `_fwd` split is the smaller, more tractable half — moving the
imagenette forwards into `Proofs/Codegen/` makes them match mnist/cifar.

**Scope, checked 2026-07-27: this is a rewrite, not a lift-and-shift.**
`tests/TestResnet34Fwd.lean` (308 lines) carries its *own* hand-written string
fragments — `conv`, `convStem`, `bnPC`, `relu`, `maxpool`, `addOp`, `idBlock`,
`downBlock`, `idChain` — and barely references `Proofs.` at all.
`LeanMlir/Proofs/Codegen/ResNet34Render.lean` (307 lines) independently has
`idFwd`/`downFwd` built on the proven graph ops and renders through `pretty`, but
it only emits the **train step**; there is no forward-only module. So the two are
parallel independent implementations of the same forward, which is precisely the
drift risk worth closing.

The work is: assemble a forward-only `#eval` in `ResNet34Render.lean` from the
fragments it already has, then point `verified_mlir/resnet34_fwd.mlir` at it and
retire the tests copy. The fiddly part is the **signature** — the driver feeds
parameters in `net.paramShapes` order (147 inputs for `resnet34_fwd`, 219 for
`resnet34_fwd_eval`, the latter taking running BN stats named `...mu`/`...var`).
Diff the new render against the committed artifact before switching over.

Watch out for:
- **Seven artifacts already have two writers** with no consistency check
  (`resnet34_train_step.mlir` ← both `Proofs/ResNet34Render.lean:306` and
  `tests/TestResnet34Train.lean:464`; also convnext, efficientnet, mobilenetv2,
  cifar8, and both vit files). Whoever runs last wins, silently. Moving a render
  is a good moment to add the diff check.
- `LeanMlir/Proofs/Codegen/ViTRender.lean` is the existing **drift guard** — it
  renders the same forward via `pretty` of the proven graph. Copy that pattern.
- `vit_adam_train_step.mlir` is written by the **trainer app itself** at run time
  (`apps/imagenette/MainViTVerifiedAdam.lean:31`) — unique lifecycle, don't be
  surprised by it.
- **There is no canonical regeneration entry point.** No lake target or script
  rebuilds `verified_mlir/`; two `lakefile.lean` comments describe it in prose.
  Worth adding while you are in here.

### 2b. R34 on 2 GPUs ✅ RUNS — 1.46×, and the shortfall is diagnosed

`tests/TestResnet34Train.lean` has a `REPLICAS` knob (at 1 it re-renders
byte-identical; at 2 it emits 146 collectives, one per parameter). Measured,
3 epochs of Imagenette across both 7900 XTXs:

| | steps/epoch | s/epoch | ms/img |
|---|---|---|---|
| 1 GPU, bs32 | 295 | 52.5 | 5.06 |
| **2 GPU, global 64** | 147 | **~36** | **3.81** |

**1.46×, not 2×** — and the cause is already identified in `xla_pjrt_ladder.md`
§10.3a: parameters are still host-resident, so each step pushes the full 272 MB
`[θ|m|v]` to *every* replica. Compute halves while transfer doubles.
**Device-resident parameters is a prerequisite for multi-GPU scaling, not an
independent optimisation** — this measurement is the evidence.

It learns: val 36.4 / 41.9 / **49.4%** over three epochs. Lower per-epoch than the
single-GPU run (which reached ~51.7% by epoch 2) because global batch 64 with an
**unscaled** LR does half as many steps per epoch. That is the expected
large-batch recipe cost, not a defect — apply the linear-scaling rule before
comparing convergence.

A gradient check against 1×64 would be **inexact by design** here: BN normalises
per replica, so 2×32 ≠ 1×64 (§10.3b). The exact check lives on cifar8 (no BN),
where it passed at 1.015e-06.

Also fixed while running this: **the replica count is per-graph, not
per-process.** A module with no collective (the eval forwards) is compiled
single-replica; otherwise `Execute` rejects it with *"Attempted to execute with 1
argument lists when local device count is 2"*. The single-device invoke now also
refuses a multi-replica executable rather than mis-executing it.

### 2c. Then, in value order

1. **bs256 re-render + measure.** Batch is worth **1.8×** on this net
   (5.06 → 2.87 ms/img from bs32 → bs256, measured), it is a one-line `BS` edit,
   and bs256 **fits** on a 7900 XTX. Needed for ImageNet anyway.
2. **Rung 4** — the FPN detector, and the 35.5× headline nobody has verified
   end to end.
3. **Device-resident parameters.** Two rounds of transfer work are already done
   (batching: 256→205 ms; killing the per-step host memcpys: 205→162 ms). What
   remains is smaller than it looks — see §3.
4. **Executable cache** (`PJRT_Executable_Serialize` / `DeserializeAndLoad`).
   Worth **0.1%** on an R34 training run and **53%** on the MNIST-MLP demo: a
   dev-loop and CI win, not a throughput one.

---

## 3. Corrections a new session should not have to re-derive

**"Within 1.04× of JAX" was measured in a data-bound regime and does not
generalise.** JAX on Imagenette is flat at 46.0 / 45.0 / 44.1 s per epoch for
bs32 / bs192 / bs256 — both paths idle the GPU on data loading. Compute-bound at
bs256 the honest number is **1.42×** (711 vs 501 ms/step, same box, same net,
synthetic data, `jax/scripts/jax_r34_bf16_bench.py 256`).

**XLA is NOT deterministic at epoch scale.** It is bit-identical run-to-run for a
*single step*, which is why the 1-step gate is sound — but three runs of the same
binary at the same seed gave epoch-1 val accuracy of 43.21 / 46.80 / 47.29%. Run
determinism controls **at the scale you care about**; a 1-step check is necessary,
not sufficient.

**R34's raw G2 number (gradient rel 5.20e-02) is not a defect.** The gradient
does not reproduce to better than ~6e-3 against the *same backend* under a
sub-ULP nudge, so a 1e-4 gate fails XLA-vs-XLA by 60×. Correctness comes from the
layer-level oracle instead: all six R34 layer families tie to JAX autodiff at
≤ 6.4e-06 (`tests/vjp_oracle/run.sh`).

**bf16 is worse than fp32 on this box** — measured ×0.96 for R34. Do not reach
for it here. It matters on ares only if someone measures it there.

**For Adam nets, gate G2 on the gradient (`m` after one step), never on θ.**
Adam's update is scale-free, so a near-zero-gradient parameter flips sign on a
1-ULP difference and moves a full ±lr. θ lands at ~1e-4 regardless of whether
anything is wrong.

---

## 4. Gotchas that cost time

- **`use_global_device_ids` must NOT be set** on `stablehlo.all_reduce`. It needs
  a positive `channel_id`; without one, compilation fails with *"channel_id must
  be positive when useGlobalDeviceIds is set"*.
- **Multi-GPU needs `HIP_VISIBLE_DEVICES` unset**, or the client sees one device.
  `ffi/test_pjrt_dp.c` and `test_pjrt_allreduce.c` refuse rather than silently
  running single-replica.
- **Empty `compile_options` is not "defaults"** — proto3 zeros give
  `replica_count = 0` and XLA aborts. Hence the generated blob table.
- **`.venv/bin/python3` must be a wrapper, not a symlink.** A symlinked
  interpreter derives `sys.prefix` from the symlink's location and cannot find
  jax. (Also: `python3 -c 'import jax'` from the repo root imports the local
  `jax/` *directory*.)
- **Checkpoints and `.vmfb` paths are now backend-scoped.** They were shared, so
  an XLA run could resume from, or reuse, an IREE artifact while looking normal.
  Any new driver with resume needs the same treatment.
- **`git stash -u` here would write ~17 GB** (`runs/` 5 GB, `figures/` 12 GB)
  into `.git`. Stash tracked modifications only; use `.gitignore` for the rest.
- Rendering at a non-default batch breaks eval unless the forward graph is
  re-rendered too — that is what `LEAN_MLIR_SKIP_EVAL` is for.

---

## 5. The claim ceiling

Adding a collective does not make this "verified multi-GPU", and nothing here
should be described that way. Each replica evaluates the *same tied graph* at the
batch size it was rendered for, and the collective averages gradients of that
function over disjoint equal batches. The strongest honest statement is:

> the gradient averaging is a proven identity; the collective implementing it is
> trusted, exactly like the lowerer.

And prefer **scaling** the global batch over **splitting** a fixed one: scaling
keeps each replica's BatchNorm group at the size it was tied at, so the BN caveat
never arises (`xla_pjrt_ladder.md` §10.3b).
