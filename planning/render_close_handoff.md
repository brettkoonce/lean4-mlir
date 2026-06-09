# Handoff — CNN/CIFAR render+close: **COMPLETE** (next: MobileNetV2)

This continued `planning/verified_train_step.md`. Status as of the session that closed it:
**every train-step on the chapter ladder (linear → MLP → CNN → CIFAR → CIFAR-BN) is now
closed BOTH ways** — text name-threaded from proven forward graphs (GPU-validated) AND every
parameter output certified as `θ − lr·certified-gradient` (audited 3-axiom-clean). The conv
closes are additionally pinned to the actual backward chain. Everything below is DONE; the
live frontier is **MobileNetV2** → see `planning/mobilenetv2_close.md`.

All work targets the `ℝ`-level / per-op-template trusted base unchanged (see
`verified_train_step.md` §"What stays trusted"): ℝ→Float32, `iree-compile`, the op templates.
"Closed" means text↔denotation↔certified-math within that base.

---

## What got done (commit trail)

| Commit | Deliverable | Validation |
|---|---|---|
| `0e4be10` | **CNN render** — `cnnTrainStepStructured` (`LeanMlir/Proofs/CnnRender.lean`) | GPU `mnist-cnn-verified` 98.99%; backward tail byte-identical to `cnnTrainStepText` |
| `4ab917d` | **CIFAR non-BN render** — `cifarTrainStepStructured` (close was free: conv bridges generic in dims) | bit-identical to committed, all 10 epochs |
| `de5f80f` | **CIFAR-BN render** — `cifarBnTrainStepStructured` (a′-BN: BN fwd+input-grad proof-rendered) | `cifar-bn-verified` 70.65% stable; identical to committed all 10 epochs |
| `73b3a68` | **CIFAR-BN close** — `bnPerChannel_grad_{gamma,beta}_correct` (`CifarBnClose.lean`) | dγ/dβ bridges, 3-axiom clean |
| `1fd7757` | **Conv-close upgrade** — `cnn_render_conv{W2,b2,W1,b1}_chain_certified` (`CnnChainClose.lean`) | cotangent pinned to the Back3 chain, 3-axiom clean |

## The two reusable patterns (now battle-tested)

**Render (a′ — text = render of proven graphs).** Pretty the WHOLE forward (convs/BN included)
from the proven `SHlo` graph (`cnnFwdGraph`/`cifarFwdGraph`/`cifarBnFwdGraph`), capturing flat
SSA names; the conv/BN tail's **4-D NCHW** consumers are recovered by explicit `reshape` glue
(a GPU no-op). Backward/grad/SGD tail = the committed hand-written text wired to captured names
(byte-identical modulo names). For BN, the `bnPerChannelF`/`bnPerChannelBack` tokens proof-render
both the BN forward and input-grad (recompute-from-input, `0<ε`); only dγ/dβ are hand-emitted.

**Close (param outputs denote certified).** Each rendered SGD output `θ − lr·(emitted grad)`
equals `θ − lr·(certified Jacobian · cotangent)` by the layer bridge, generic in the cotangent:
- dense → `IR.weight_grad_bridge` / `bias_grad_bridge`
- conv → `conv_weight_grad_bridge` / `conv_bias_grad_bridge` (`cnn_render_conv{W,b}_certified`)
- per-channel BN → `bnPerChannel_grad_{gamma,beta}_correct` (affine in γ/β, **no `0<ε`**)
- **cotangent chain** (optional polish, done for CNN): pin the generic `c` to the actual chain
  via `Back`/`Back3` subgraphs (`cnnDenseHeadCot` / `cnnChainCotW{1,2}` in `CnnChainClose.lean`).

## Validation recipe (GPU, reused all session)

Infra under `/home/skoonce/lean/claude_max` (see memory `running-verified-trainers-locally`):
FFI `cp claude_max/lean4-jax/ffi/libiree_ffi.so verify-v2/ffi/`; `iree-compile` from
`claude_max/lean4-jax/.venv/bin` on PATH; data `claude_max/mnist-lean4/data` (+ `cifar-10`);
GPU `IREE_BACKEND=rocm HIP_VISIBLE_DEVICES=0 --iree-rocm-target=gfx1100`. Per render: `#eval`
to `/tmp`, `iree-compile`, then **swap-train-restore** (back up `verified_mlir/<net>_train_step.mlir`,
swap the render in, `rm .lake/build/<slug>_ts_v.vmfb`, run `<slug>-verified`, confirm epoch
parity, restore the committed file byte-identical — it must NEVER change). Each new theorem:
`#print axioms` in `tests/AuditAxioms.lean`, closure stays `[propext, Classical.choice, Quot.sound]`.

## What's still open (program-wide, all pre-existing)

- **MobileNetV2** (ch7) — the live frontier. See `planning/mobilenetv2_close.md`.
- The render-close → **total-loss-gradient** fold (`pdiv G = Back.denote`, the `mlp_*_total_loss_grad`
  analogue) for CNN/CIFAR — the conv chain is pinned (`CnnChainClose`) but not yet folded to `∂loss/∂θ`.
- The CIFAR non-BN epoch-10 plain-SGD divergence (demo-config; BN sidesteps it).
- The unchanged trusted base: `ℝ→Float32`, `iree-compile`, the op templates.
