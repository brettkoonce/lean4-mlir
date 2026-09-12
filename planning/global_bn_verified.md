# global_bn_verified.md — all-reducing the BatchNorm statistics in the verified renders

**Opened 2026-09-12**, out of §6's phase-4 half. Every verified ImageNet pair that has a
BatchNorm net carries the same unmeasured confound, and this is the plan to remove it — or to
bound it cheaply and decide it is not worth removing.

---

## 0. The finding, and why it is fleet-wide rather than per-net

⭐⭐ **`allReduceMeanF` is the ONLY constructor in `SHlo` that takes a replica family.**

    | allReduceMeanF {n : Nat} (R : Nat) (hR : 0 < R) (t : String) (ds : List Nat)
        (g : Fin R → SHlo n) : SHlo n
    den (.allReduceMeanF R _ _ _ g) = fun i => (1 / (R : ℝ)) * ∑ r : Fin R, den (g r) i

Every BatchNorm constructor — `bnBatchF`, `bnBatchBack`, `bnBatchLABack`, `bnF`, `bnBack`,
`bnPerChannelF`, `bnPerChannelBack`, `bnPerChannelEvalF` — has the shape `SHlo n → SHlo n`.
A single expression. No replica family, so **structurally nothing to reduce over**.

⇒ **No verified render can compute global BN statistics, for any net, by construction.** This is
not an omission that some net's emitter forgot; it is a property of the AST. Do not go looking
for the one net that got it right.

Confirmed independently in the artifact: in `mobilenetv2in_rmsdp64bf16_train_step.mlir` every
`"stablehlo.all_reduce"` is named `%arsum<param-tag>` (`W`, `g`, `bt`, …), the first appears at
line 6964 of 12037, and the forward pass (lines 54–6963) contains no collective at all.

## 0b. And every JAX reference is on the other side of it

Audited 2026-09-12 across `jax/generated/`: **`pmap` 0, `shard_map` 0, `NamedSharding` everywhere,
`train_step` under `@jit`**, data placed with `NamedSharding(mesh, P('batch'))`. Under `@jit` the
program is written against the GLOBAL array, so `jnp.mean(x, axis=(0,2,3))` has global semantics
and GSPMD inserts the collective. ⚠ **The absence of an explicit `lax.pmean` is not evidence of
per-device statistics — it is what global semantics look like under GSPMD.** Had these been
`pmap`, that same line would be per-device and there would be no confound at all.

## 0c. Who is affected

| net | render | JAX ref | confound | disclosed in the book |
|---|---|---|---|---|
| ResNet-34 | BN, per-replica 64 | global 256 | yes | ✅ `BN statistic group` row, §5 |
| ResNet-50 | BN, per-replica 64 | global 512 | yes | ✅ `BN group` column; called "the largest known difference between the two paths, and *untested*" |
| MobileNetV2 | BN, per-replica 64 | global 256 | yes | ✅ §6, 2026-09-12 |
| MobileNetV4 | BN, per-replica | global | yes | ▶ not run yet — will need the row |
| EfficientNet-B0 | BN, per-replica | global | yes | ▶ not run yet — BN-dense, needs it most |
| ConvNeXt-T | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |
| ViT-Ti | LayerNorm, no `bnBatchF` | no batch stats | **no** | n/a |

⚠ LayerNorm normalizes per example over channels, so it is replica-invariant by construction.
ConvNeXt and ViT are the only two nets whose pair isolates the lowerer, and that is *why*.

---

## 1. ⭐ DO THIS FIRST — bound the effect without touching a single proof

The question the confound raises is "how much of the verified-vs-reference delta is BN grouping?"
That can be answered entirely on the JAX side, with **zero** proof work:

▶ Take the reference trainer, keep everything identical, and make BN normalize **per shard**
  (group 64) instead of globally — `shard_map` over the batch axis, or an explicit reshape to
  `[R, N/R, …]` with the mean taken over the inner axis only. Train. Compare against the same
  reference at global BN.

**reference@BN64 − reference@BN256 IS the confound, measured**, on the same lowerer, same seed
discipline, same everything. Whatever is left over in the verified-vs-reference delta is not BN.

⚠ This is what §5's `sec:r34_pjrt` effectively did for ResNet-34 to get its $0.02$ figure. It is
cheap, it is a JAX-only diff, and **it may well close this whole thread** — if the effect is
hundredths on the BN-dense nets too, the render work below is not worth doing.

⚠ Do it at short schedule first (Imagenette, or ~30 ImageNet epochs) to get the sign and the order
before spending a 50-hour run on it.

---

## 2. If the measurement says it matters — the render work

⛔ **This is a new node, a new spec and a re-proof, not a patch.** Scoped 2026-09-12.

### 2a. A new AST node
`bnBatchDpF` (name TBD): BatchNorm whose mean and variance are replica-means. It must take the
replica family `(Fin R → SHlo …)` the way `allReduceMeanF` does, because the collective has to sit
**inside** the BN node, around the statistics — you cannot compose `allReduceMeanF` with
`bnBatchF`, since the latter consumes a single expression and computes its statistics internally.

⚠ Per the repo's own rule, adding an `SHlo` op is **~10 sites across 2 files**: the inductive,
`den`, `skel`/`toToks`, the parser round-trip (⛔ under `Certs` — a bare `lake build` misses it),
`pretty`/`emitTok`, shape lemmas. Budget the parser round-trip explicitly; it is the one that gets
forgotten.

### 2b. Two collectives per BN layer, not one
Variance needs $E[x^2] - E[x]^2$, so both moments are reduced. MobileNetV2 has **52 BN layers**
(stem + 2 in block 1 + 3 × blocks 2–17 + head) ⇒ **~104 collectives added to the forward pass**.
EfficientNet-B0 is denser still.

### 2c. The backward changes too
Under global BN the input gradient carries terms summed over the global batch, so the backward
needs its own collectives, and the existing certified BN-backward lemmas (`bnBatchBack`,
`bnBatchLABack`) do not cover that form. This is a second node, not a free consequence of 2a.

### 2d. ⛔ The spec moves — this is the real bill
Global BN is a **different mathematical function** from local BN. The per-net forward specs, the
§1a ties, the whole-net backward ties and the StepTie theorems are all stated against the current
spec. Every one of them is restated and reproved for any net that adopts the new node.

▶ Sequence it as: one net first (MobileNetV2 or ResNet-34, whichever has the tidier tie chain),
land it end to end including the parser round-trip and a numerical check against JAX at
global BN, and only then consider the others.

### 2e. It will be slower
~104 synchronizing collectives per step in the forward plus the backward's own, on nets whose step
is already feed-bound. Expect the verified column's throughput to regress. That is a real cost to
weigh against a confound that §1's measurement may show to be hundredths of a point.

---

## 3. What "done" looks like

* A BN net whose verified render normalizes over the global batch, tied to a spec that says so.
* Its pair re-run, and the side-by-side's `BN statistic group` row reading the same on both sides.
* Only then may a chapter say the remaining difference between the two columns is the lowerer.
  ⚠ Until then, §5, §6 and any future BN chapter say "agree to within the BatchNorm-group effect".

## 4. Related

* `sec:r34_pjrt` — the $0.02$ measurement on ResNet-34 at a 4× group difference.
* §5's ResNet-50 RSB-A3 discussion — an 8× gap, bounded at $-0.28$ and explicitly *untested*.
* `LeanMlir/Proofs/Foundation/DataParallelNode.lean` — `den (allReduceMeanF …) = lossGrad (meanLoss L) θ`,
  the theorem that makes the gradient collective correct. The BN analogue does not exist.
