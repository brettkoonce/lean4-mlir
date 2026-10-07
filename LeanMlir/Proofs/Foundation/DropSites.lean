import LeanMlir.Proofs.Foundation.DataParallel.Sync

/-! # Optional drop sites — stochastic depth and classifier dropout as `Option` binders

A verified render carries stochastic depth and classifier dropout as HOST-DRAWN mask inputs
(`Proofs.Training.DropPath`: the keep probability is folded into the mask, the op is a diagonal
scale, its VJP is itself). Whether a site is RENDERED is a renderer flag (`sd`, `cd`), so a graph
or chain statement that covers both the drop-free artifact and the `*drop*` / `*do*` one takes
the site as an `Option`: `none` emits no node and denotes the identity, `some s` emits the
`dropPathB` / `dropoutB` node at the mask `s`. The forward graphs state their sites this way
(`Proofs.EfficientNetFullB0Drop`'s `efficientnetFwdGraphBFullDrop`); the step, sync and
loss-gradient ties thread the same binder through their cotangent chains
(`Proofs.MobileNetV2TieB`'s classifier dropout first), so the drop-free tie is the `none`
instance verbatim and the `some` instance is the artifact that trained.

* `dropPathOpt` / `dropoutOpt` — the real-valued site; `_none` is `rfl`, `_ones` the eval
  identity (`dropPath_ones_id`, `dropout_ones_id`).
* `dropPathOptG` / `dropoutOptG` — the graph node; `den_*` by cases on the site.
* `dropoutOptHasVJP` / `dropPathOptHasVJP` — the VJP, its backward THE OP ITSELF at the same
  mask at either value (`dropout_vjp_is_self` one `Option` up), stated as the `backward` field so
  a chain lemma that is `rfl` at the drop-free chain stays `rfl` with the site in; the
  differentiability each tie's `HasGradAt.comp` needs beside it.
* `dropoutOpt_smul`, `dropoutOpt_shard` — what a data-parallel tie needs: the site is linear in
  the cotangent and commutes with the batch cut when replica `r` holds shard `r` of the mask, as
  the DP renders' per-replica `%do` inputs are. `dropPathOpt_smul`, `dropPathOpt_shard` are the
  per-example-mask peers: a stochastic-depth mask is a `Vec N` of scalars, cut by `exampleShard`
  (`batchShard`'s cut at width one), and `dropPathOptFam` is the replica family a DP chain feeds
  a drop site, with its `_shard` and `_scaled` (the invariant the sync ties carry block to block).
-/

namespace Proofs

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § The sites — `none` is the identity, as the renderer emits nothing
-- ════════════════════════════════════════════════════════════════

/-- Drop-path at a site that may be absent: `none` is the identity. -/
noncomputable def dropPathOpt (N n : Nat) : Option (Vec N) → Vec (N * n) → Vec (N * n)
  | none => id
  | some s => dropPath N n s

/-- Dropout at a site that may be absent: `none` is the identity. -/
noncomputable def dropoutOpt {m : Nat} : Option (Vec m) → Vec m → Vec m
  | none => id
  | some mk => dropout mk

@[simp] theorem dropPathOpt_none (N n : Nat) : dropPathOpt N n none = id := rfl

@[simp] theorem dropPathOpt_some (N n : Nat) (s : Vec N) :
    dropPathOpt N n (some s) = dropPath N n s := rfl

@[simp] theorem dropoutOpt_none {m : Nat} : dropoutOpt (none : Option (Vec m)) = id := rfl

@[simp] theorem dropoutOpt_some {m : Nat} (mk : Vec m) : dropoutOpt (some mk) = dropout mk := rfl

/-- At the all-ones scale drop-path is the identity — the masks the driver passes at eval. -/
theorem dropPathOpt_ones (N n : Nat) : dropPathOpt N n (some fun _ => 1) = id :=
  funext (dropPath_ones_id N n)

/-- At the all-ones mask dropout is the identity. -/
theorem dropoutOpt_ones {m : Nat} : dropoutOpt (some (fun _ => 1 : Vec m)) = id :=
  funext dropout_ones_id

-- ════════════════════════════════════════════════════════════════
-- § The VJP — the op itself at the same mask, at either value
-- ════════════════════════════════════════════════════════════════

theorem dropoutOpt_differentiable {m : Nat} (cd : Option (Vec m)) :
    Differentiable ℝ (dropoutOpt cd) := by
  cases cd with
  | none => exact differentiable_id
  | some mk => exact layerScale_differentiable mk

theorem dropPathOpt_differentiable (N n : Nat) (sd : Option (Vec N)) :
    Differentiable ℝ (dropPathOpt N n sd) := by
  cases sd with
  | none => exact differentiable_id
  | some s => exact layerScale_differentiable (dropScale N n s)

/-- **The backward is the forward at the same mask, at either value** — stated as the witness's
    `backward` field, not derived by cases, so `(dropoutOptHasVJP cd).backward x dy` unfolds to
    `dropoutOpt cd dy` for a symbolic `cd`. -/
noncomputable def dropoutOptHasVJP {m : Nat} (cd : Option (Vec m)) : HasVJP (dropoutOpt cd) where
  backward := fun _ dy => dropoutOpt cd dy
  correct := by
    intro x dy i
    cases cd with
    | none => exact (identityHasVJP m).correct x dy i
    | some mk => exact (dropoutHasVJP mk).correct x dy i

theorem dropoutOpt_vjp_is_self {m : Nat} (cd : Option (Vec m)) (x dy : Vec m) :
    (dropoutOptHasVJP cd).backward x dy = dropoutOpt cd dy := rfl

/-- `dropoutOptHasVJP` one rank down — the per-example scale. -/
noncomputable def dropPathOptHasVJP (N n : Nat) (sd : Option (Vec N)) :
    HasVJP (dropPathOpt N n sd) where
  backward := fun _ dy => dropPathOpt N n sd dy
  correct := by
    intro x dy i
    cases sd with
    | none => exact (identityHasVJP (N * n)).correct x dy i
    | some s => exact (dropPathHasVJP N n s).correct x dy i

theorem dropPathOpt_vjp_is_self (N n : Nat) (sd : Option (Vec N)) (x dy : Vec (N * n)) :
    (dropPathOptHasVJP N n sd).backward x dy = dropPathOpt N n sd dy := rfl

-- ════════════════════════════════════════════════════════════════
-- § Data parallel — linear in the cotangent, commutes with the batch cut
-- ════════════════════════════════════════════════════════════════

/-- The site is linear in what flows through it. -/
theorem dropoutOpt_smul {m : Nat} (cd : Option (Vec m)) : IsHomog (dropoutOpt cd) := by
  intro s v
  cases cd with
  | none => rfl
  | some mk =>
    funext i
    show mk i * (s * v i) = s * (mk i * v i)
    ring

/-- Replica `r` applying ITS shard of the mask to ITS shard of a value is shard `r` of the global
    site — `batchShard_zipWith` at the site's multiply. -/
theorem dropoutOpt_shard {R N a : Nat} (cd : Option (Vec ((R * N) * a))) (X : Vec ((R * N) * a))
    (r : Fin R) :
    dropoutOpt (cd.map fun M => batchShard R N a M r) (batchShard R N a X r)
      = batchShard R N a (dropoutOpt cd X) r := by
  cases cd <;> rfl

/-- Replica `r`'s block of a per-example scalar family laid out `[R·N]`: example `(r, n)` of the
    global batch is example `n` of shard `r` — `batchShard`'s cut at width one, for the
    stochastic-depth masks (`Vec N` per site, not `Vec (N * 1)`). The DP renders' per-replica
    `%dp<i>` inputs are these. -/
noncomputable def exampleShard (R N : Nat) (S : Vec (R * N)) (r : Fin R) : Vec N :=
  fun n => S (finProdFinEquiv (r, n))

/-- The per-example site is linear in what flows through it. -/
theorem dropPathOpt_smul (N n : Nat) (sd : Option (Vec N)) : IsHomog (dropPathOpt N n sd) := by
  intro s v
  cases sd with
  | none => rfl
  | some sc =>
    funext i
    show sc (finProdFinEquiv.symm i).1 * (s * v i) = s * (sc (finProdFinEquiv.symm i).1 * v i)
    ring

/-- Replica `r` scaling ITS shard of a value by ITS shard of the per-example mask is shard `r` of
    the global site: the example a cell belongs to is the same on both sides of the cut
    (`Equiv.symm_apply_apply` at `batchShard`'s index). -/
theorem dropPathOpt_shard {R N n : Nat} (sd : Option (Vec (R * N))) (X : Vec ((R * N) * n))
    (r : Fin R) :
    dropPathOpt N n (sd.map fun S => exampleShard R N S r) (batchShard R N n X r)
      = batchShard R N n (dropPathOpt (R * N) n sd X) r := by
  cases sd with
  | none => rfl
  | some S =>
    funext i
    simp only [Option.map_some, dropPathOpt_some, dropPath_apply, batchShard, exampleShard,
      Equiv.symm_apply_apply]

/-- `dropPathOpt_shard` at a rendered site. -/
theorem dropPath_shard {R N n : Nat} (S : Vec (R * N)) (X : Vec ((R * N) * n)) (r : Fin R) :
    dropPath N n (exampleShard R N S r) (batchShard R N n X r)
      = batchShard R N n (dropPath (R * N) n S X) r :=
  dropPathOpt_shard (some S) X r

/-- The replica family a data-parallel chain feeds a drop site: replica `r`'s cotangent through
    replica `r`'s shard of the mask. -/
noncomputable def dropPathOptFam (R N n : Nat) (sd : Option (Vec (R * N)))
    (dys : Fin R → Vec (N * n)) : Fin R → Vec (N * n) :=
  fun r => dropPathOpt N n (sd.map fun S => exampleShard R N S r) (dys r)

theorem dropPathOptFam_shard {R N n : Nat} (sd : Option (Vec (R * N))) (dys : Fin R → Vec (N * n))
    (DY : Vec ((R * N) * n)) (hdys : ∀ r, dys r = batchShard R N n DY r) :
    ∀ r, dropPathOptFam R N n sd dys r = batchShard R N n (dropPathOpt (R * N) n sd DY) r := by
  intro r
  unfold dropPathOptFam
  rw [hdys r, dropPathOpt_shard]

/-- **The scaled-shard invariant passes a drop site**: replicas at `R ×` the shards of a global
    cotangent hand the branch `R ×` the shards of the global branch cotangent. -/
theorem dropPathOptFam_scaled {R N n : Nat} (sd : Option (Vec (R * N))) (dys : Fin R → Vec (N * n))
    (DY : Vec ((R * N) * n))
    (hdys : ∀ r, dys r = batchShard R N n (fun i => (R : ℝ) * DY i) r) :
    ∀ r, dropPathOptFam R N n sd dys r
      = batchShard R N n (fun i => (R : ℝ) * dropPathOpt (R * N) n sd DY i) r := by
  intro r
  unfold dropPathOptFam
  rw [hdys r, dropPathOpt_shard, dropPathOpt_smul]

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The graph nodes — a `dropPathB` / `dropoutB` exactly where the site is `some`
-- ════════════════════════════════════════════════════════════════

/-- A `dropPathB` node when the site is rendered, nothing otherwise. -/
def dropPathOptG (mN : String) {N n : Nat} : Option (Vec N) → SHlo (N * n) → SHlo (N * n)
  | none, e => e
  | some s, e => .dropPathB mN s e

theorem den_dropPathOptG (mN : String) {N n : Nat} (s : Option (Vec N)) (e : SHlo (N * n)) :
    den (dropPathOptG mN s e) = dropPathOpt N n s (den e) := by
  cases s <;> rfl

/-- A `dropoutB` node when the site is rendered, nothing otherwise. -/
def dropoutOptG (mN : String) {N n : Nat} : Option (Vec (N * n)) → SHlo (N * n) → SHlo (N * n)
  | none, e => e
  | some m, e => .dropoutB mN m e

theorem den_dropoutOptG (mN : String) {N n : Nat} (m : Option (Vec (N * n))) (e : SHlo (N * n)) :
    den (dropoutOptG mN m e) = dropoutOpt m (den e) := by
  cases m <;> rfl

end Proofs.StableHLO
