module

public import LeanMlir.Proofs.Training.Optim.Lamb

@[expose] public section

/-! # Gradient accumulation over ℝ — the `acc{k}x` / `lambacc{k}x` recipes' two phases

The ℝ reference for what `RenderKit.optOne` emits under `.adamwAccum k` / `.lambAccum k`. One
graph serves both phases, two runtime scalars deciding which it is:

* **the accumulator** `Gt = akeep·G + g` is `momVNextF` at `(μ := %akeep, v := G)`, i.e.
  `Proofs.momVNext`; `%akeep` is `0` on the first micro-batch of a cycle (the reset) and `1`
  after, so after `k` micro-batches `Gt` is their SUM (`accumCycle_sum`);
* **the accumulate phase** (`%aup = 0`) feeds the moments `β = 1`, `1 − β = 0` and the
  parameter `lr = 0`: the moments and the parameter pass through unchanged
  (`adamWStepAcc_accumulate`, `lambStepAcc_accumulate`);
* **the apply phase** (`%aup = 1`) feeds the moments the hand-written pair `(1 − β₁)/k` and
  `(1 − β₂)/k²` on the SUM `Gt` (`RenderKit.accumScalarConsts`, where the `1/k²` on the second
  moment is explained). That pair at the sum IS the plain step at the MEAN `Gt/k`
  (`adamWStepAcc_eq_mean`, `lambStepAcc_eq_mean`), which is what makes accumulation a real
  large-batch step rather than a mean of per-micro-batch second moments.

**Claim boundary.** The typed graph's `adamMNextF`/`adamVNextF` carry one `β` and read the two
scalar operands as `β` and `1 − β`; the `/k` and `/k²` live in the hand-written scalar block
the train-step banner names (`trainStepHandNote acc`). So the faithfulness theorems in
`StableHLO.Basic` cover the ops at the mean, and the identities here are what connects the
emitted constants to that reading. They are ring identities over ℝ, nothing more. -/

namespace Proofs

variable {n : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The accumulator
-- ════════════════════════════════════════════════════════════════

/-- `Gt = akeep·G + g` at `akeep = 1`: the running sum grows by `g`. -/
theorem momVNext_mu_one (G g : Vec n) (i : Fin n) : momVNext 1 G g i = G i + g i := by
  simp [momVNext]

/-- At `akeep = 0` the previous total is dropped: the cycle resets by arithmetic, not by a
    separate zeroing step. -/
theorem momVNext_mu_zero (G g : Vec n) (i : Fin n) : momVNext 0 G g i = g i := by
  simp [momVNext]

/-- A cycle's accumulator, from whatever total `G₀` the previous cycle left: the first micro-batch
    `g` at `akeep = 0` (the reset), then the rest at `akeep = 1`. -/
def accumCycle (G₀ g : Vec n) (gs : List (Vec n)) : Vec n :=
  gs.foldl (fun G g => momVNext 1 G g) (momVNext 0 G₀ g)

theorem foldl_momVNext_one (gs : List (Vec n)) (G : Vec n) (i : Fin n) :
    (gs.foldl (fun G g => momVNext 1 G g) G) i = G i + (gs.map (fun g => g i)).sum := by
  induction gs generalizing G with
  | nil => simp
  | cons g gs ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih, momVNext_mu_one]; ring

/-- After the cycle, `Gt` is the SUM of its micro-batch gradients, whatever the previous total. -/
theorem accumCycle_sum (G₀ g : Vec n) (gs : List (Vec n)) (i : Fin n) :
    accumCycle G₀ g gs i = ((g :: gs).map (fun g => g i)).sum := by
  simp [accumCycle, foldl_momVNext_one]

-- ════════════════════════════════════════════════════════════════
-- § The apply phase: the hand-written pair at the sum is the step at the mean
-- ════════════════════════════════════════════════════════════════

/-- The mean gradient `Gt/k`, which the render never materialises. -/
noncomputable def meanOf (k : ℝ) (Gt : Vec n) : Vec n := fun i => Gt i / k

/-- The first moment as the apply phase computes it: `(1 − β₁)/k` on the SUM. -/
noncomputable def adamMNextAcc (β₁ k : ℝ) (m Gt : Vec n) : Vec n :=
  fun i => β₁ * m i + ((1 - β₁) / k) * Gt i

/-- The second moment as the apply phase computes it: `(1 − β₂)/k²` on the SUM squared. -/
noncomputable def adamVNextAcc (β₂ k : ℝ) (v Gt : Vec n) : Vec n :=
  fun i => β₂ * v i + ((1 - β₂) / k ^ 2) * (Gt i) ^ 2

theorem adamMNextAcc_eq_mean (β₁ k : ℝ) (hk : k ≠ 0) (m Gt : Vec n) :
    adamMNextAcc β₁ k m Gt = adamMNext β₁ m (meanOf k Gt) := by
  funext i; simp only [adamMNextAcc, adamMNext, meanOf]; field_simp

theorem adamVNextAcc_eq_mean (β₂ k : ℝ) (hk : k ≠ 0) (v Gt : Vec n) :
    adamVNextAcc β₂ k v Gt = adamVNext β₂ v (meanOf k Gt) := by
  funext i; simp only [adamVNextAcc, adamVNext, meanOf]; field_simp

/-- The AdamW parameter update as the apply phase computes it (`adamWParamF` reads the same
    `%ob1`/`%ob2`). -/
noncomputable def adamWParamAcc (β₁ β₂ ε lr wd bc₁ bc₂ k : ℝ) (θ m v Gt : Vec n) : Vec n :=
  fun i =>
    let mh := adamMNextAcc β₁ k m Gt i / bc₁
    let vh := adamVNextAcc β₂ k v Gt i / bc₂
    θ i - lr * (mh / (Real.sqrt vh + ε)) - (wd * lr) * θ i

/-- The `(θ', m', v')` triple of an accumulating AdamW render on one micro-batch. -/
noncomputable def adamWStepAcc (β₁ β₂ ε lr wd bc₁ bc₂ k : ℝ) (θ m v Gt : Vec n) :
    Vec n × Vec n × Vec n :=
  (adamWParamAcc β₁ β₂ ε lr wd bc₁ bc₂ k θ m v Gt, adamMNextAcc β₁ k m Gt, adamVNextAcc β₂ k v Gt)

theorem adamWParamAcc_eq_mean (β₁ β₂ ε lr wd bc₁ bc₂ k : ℝ) (hk : k ≠ 0) (θ m v Gt : Vec n) :
    adamWParamAcc β₁ β₂ ε lr wd bc₁ bc₂ k θ m v Gt
      = adamWParam β₁ β₂ ε lr wd bc₁ bc₂ θ m v (meanOf k Gt) := by
  funext i
  simp only [adamWParamAcc, adamWParam, adamMNextAcc_eq_mean β₁ k hk, adamVNextAcc_eq_mean β₂ k hk]

/-- **Apply phase: the accumulating step at the SUM is the plain AdamW step at the MEAN.** -/
theorem adamWStepAcc_eq_mean (β₁ β₂ ε lr wd bc₁ bc₂ k : ℝ) (hk : k ≠ 0) (θ m v Gt : Vec n) :
    adamWStepAcc β₁ β₂ ε lr wd bc₁ bc₂ k θ m v Gt
      = adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v (meanOf k Gt) := by
  simp only [adamWStepAcc, adamWStep, adamWParamAcc_eq_mean β₁ β₂ ε lr wd bc₁ bc₂ k hk,
    adamMNextAcc_eq_mean β₁ k hk, adamVNextAcc_eq_mean β₂ k hk]

/-- **Accumulate phase: with `β = 1` on both moments and `lr = 0`, nothing moves.** -/
theorem adamWStepAcc_accumulate (ε wd bc₁ bc₂ k : ℝ) (θ m v Gt : Vec n) :
    adamWStepAcc 1 1 ε 0 wd bc₁ bc₂ k θ m v Gt = (θ, m, v) := by
  refine Prod.ext ?_ (Prod.ext ?_ ?_) <;> funext i <;>
    simp [adamWStepAcc, adamWParamAcc, adamMNextAcc, adamVNextAcc]

/-- LAMB's direction as the apply phase computes it. -/
noncomputable def lambDirAcc (β₁ β₂ ε wd bc₁ bc₂ k : ℝ) (θ m v Gt : Vec n) : Vec n := fun i =>
  (adamMNextAcc β₁ k m Gt i / bc₁) / (Real.sqrt (adamVNextAcc β₂ k v Gt i / bc₂) + ε) + wd * θ i

/-- The `(θ', m', v')` triple of an accumulating LAMB render on one micro-batch. -/
noncomputable def lambStepAcc (β₁ β₂ ε lr wd bc₁ bc₂ wn2 k : ℝ) (θ m v Gt : Vec n) :
    Vec n × Vec n × Vec n :=
  (sgdParam lr θ (lambScale wn2 (lambDirAcc β₁ β₂ ε wd bc₁ bc₂ k θ m v Gt)),
   adamMNextAcc β₁ k m Gt,
   adamVNextAcc β₂ k v Gt)

theorem lambDirAcc_eq_mean (β₁ β₂ ε wd bc₁ bc₂ k : ℝ) (hk : k ≠ 0) (θ m v Gt : Vec n) :
    lambDirAcc β₁ β₂ ε wd bc₁ bc₂ k θ m v Gt = lambDir β₁ β₂ ε wd bc₁ bc₂ θ m v (meanOf k Gt) := by
  funext i
  simp only [lambDirAcc, lambDir, adamMNextAcc_eq_mean β₁ k hk, adamVNextAcc_eq_mean β₂ k hk]

/-- **Apply phase: the accumulating LAMB step at the SUM is `lambStep` at the MEAN** — RSB-A3's
    actual optimizer, `lambaccdp4x`, read as LAMB on the 2048-image batch. -/
theorem lambStepAcc_eq_mean (β₁ β₂ ε lr wd bc₁ bc₂ wn2 k : ℝ) (hk : k ≠ 0) (θ m v Gt : Vec n) :
    lambStepAcc β₁ β₂ ε lr wd bc₁ bc₂ wn2 k θ m v Gt
      = lambStep β₁ β₂ ε lr wd bc₁ bc₂ wn2 θ m v (meanOf k Gt) := by
  simp only [lambStepAcc, lambStep, lambDirAcc_eq_mean β₁ β₂ ε wd bc₁ bc₂ k hk,
    adamMNextAcc_eq_mean β₁ k hk, adamVNextAcc_eq_mean β₂ k hk]

/-- **Accumulate phase for LAMB: `lr = 0` discards the direction, the moments pass through.**
    The direction computed on an accumulate micro-batch is built from `β₁·m` rather than a real
    moment (`StableHLO.OptRecipe.lambAccum` says so); this is why that is harmless. -/
theorem lambStepAcc_accumulate (ε wd bc₁ bc₂ wn2 k : ℝ) (θ m v Gt : Vec n) :
    lambStepAcc 1 1 ε 0 wd bc₁ bc₂ wn2 k θ m v Gt = (θ, m, v) := by
  refine Prod.ext ?_ (Prod.ext ?_ ?_) <;> funext i <;>
    simp [lambStepAcc, sgdParam, adamMNextAcc, adamVNextAcc]

end Proofs
