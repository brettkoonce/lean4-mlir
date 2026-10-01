/-
PR 3 — feat(Probability/Distributions/Gaussian/Multivariate): the standard Gaussian has full support
(depends on PR 2: `instIsOpenPosMeasureGaussianReal`)

Content below is to be APPENDED to `Mathlib/Probability/Distributions/Gaussian/Multivariate.lean`,
inside its `namespace ProbabilityTheory`. No new imports are needed
(`Mathlib.MeasureTheory.Constructions.Pi` is transitive). Add `Brett Koonce` to the file's
`Authors:` line.

Verified to compile against the pinned Mathlib by `LeanMlir/Proofs/Foundation/UpstreamDraft.lean`
(namespace `MathlibUpstream`); keep the two in sync.
-/

section StdGaussian

/-- The standard Gaussian on a finite-dimensional inner-product space gives positive mass to
every nonempty open set: it is the pushforward of the product of standard real Gaussians (open-pos
by `instIsOpenPosMeasureGaussianReal` and `Measure.pi.isOpenPosMeasure`) under the surjective
continuous basis sum. -/
instance instIsOpenPosMeasureStdGaussian {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] [MeasurableSpace E] [BorelSpace E] :
    (stdGaussian E).IsOpenPosMeasure := by
  refine Continuous.isOpenPosMeasure_map (by fun_prop) fun e => ?_
  exact ⟨fun i => (stdOrthonormalBasis ℝ E).repr e i,
    by simpa using (stdOrthonormalBasis ℝ E).sum_repr e⟩

end StdGaussian

