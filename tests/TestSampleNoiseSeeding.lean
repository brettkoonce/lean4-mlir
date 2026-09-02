import LeanMlir.Ddpm
import LeanMlir.F32Array

/-!
# Seeding regression test for `Ddpm.sampleNoise`

`lean_ddpm_sample_noise` seeded its xorshift64 stream by XOR alone
(`s = seed ^ K`) and then read the first uniform from the **top** 53 bits
(`s >> 11`). xorshift64 is linear over GF(2), so two seeds differing only in
their low bits still differ only in low bits after one round: the top of the
word never moves. Every call therefore drew the same `u1`, hence the same
Box–Muller radius `√(-2 ln u1)`.

Over the 2048 seeds the 2-D diffusion demo uses for `x_T` this produced **two
distinct radii**, both 1.9130 — so every sample started life on a *circle*
rather than being drawn from `N(0, I)`, and the reverse process was fed a
distribution it was never trained on.

Nothing downstream could see it. The image DDPMs draw one long vector per call
(`sampleNoise (B * nPix) 0xc0ffee`), where only the *first* pair of each call is
correlated across seeds and everything after it comes from a well-mixed stream,
so their grids looked normal. Per-axis mean and variance are both correct under
the defect — the mass is merely on a shell instead of filling the ball — so any
summary statistic on the coordinates agrees with a healthy Gaussian. The radius
is what separates them, which is what this file asserts.

▶ Found by the reverse-process strip of `planning/diffusion_2d_demo.md` §5: the
`t = T` panel is meant to be an isotropic blob and it was a ring. That is the
figure earning its place — no number in the demo's metric suite moved.

Hermetic: no data files, no GPU.
-/

namespace SampleNoiseSeeding

/-- `|x|` for `n` two-dimensional draws taken at seeds `f 0 … f (n-1)`. -/
private def radii (n : Nat) (f : Nat → Nat) : IO (Array Float) := do
  let mut out : Array Float := #[]
  for i in [0:n] do
    let z ← Ddpm.sampleNoise 2 (f i).toUSize
    let x := F32.read z 0
    let y := F32.read z 1
    out := out.push (Float.sqrt (x * x + y * y))
  return out

private def mean (a : Array Float) : Float :=
  a.foldl (· + ·) 0.0 / a.size.toFloat

private def stdev (a : Array Float) : Float :=
  let m := mean a
  Float.sqrt ((a.foldl (fun s x => s + (x - m) * (x - m)) 0.0) / a.size.toFloat)

/-- For `x ~ N(0, I₂)`, `|x|` is Rayleigh(1): mean `√(π/2) ≈ 1.2533`,
    standard deviation `√(2 - π/2) ≈ 0.6551`, and `E|x|² = 2`. The bounds are
    wide because the point is not to test the quality of the Box–Muller
    transform — it is to separate a Rayleigh from a constant. The defect scored
    a standard deviation of about `1e-6` and `E|x|² = 3.66`. -/
private def check (label : String) (n : Nat) (f : Nat → Nat) : IO Bool := do
  let r ← radii n f
  let m  := mean r
  let sd := stdev r
  let m2 := mean (r.map (fun x => x * x))
  let ok := sd > 0.40 && sd < 0.90 && m2 > 1.70 && m2 < 2.30 && m > 1.05 && m < 1.45
  IO.println s!"  {label}: mean|x|={m} sd|x|={sd} E|x|^2={m2}  \
    [{if ok then "PASS" else "FAIL"}]"
  return ok

def main : IO UInt32 := do
  IO.println "Ddpm.sampleNoise seeding — radii must be Rayleigh, not constant"
  -- The two seed patterns `demos/archive/MainDiffusion2d.lean` actually uses: nearly
  -- consecutive seeds for `x_T`, and a strided pattern for the eta > 0 noise.
  -- The second was biased rather than degenerate (its differences reach higher
  -- bits), which is why it needs its own row instead of being assumed covered.
  let a ← check "consecutive  (i + 7919)      " 1024 (fun i => i + 7919)
  let b ← check "strided      (i·131071 + 17) " 1024 (fun i => i * 131071 + 17)
  let c ← check "strided      (i·8191 + 17)   " 1024 (fun i => i * 8191 + 17)
  if a && b && c then
    IO.println "PASS"
    return 0
  else
    IO.println "FAIL — sampleNoise is not drawing from N(0, I); see the header"
    return 1

end SampleNoiseSeeding

def main : IO UInt32 := SampleNoiseSeeding.main
