import LeanMlir.F32Array

/-! DDPM noise schedule + per-step input plumbing.

A DDPM trainer:

  1. Precomputes the cumulative-α table once via `cosineSchedule`,
     stored as a `[T]` f32 ByteArray.
  2. Per training step, calls `stepInputs` to:
       - sample a timestep `t_b ∈ [0, T)` per image
       - sample Gaussian noise `ε ~ N(0, I)`
       - compute `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
     and returns `(x_t, ε, t)` as three ByteArrays.
  3. Trains the model to predict `ε` from `x_t` (per-pixel MSE).

The loss + backward live in the `useDdpm` codegen branch of
`MlirCodegen.generateTrainStep`.

The schedule is the cosine variant from Nichol & Dhariwal 2021,
which trains more stably than Ho et al.'s original linear schedule.
-/

namespace Ddpm

/-- Build the cosine noise schedule (Nichol & Dhariwal 2021) as a
    `[T]` f32 LE ByteArray of `ᾱ_t` values. With `s = 0.008` the
    schedule keeps log-SNR roughly linear in `t`. -/
@[extern "lean_ddpm_cosine_schedule"]
opaque cosineSchedule (T : USize) : IO ByteArray

/-- Sample `n` f32 from N(0, 1) via Box–Muller. Used for sampling-time
    noise (training noise comes from `stepInputs`). -/
@[extern "lean_ddpm_sample_noise"]
opaque sampleNoise (n : USize) (seed : USize) : IO ByteArray

/-- DDIM (η = 0) deterministic update: `x_{t-1} = a · x_t + b · ε`,
    where `a = √ᾱ_{t-1} / √ᾱ_t` and `b = √(1-ᾱ_{t-1}) - a·√(1-ᾱ_t)`.
    Caller precomputes `a, b` from the schedule. -/
@[extern "lean_ddim_step"]
opaque ddimStep (xt : @& ByteArray) (eps : @& ByteArray)
    (a : Float) (b : Float) (n : USize) : IO ByteArray

/-- Prepend a timestep-encoding channel to each image. Output is
    `[B, C+1, H, W]` (flat) where channels 0..C-1 are the input image
    and channel C is filled with `t[i] / T_max`. Lets the UNet
    condition on the diffusion timestep without a new codegen
    primitive — it just sees a (C+1)-channel input. `t` is a `[B]`
    int32 array (one timestep per image). -/
@[extern "lean_ddpm_prepend_t_channel"]
opaque prependTChannel (xt : @& ByteArray) (t : @& ByteArray)
    (B : USize) (C : USize) (H : USize) (W : USize) (Tmax : USize) : IO ByteArray

/-- Scalar variant of `prependTChannel` for the sampler — broadcasts
    a single timestep to all images in the batch. -/
@[extern "lean_ddpm_prepend_t_channel_scalar"]
opaque prependTChannelScalar (xt : @& ByteArray)
    (B : USize) (C : USize) (H : USize) (W : USize) (t : USize) (Tmax : USize) : IO ByteArray

/-- Sinusoidal time embedding: prepend `2 * nFreq` channels of
    `[sin(t · ω_k), cos(t · ω_k)]` at log-spaced frequencies
    (Vaswani / NeRF convention). Replaces the cruder single-channel
    `t/T_max` tile with multi-frequency information.
    Output: `[B, C + 2·nFreq, H, W]` flat. -/
@[extern "lean_ddpm_prepend_sincos_t"]
opaque prependSinCosT (xt : @& ByteArray) (t : @& ByteArray)
    (B : USize) (C : USize) (H : USize) (W : USize)
    (nFreq : USize) (Tmax : USize) : IO ByteArray

/-- Scalar variant of `prependSinCosT` for the sampler. -/
@[extern "lean_ddpm_prepend_sincos_t_scalar"]
opaque prependSinCosTScalar (xt : @& ByteArray)
    (B : USize) (C : USize) (H : USize) (W : USize)
    (t : USize) (nFreq : USize) (Tmax : USize) : IO ByteArray

/-- Per training step: sample `t_b ∈ [0, T)` per image, sample ε,
    compute `x_t`. Returns `(x_t, ε, t)` where:
      - `x_t` is `[B, npixels]` f32
      - `ε`   is `[B, npixels]` f32 (the loss target — what the model
              should learn to predict)
      - `t`   is `[B]` int32 LE (the per-image timesteps; useful for
              future time-conditioning, currently unused by codegen). -/
@[extern "lean_ddpm_step_inputs"]
opaque stepInputs (x0 : @& ByteArray) (alphaBar : @& ByteArray)
    (B : USize) (npixels : USize) (seed : USize)
    : IO (ByteArray × ByteArray × ByteArray)

/-! ### The continuous-time (VP-SDE) view of the same schedule

    `cosineSchedule` tabulates ᾱ at the integers `t = 0 … T-1`. Score-SDE
    (Song et al. 2021) treats the same schedule as a function of continuous
    `t ∈ [0,1]`, which is what lets an ODE or SDE solver choose its own steps.
    ⭐ **No retraining is needed to get there.** Under the VP SDE the score is
    `∇ log p_t(x) = -ε̂(x,t)/σ_t`, so an ε-predicting network *is* a score model
    up to that factor. Everything here is sampler-side arithmetic, shared by the
    2-D and MNIST drivers so the two cannot drift.
-/

/-- π. Lean core has no `Float.pi`; the C side spells the same digits inline. -/
def piF : Float := 3.14159265358979323846

/-- Nichol & Dhariwal's `s`, the same 0.008 `cosineSchedule` uses. -/
def sBias : Float := 0.008

private def theta (t : Float) : Float :=
  (t + sBias) / (1.0 + sBias) * (piF / 2.0)

/-- ᾱ(t) = cos²θ(t) / cos²θ(0) — the closed form of the tabulated schedule. -/
def abarC (t : Float) : Float :=
  let c := Float.cos (theta t)
  let c0 := Float.cos (theta 0.0)
  (c * c) / (c0 * c0)

/-- σ(t) = √(1-ᾱ(t)), the noise scale of the marginal at time `t`. -/
def sigC (t : Float) : Float := Float.sqrt (max (1.0 - abarC t) 0.0)

/-- β(t) = -d/dt log ᾱ(t) = π·tan θ(t)/(1+s), differentiated in closed form
    rather than differenced — the schedule is analytic, so approximating its
    derivative would be inventing error. ⚠ β diverges as t → 1. That stiffness
    is why an explicit solver in `x`-space struggles here and why DDIM, which
    integrates the linear part exactly, does not. -/
def betaC (t : Float) : Float :=
  piF / (1.0 + sBias) * Float.tan (theta t)

/-- Inverse of `abarC`: `t = (2(1+s)/π)·arccos(cos θ₀·√ᾱ) - s`. Lets a solver
    lay its grid out in ᾱ (or σ, or log-SNR) and land on exact times. -/
def tOfAbar (ab : Float) : Float :=
  let c0 := Float.cos (theta 0.0)
  let arg := min 1.0 (max (-1.0) (c0 * Float.sqrt (max ab 0.0)))
  Float.acos arg * 2.0 * (1.0 + sBias) / piF - sBias

/-- The samplers the drivers can run, and what one step of each costs in network
    evaluations. ⚠ Comparisons are made at matched **NFE**, not matched steps:
    Heun is second-order and pays two evaluations per step, so on a fixed budget
    it takes half as many. Reporting steps instead would flatter it. -/
def samplerNfe : List (String × Nat) :=
  [("ddim", 1), ("euler", 1), ("heun", 2), ("sde", 1)]

end Ddpm
