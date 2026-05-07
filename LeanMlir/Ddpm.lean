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

end Ddpm
