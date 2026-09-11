import LeanMlir.Spec

/-! # Fourier neural operator (FNO) — Bestiary entry

Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart & Anandkumar,
"Fourier neural operator for parametric partial differential equations",
ICLR 2021 (arXiv:2010.08895).

**An operator, not a function.** A PINN learns one solution u(x, t) of one
PDE. An FNO learns the solution OPERATOR — initial condition in, solution
out — for a whole family, from data, and then evaluates in one forward
pass at any resolution, because nothing in it is tied to the grid it was
trained on. Four blocks, each of which is

    v ↦ σ( W v  +  F⁻¹( R · F v ) )

a pointwise linear map W (a 1×1 conv) in parallel with a **spectral
convolution**: FFT the field, keep the lowest k modes, multiply each mode by
its own learned complex matrix R ∈ ℂ^{width×width}, inverse FFT. Lifting P
(1×1 conv, a(x) → width) in front, projection Q (1×1 conv, width → 128 → 1)
behind. The spectral convolution is where the parameters and the idea live,
and it is the one primitive our `NetSpec` language has no constructor for:
it is an FFT, a truncation and a per-mode matmul, and none of Part 1's
layers is any of those. The specs below are the **non-spectral skeleton** —
P, the four W bypasses, Q — with the spectral weights counted in prose,
the way the DDPM entry counts what its UNet backbone omits.

## Variants

- `fno2dNavierStokes` — the 2-D Navier–Stokes config: 64×64 grid, width 20,
                        12×12 modes. Skeleton 4,577; the four R tensors add
                        4 × 2 × 20·20·12·12 complex = 921,600 real weights,
                        so the paper's ~926k total is 99.5 % spectral.
- `fno1dBurgers`      — the 1-D Burgers config: 1024 points, width 64, 16 modes.
                        Skeleton 25,281 + 4 × 64·64·16 complex = 524,288 reals.
- `tinyFno`           — 16×16 grid, width 8.

The point the entry makes: the parameter count is almost entirely in a
layer the linear spec cannot spell, and the resolution-invariance the paper
is known for comes from that layer alone — P, W and Q are pointwise and
would be resolution-invariant in any architecture.
-/

-- ════════════════════════════════════════════════════════════════
-- § FNO-2d, Navier–Stokes (the paper's headline config)
-- ════════════════════════════════════════════════════════════════
-- Input channels: the field a(x, y) plus the two grid coordinates = 3.
-- Every layer here is a 1×1 conv, i.e. pointwise; the spectral branch of
-- each Fourier layer runs beside the W conv and is not a layer here.

def fno2dNavierStokes : NetSpec where
  name   := "FNO-2d (Navier-Stokes, width 20, 12 modes; non-spectral skeleton)"
  imageH := 64
  imageW := 64
  layers := [
    .conv2d 3 20 1 .same .identity,     -- P: lift (a, x, y) → width
    .conv2d 20 20 1 .same .gelu,        -- Fourier layer 1: W bypass (+ R, 230,400 reals)
    .conv2d 20 20 1 .same .gelu,        -- Fourier layer 2
    .conv2d 20 20 1 .same .gelu,        -- Fourier layer 3
    .conv2d 20 20 1 .same .identity,    -- Fourier layer 4 (no activation after the last)
    .conv2d 20 128 1 .same .gelu,       -- Q: project
    .conv2d 128 1 1 .same .identity     --    → u(x, y)
  ]

-- ════════════════════════════════════════════════════════════════
-- § FNO-1d, Burgers
-- ════════════════════════════════════════════════════════════════

def fno1dBurgers : NetSpec where
  name   := "FNO-1d (Burgers, width 64, 16 modes; non-spectral skeleton)"
  imageH := 1024
  imageW := 1
  layers := [
    .conv2d 2 64 1 .same .identity,     -- P: (a, x) → width
    .conv2d 64 64 1 .same .gelu,        -- 4 Fourier layers' W bypasses (+ R, 131,072 reals each)
    .conv2d 64 64 1 .same .gelu,
    .conv2d 64 64 1 .same .gelu,
    .conv2d 64 64 1 .same .identity,
    .conv2d 64 128 1 .same .gelu,       -- Q
    .conv2d 128 1 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Tiny fixture
-- ════════════════════════════════════════════════════════════════

def tinyFno : NetSpec where
  name   := "tiny-FNO (16x16, width 8; non-spectral skeleton)"
  imageH := 16
  imageW := 16
  layers := [
    .conv2d 3 8 1 .same .identity,
    .conv2d 8 8 1 .same .gelu,
    .conv2d 8 8 1 .same .identity,
    .conv2d 8 16 1 .same .gelu,
    .conv2d 16 1 1 .same .identity
  ]

-- ════════════════════════════════════════════════════════════════
-- § Main: print-only summary
-- ════════════════════════════════════════════════════════════════

def main : IO Unit := do
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Bestiary — Fourier neural operator"
  IO.println "════════════════════════════════════════════════════════════════"
  IO.println "  Learns the solution operator of a PDE family; evaluates at any"
  IO.println "  resolution. The parameters live in a layer NetSpec cannot spell."

  fno2dNavierStokes.summarize (size := .omitted) (unit := .thousands)
  fno1dBurgers.summarize (size := .omitted) (unit := .thousands)
  tinyFno.summarize (size := .omitted) (unit := .thousands)

  IO.println ""
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  Notes"
  IO.println "────────────────────────────────────────────────────────────────"
  IO.println "  • ONE new primitive, NOT shown: the spectral convolution"
  IO.println "    F⁻¹(R · F v) with R ∈ ℂ^{modes × width × width} per layer."
  IO.println "  • The counts above are the pointwise skeleton (P, W, Q)."
  IO.println "    FNO-2d adds 921,600 spectral reals → ~926k total, 99.5 %"
  IO.println "    of the model; FNO-1d adds 524,288."
  IO.println "  • Resolution invariance is the spectral layer's property:"
  IO.println "    truncating to k modes is the same operation on any grid."
