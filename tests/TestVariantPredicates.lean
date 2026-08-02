import LeanMlir.VerifiedNets

/-! # The variant-string predicates, run rather than reasoned about

`LEAN_MLIR_VARIANT` now encodes **three independent axes**, and `trainAdamSched` recovers each with
a string test on the name:

| axis | predicate | what it decides |
|---|---|---|
| EMA | `variant.startsWith "ema"` | a FOURTH `[θ\|m\|v\|ema]` blob region, 5 scalars not 3 |
| RMSProp | `"rms"` substring | the mean-square slot initialises to **1.0**, not 0 |
| stochastic depth | `"drop"` substring | N extra `tensor<Bxf32>` scale inputs |

Every one of those is a SILENT wrong answer if it misfires: a 3-region blob fed to a 4-region graph
misaligns every parameter, a zero-initialised mean-square is a different optimizer, and a spurious
drop-scale block is an arity error at best.

**This file exists because the naming has now broken TWICE, and the second time was not visible by
reading names one at a time.**

1. `planning/ema.md` — the RMSProp test was `startsWith "rms"`, and the RMSProp+EMA variant is
   **`emarms`**, which does not start with "rms". A prefix test on a two-axis name fails quietly;
   the mean-square would have initialised to 0, which is the exact defect that thread existed to fix.
2. `planning/stochastic_depth.md` — the drop marker was `"sd"`, and **`rms` ++ `dp` spells `rmsdp`,
   which CONTAINS "sd"**. So the test fired on `rmsdp64` and `emarmsdp64` — every RMSProp
   data-parallel variant, including the committed and gated `enetin_rmsdp64`. ⚠ The collision was
   between two OTHER markers meeting, not between the new marker and an old one, which is precisely
   what reading a name at a time cannot show you. Renamed to `"drop"`.

So the rule, and it is what this file enforces: **with N markers the collisions are between PAIRS
of them, so a new marker has to be checked against every CONCATENATION, not every marker.**

    lake env lean tests/TestVariantPredicates.lean
-/

private def emaOn (v : String) : Bool := v.startsWith "ema"
private def rmsOn (v : String) : Bool := (v.splitOn "rms").length > 1
private def sdOn  (v : String) : Bool := (v.splitOn "drop").length > 1

/-- Every variant string any renderer can produce today, with what each axis MUST read.
    Grown from the `*AdamVariant` functions, not from the artifacts — an artifact that does not
    exist yet is exactly the one whose name will collide. -/
private def table : List (String × Bool × Bool × Bool) :=
  [ -- AdamW
    ("adam", false, false, false), ("adamdp", false, false, false)
  , ("adam64", false, false, false), ("adam128", false, false, false)
  , ("adamdp64", false, false, false), ("adamdp32x4", false, false, false)
  , ("adamdp128x4", false, false, false)
    -- heavy-ball / Nesterov / plain SGD
  , ("mom", false, false, false), ("mom256", false, false, false)
  , ("momdp64", false, false, false), ("sgd", false, false, false)
    -- RMSProp. ⚠ `rmsdp*` is the pair that broke the `"sd"` marker.
  , ("rms", false, true, false), ("rms64", false, true, false)
  , ("rmsdp64", false, true, false)
    -- EMA
  , ("ema", true, false, false), ("emadp", true, false, false)
  , ("ema128", true, false, false), ("emadp128x4", true, false, false)
    -- EMA × RMSProp — the reference's actual EfficientNet recipe
  , ("emarms", true, true, false), ("emarms64", true, true, false)
  , ("emarmsdp64", true, true, false)
    -- stochastic depth, and all three axes at once
  , ("adamdrop", false, false, true)
  , ("emarmsdrop64", true, true, true)
    -- ▶ v1.4 `wx` = timm no_weight_decay (`vitWdDecays`). ⚠ It needs NO driver predicate of its
    -- own — excluding a param binds a different CONSTANT to `%wd` and changes no arity, no type
    -- and no region — so it appears here for the OPPOSITE reason to the other three: to prove the
    -- new marker does not disturb any of them. That is the check the `sd`/`rmsdp` collision
    -- showed reading names one at a time cannot do, since a collision lives in a CONCATENATION.
  , ("adamwx", false, false, false), ("adam128wx", false, false, false)
  , ("adamdp128x4wx", false, false, false), ("emawx", true, false, false)
    -- and `wx` composed with each of the three axes, since that is where a collision would be
  , ("emarmswx", true, true, false), ("adamdropwx", false, false, true)
  , ("emarmsdrop64wx", true, true, true) ]

#guard table.all (fun (v, e, r, s) => emaOn v == e && rmsOn v == r && sdOn v == s)

-- ⚠ The regression, pinned directly: the OLD `"sd"` marker fires on `rmsdp64`, the new one does not.
#guard (("rmsdp64".splitOn "sd").length > 1) == true
#guard sdOn "rmsdp64" == false

-- ⚠ And why the drop marker must TRAIL: a leading one breaks the EMA test, which is defect #1's
-- shape. `dropema` is not a name anything produces today — that is the point of checking it here.
#guard emaOn "dropema" == false
#guard emaOn "emadrop" == true

-- ⚠ The `wx` marker against every CONCATENATION, not against the other markers one at a time.
-- The failure that produced this file was `rms` ++ `dp` spelling `rmsdp` ⊇ "sd"; the analogous
-- question for `wx` is whether any pair of markers can spell it. None can — no marker ends in `w`
-- and none begins with `x` — and these are the checks that say so rather than the reasoning.
#guard (("adamdp128x4wx".splitOn "rms").length == 1)
#guard (("adamdp128x4wx".splitOn "drop").length == 1)
#guard sdOn "adamdp128x4wx" == false
#guard rmsOn "adamdp128x4wx" == false
-- and `wx` must not create an "ema" prefix where there was none
#guard emaOn "adamwx" == false

#eval do
  IO.println "── variant predicates ──"
  for (v, e, r, s) in table do
    let regions := if e then 4 else 3
    IO.println s!"  {v} — ema={e} rms={r} drop={s} → {regions} blob regions"
  IO.println s!"✓ {table.length} variant spellings, 3 axes, no collision"
