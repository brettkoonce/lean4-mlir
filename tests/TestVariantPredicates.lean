import LeanMlir.VerifiedNets

/-! # The variant-string predicates, run rather than reasoned about

`LEAN_MLIR_VARIANT` now encodes **four independent axes**, and `trainAdamSched` recovers each with
a string test on the name:

| axis | predicate | what it decides |
|---|---|---|
| EMA | `variant.startsWith "ema"` | a FOURTH `[θ\|m\|v\|ema]` blob region, 5 scalars not 3 |
| RMSProp | `"rms"` substring | the mean-square slot initialises to **1.0**, not 0 |
| stochastic depth | `"drop"` substring | N extra `tensor<Bxf32>` scale inputs |
| classifier dropout | `"do"` substring | ONE extra `tensor<Bx1280xf32>` mask input |

Every one of those is a SILENT wrong answer if it misfires: a 3-region blob fed to a 4-region graph
misaligns every parameter, a zero-initialised mean-square is a different optimizer, and a spurious
drop-scale block is an arity error at best.

**This file exists because the naming has now broken TWICE, and the second time was not visible by
reading names one at a time. The THIRD collision was caught before it shipped, by this file.**

1. `planning/ema.md` — the RMSProp test was `startsWith "rms"`, and the RMSProp+EMA variant is
   **`emarms`**, which does not start with "rms". A prefix test on a two-axis name fails quietly;
   the mean-square would have initialised to 0, which is the exact defect that thread existed to fix.
2. `planning/stochastic_depth.md` — the drop marker was `"sd"`, and **`rms` ++ `dp` spells `rmsdp`,
   which CONTAINS "sd"**. So the test fired on `rmsdp64` and `emarmsdp64` — every RMSProp
   data-parallel variant, including the committed and gated `efficientnetin_rmsdp64`. ⚠ The collision was
   between two OTHER markers meeting, not between the new marker and an old one, which is precisely
   what reading a name at a time cannot show you. Renamed to `"drop"`.

3. `recipe_gaps.md` gap C — classifier dropout's obvious marker is `"dropout"`, which **contains
   `"drop"`**, so every dropout render would have read as a stochastic-depth one and the driver
   would have packed nine mask slots the graph does not have. ⚠ Unlike the first two this was
   caught *before* it shipped, and the difference is only that this file existed to be extended.
   Marker `"do"`; the counterfactual is pinned below (`sdOn "adamdropout" == true`).

So the rule, and it is what this file enforces: **with N markers the collisions are between PAIRS
of them, so a new marker has to be checked against every CONCATENATION, not every marker.**

⚠ And a second rule this file learned the hard way in its own table (2026-08-03): **grow the table
from the `*AdamVariant` FUNCTIONS, not from memory.** It carried `emarmsdrop64` for a session — a
spelling no renderer emits, while the real `emarms64drop` was absent — and the artifact named after
the wrong spelling turned out to be unloadable. A hand-written table of derived names drifts.

    lake env lean tests/TestVariantPredicates.lean
-/

/-! ⚠⚠ **THIS FILE USED TO PIN COPIES, WHICH IS WHY IT NOW OPENS `VerifiedVariant` (2026-08-14).**

Every predicate below was declared here as a `private def` transcribing what `trainAdamSched`
computed inline. So the table gated *a* definition of each axis and not *the* definition: an edit
to the driver's own `variant.startsWith "ema"` could not turn this file red, and the two could
drift exactly the way `emarmsdrop64` drifted from `emarms64drop` — the drift this file's own
closing rule warns about, one level up from names to logic.

▶ The five predicates and `accK` now live in `LeanMlir/VerifiedTrain.lean`'s `VerifiedVariant`
namespace, with their history in their docstrings; `trainAdamSched` and
`VerifiedNet.scoreCheckpoint` both consume them, and so does this table. That is
`next_session_verified_trainer_code.md` §5's lesson applied here: *a gate on "the feature is
enabled" is not a gate on "the feature is correct"* — and a gate on a transcription is not a gate
on the thing transcribed.
-/
open VerifiedVariant

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
    -- ⚠ `adamdpdrop` — the DP drop render (`stochastic_depth.md` §5b's gate vehicle). It is the
    -- `dp` and `drop` markers MEETING, which is the shape that broke `sd`: `rms` ++ `dp` spelled
    -- `rmsdp` ⊇ "sd". `dp` ++ `drop` spells `dpdrop`, which contains neither "sd" nor "rms" nor
    -- an "ema" prefix — but that is the reasoning the earlier collision falsified, so it is run.
  , ("adamdpdrop", false, false, true)
  , ("rmsdpdrop64", false, true, true)
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
  , ("emarmsdrop64wx", true, true, true)
    -- ▶ v1.4b `clip` = global-norm gradient clipping (`planning/grad_clip.md`). Like `wx` it needs
    -- NO driver predicate — the clip changes no arity, no type and no region — so it is here to
    -- prove it disturbs none of the three, in every CONCATENATION it can appear in. It TRAILS `wx`
    -- because the ViT/ConvNeXt reference sets both, so `wxclip` is the shipping spelling.
  , ("adamclip", false, false, false), ("adamwxclip", false, false, false)
  , ("adam128wxclip", false, false, false), ("adamdp128x4wxclip", false, false, false)
  , ("adamdpwxclip", false, false, false), ("emaclip", true, false, false)
    -- and composed with each of the three axes
  , ("emarmsclip", true, true, false), ("adamdropclip", false, false, true)
  , ("emarmsdrop64wxclip", true, true, true)
    -- ▶ ConvNeXt's SD spellings (handoff §0.10), and they are here because the ORDER differs from
    -- EfficientNet's. `cnxAdamVariant` appends `drop` LAST — after `wx` and `clip` — where
    -- `enetAdamVariant` appends it after the batch suffix, so the same three markers concatenate
    -- into different strings on the two nets. Reading either net's order and assuming the other
    -- matches is exactly the one-name-at-a-time reasoning the `sd`/`rmsdp` collision falsified.
    -- ⚠ `adamdpwxclipdrop` is the shipping ImageNet spelling — `convNeXtTinyImagenetConfig` sets
    -- decay 0.05 + `wdExcludeNormBias` + `gradClipNorm` + `dropPath`, so it is all four at once.
  , ("adamwxclipdrop", false, false, true), ("adamdpwxclipdrop", false, false, true)
  , ("adamdpdrop", false, false, true), ("emawxclipdrop", true, false, true)
    -- ▶ ViT's SD spellings. ViT's variant carries a BATCH suffix the other two do not
    -- (`adam64`, `adam128`), so `drop` concatenates against a DIGIT here — a third ordering of the
    -- same three markers, and the reason this table is run rather than reasoned about.
  , ("adam128wxclipdrop", false, false, true), ("adamdp128x4drop", false, false, true)
  , ("adam64drop", false, false, true)
    -- ▶ CLASSIFIER DROPOUT's spellings (`recipe_gaps.md` gap C). ⚠ `emarms64drop*` is EfficientNet's
    -- REAL ordering — the batch suffix precedes the regulariser markers — which this table had
    -- wrong as `emarmsdrop64` until 2026-08-03, when a `#guard` against `enetAdamVariant` caught
    -- that no renderer emits it. Both are kept: the real one because it ships, the other because a
    -- collision table should cover strings whether or not anything emits them. That the two
    -- disagreed for a session is the sharpest possible argument for this file's own rule — a table
    -- of names written by hand drifts from the function that derives them.
  , ("adamdo", false, false, false), ("emarms64dropdo", true, true, true)
  , ("emarms64drop", true, true, true), ("rmsdo64", false, true, false)
    -- ▶▶ GRADIENT ACCUMULATION's spellings (`r34AdamVariant .adamwAccum`). ⚠ Both carry `k` and a
    -- batch, so the marker concatenates against DIGITS and an `x` — a shape none of the other four
    -- markers has, and `accdp` puts `dp` INSIDE the prefix rather than after it.
  , ("acc4x64", false, false, false), ("accdp8x64", false, false, false)
  , ("acc2x128", false, false, false)
    -- ▶▶ RSB-A3's COMPOSED optimizer (`r34AdamVariant .lambAccum`, 2026-08-06) — LAMB × accumulation
    -- × BCE, i.e. the marker `acc` with ANOTHER OPTIMIZER NAME IN FRONT OF IT. ⚠⚠ This is the row
    -- that falsified `accOn`'s prefix test (defect #4): `lamb` ++ `acc` is the concatenation the
    -- old docstring's "the marker leads" reasoning had ruled out by assumption. It also trails
    -- `bce`, so `acc` here is bracketed on BOTH sides — a placement no other marker has had.
  , ("lambacc8x64bce", false, false, false), ("lambaccdp8x64bce", false, false, false)
  , ("lambacc4x64bce", false, false, false), ("lambaccdp4x64", false, false, false)
    -- ▶ LAMB (`r34AdamVariant .lamb`). ⚠ It needs NO driver predicate — three regions, the same
    -- `[θ|m|v]` signature as `adam`, because the trust ratio is computed inside the graph from θ
    -- and the direction and needs no extra state. So it is here for `wx`/`clip`'s reason: to prove
    -- the new marker disturbs NONE of the five. `lamb` ends in `b` and `lambdp` puts `dp` after it.
  , ("lamb64", false, false, false), ("lambdp64", false, false, false) ]

#guard table.all (fun (v, e, r, s) => emaOn v == e && rmsOn v == r && sdOn v == s)

-- ⭐⭐ THE `do` MARKER AGAINST EVERY CONCATENATION IN THE TABLE, which is the check this file
-- exists for. `"do"` is two characters, so the question is not "does any marker contain it" but
-- "can any PAIR of markers, or a marker meeting a digit, spell it" — the `rms` ++ `dp` ⊇ "sd"
-- shape. Answer, RUN rather than reasoned: no existing marker ends in `d` and none begins with
-- `o`, and no marker contains `do` internally (`drop` is d-r-o-p). So `cdOn` fires on exactly the
-- variants that set it.
#guard table.all (fun (v, _, _, _) => cdOn v == ((v.splitOn "do").length > 1))
#guard cdOn "adamdo" == true
#guard cdOn "emarms64dropdo" == true
-- …and on NONE of the others. ⚠ This is the load-bearing half — every remaining row is a committed,
-- gated artifact whose driver path must not grow a mask input — and it is stated as a PARTITION
-- rather than as a count or a disjunction of excuses. §0.4 finding 5: gate the partition, not the
-- count. (The first draft of this check was a patched-up `|| v == … || s` chain, and it failed on
-- `rmsdo64` — a dropout spelling the chain had simply not been updated for. A partition cannot
-- rot that way: adding a spelling to the table without adding it here fails immediately.)
private def dropoutSpellings : List String := ["adamdo", "emarms64dropdo", "rmsdo64"]
#guard table.all (fun (v, _, _, _) => cdOn v == dropoutSpellings.contains v)
#guard dropoutSpellings.all (fun v => table.any (fun (t, _, _, _) => t == v))
#guard cdOn "emarms64drop" == false      -- ⚠ `drop` alone must NOT read as dropout
#guard cdOn "adamdpdrop" == false        -- ⚠ nor `dp` ++ `drop`
#guard cdOn "adamdp128x4wxclipdrop" == false
#guard cdOn "rmsdp64" == false
-- ⚠ And the reverse direction — the one that forced the spelling. A `"dropout"` marker would make
-- every dropout render read as a stochastic-depth one; this is that counterfactual, pinned.
#guard sdOn "adamdropout" == true        -- what `"dropout"` WOULD have done: a false SD positive
#guard sdOn "adamdo" == false            -- what `"do"` actually does
-- ⚠ and `do` must not disturb the other three axes in the combined spelling
#guard emaOn "emarms64dropdo" == true
#guard rmsOn "emarms64dropdo" == true
#guard sdOn  "emarms64dropdo" == true

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

-- ⚠ The `clip` marker against every CONCATENATION. Same question as `wx`: can any PAIR of existing
-- markers spell it? No marker ends in `c`, `cl` or `cli`, and none begins with `lip`, `ip` or `p` —
-- but "no marker begins with p" is exactly the kind of reasoning `rms` ++ `dp` ⊇ "sd" falsified, so
-- these run it instead. Note `clip` contains no substring of "ema"/"rms"/"drop" either way round.
#guard (("emarmsdrop64wxclip".splitOn "rms").length > 1)     -- still finds the REAL rms
#guard (("emarmsdrop64wxclip".splitOn "drop").length > 1)    -- still finds the REAL drop
#guard emaOn "emarmsdrop64wxclip" == true                    -- still finds the REAL ema
#guard rmsOn "adamdp128x4wxclip" == false                    -- and invents none of them
#guard sdOn  "adamdp128x4wxclip" == false
#guard emaOn "adamdp128x4wxclip" == false
-- ⚠ and the one that would bite if `clip` ever LED rather than trailed, the `dropema` shape:
-- ⚠ the DP drop spelling: `dp` ++ `drop` must not resurrect the `sd` collision, and must not
-- invent an rms/ema axis that is not there.
#guard sdOn  "adamdpdrop" == true
#guard rmsOn "adamdpdrop" == false
#guard emaOn "adamdpdrop" == false
#guard (("adamdpdrop".splitOn "sd").length == 1)      -- the OLD marker does not fire either
#guard emaOn "clipema" == false
#guard emaOn "emaclip" == true

-- ⭐⭐ THE `acc` MARKER AGAINST EVERY CONCATENATION IN THE TABLE. Stated as a PARTITION, the form
-- §0.4 finding 5 requires and the form that caught `rmsdo64` in the `do` block above.
private def accumSpellings : List String :=
  ["acc4x64", "accdp8x64", "acc2x128",
   -- the composed spellings; they MUST be in this partition or the driver reads 3 regions for a
   -- 4-region graph — which is exactly what the prefix test did.
   "lambacc8x64bce", "lambaccdp8x64bce", "lambacc4x64bce", "lambaccdp4x64"]
#guard table.all (fun (v, _, _, _) => accOn v == accumSpellings.contains v)
#guard accumSpellings.all (fun v => table.any (fun (t, _, _, _) => t == v))
-- ⚠ and accumulation must disturb NONE of the other four axes. `acc4x64` contains no "ema" prefix,
-- no "rms", no "drop"; `accdp8x64` puts `dp` inside the prefix, which is a placement no other
-- marker uses, so it is run rather than reasoned about.
#guard table.all (fun (v, _, _, _) => !accOn v || (!emaOn v && !rmsOn v && !sdOn v && !cdOn v))
-- ⚠ the reverse direction, which is the load-bearing half: every committed 3-region variant must
-- NOT read as accumulation, or the driver packs a fourth region into a graph that has three.
#guard accOn "adamdp64" == false
#guard accOn "emarmsdp64" == false
#guard accOn "momdp64" == false
#guard accOn "adamdpwxclipdrop" == false
-- ⭐⭐ THE LIMIT IS GONE, AND ITS DISAPPEARANCE IS THE POINT. This block used to pin
-- `accOn "emaacc4x64" == false` and call it "the correct answer, not a latent bug", on the grounds
-- that the driver refuses `ema` × `acc` anyway. ⚠ That justification was CIRCULAR: the driver's
-- refusal is `if emaOn && accOn then throw`, so with `accOn` false the refusal could never fire and
-- the combination would have run as EMA with accumulation SILENTLY DROPPED — a 4-region graph fed
-- 4 regions, no arity error, and no gradient accumulation. The substring test makes both true, so
-- the refusal fires for real.
#guard accOn "emaacc4x64" == true
#guard emaOn "emaacc4x64" == true
-- ▶▶ DEFECT #4's DIRECT COUNTERFACTUAL: what the OLD prefix test returned for the composed name,
-- pinned so the regression cannot come back silently.
#guard ("lambaccdp8x64bce".startsWith "acc") == false   -- the old test: MISSES it
#guard accOn "lambaccdp8x64bce" == true                 -- the new one: catches it
#guard accOn "lambacc8x64bce" == true
-- ⚠ and the composed names must disturb none of the other four axes — `acc` is bracketed by `lamb`
-- before and `bce` after, a placement no other marker has had.
#guard emaOn "lambaccdp8x64bce" == false
#guard rmsOn "lambaccdp8x64bce" == false
#guard sdOn  "lambaccdp8x64bce" == false
#guard cdOn  "lambaccdp8x64bce" == false    -- ⚠ `dp` is not `do`; the two differ in one character
-- ⚠⚠ `k` ROUND-TRIPS, both spellings, including the two-digit case where `accdp` must be stripped
-- as 5 characters and `acc` as 3. Getting this off by one gives k = 8 read as 0 or as 88.
#guard accK "acc4x64" == 4
#guard accK "accdp8x64" == 8
#guard accK "acc2x128" == 2
#guard accK "acc16x32" == 16
#guard accK "accdp16x64" == 16
#guard accK "adamdp64" == 1        -- non-accumulation variants read k = 1, i.e. no accumulation
-- ⚠⚠ `k` OUT OF THE COMPOSED NAMES, which is where the old fixed-offset parse broke: `v.drop 3`
-- assumed the string starts with `acc`, so on `lambaccdp8x64bce` it dropped "lam" and read "bacc…".
#guard accK "lambaccdp8x64bce" == 8
#guard accK "lambacc4x64bce" == 4
#guard accK "lambaccdp4x64" == 4
-- and the trailing `bce` must not leak into `k` any more than the batch does
#guard accK "lambacc8x64bce" == 8
#guard accK "lambacc8x64bce" != 64
-- ⚠ and the digits AFTER the `x` are the batch, never `k` — `acc2x128` is k = 2 at batch 128, not
-- k = 2128 and not k = 128. The `takeWhile` is what makes that true; this is the check that says so.
#guard accK "acc2x128" != 128
#guard accK "acc2x128" != 2128

-- ⚠ LAMB against every CONCATENATION. Same question as `wx` and `clip`: can any pair of markers
-- spell it? No marker ends in `l`, `la` or `lam`, and none begins with `amb`, `mb` or `b` — but
-- "no marker begins with b" is exactly the reasoning `rms` ++ `dp` ⊇ "sd" falsified, so it is run.
#guard emaOn "lamb64" == false
#guard rmsOn "lamb64" == false
#guard sdOn  "lamb64" == false
#guard cdOn  "lamb64" == false
#guard accOn "lamb64" == false
#guard accOn "lambdp64" == false
-- and LAMB must not invent an axis in the DP spelling either, where `dp` trails the marker
#guard emaOn "lambdp64" == false
#guard cdOn  "lambdp64" == false

-- ⭐ THE REGION ARITHMETIC ITSELF, and it is what `scoreCheckpoint` reads a checkpoint's size
-- against. `nRegions` is the one place `[θ|m|v]` becomes `[θ|m|v|·]`, so this pins that it agrees
-- with the two axes that cause it — the derived fact and its causes, rather than the derived fact
-- alone. ⚠ The 4-region cases are exactly EMA ∪ accumulation, stated as a partition for §0.4
-- finding 5's reason: a count would not notice a spelling that lands in neither.
#guard table.all (fun (v, e, _, _) => nRegions v == (if e || accOn v then 4 else 3))
#guard table.all (fun (v, _, _, _) => (nRegions v == 4) == (nScalars v == 5))
#guard nRegions "lambaccdp8x64bce" == 4    -- the spelling that falsified the prefix test
#guard nRegions "adamdp128x4wxclipdrop" == 3

#eval do
  IO.println "── variant predicates ──"
  for (v, e, r, s) in table do
    let regions := nRegions v
    let kNote := if accOn v then s!" k={accK v}" else ""
    IO.println s!"  {v} — ema={e} rms={r} drop={s} dropout={cdOn v} accum={accOn v}{kNote} \
→ {regions} blob regions"
  IO.println s!"✓ {table.length} variant spellings, 5 axes, no collision"
