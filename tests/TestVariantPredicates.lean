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
   data-parallel variant, including the committed and gated `enetin_rmsdp64`. ⚠ The collision was
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

private def emaOn (v : String) : Bool := v.startsWith "ema"
private def rmsOn (v : String) : Bool := (v.splitOn "rms").length > 1
private def sdOn  (v : String) : Bool := (v.splitOn "drop").length > 1
/-- ▶ CLASSIFIER DROPOUT (`recipe_gaps.md` gap C, 2026-08-03) — a FOURTH axis, and the first one
    since `sd` that the driver must actually predicate on: it adds one `tensor<B×1280xf32>` input,
    so a misfire is an arity error at best and a mis-walked blob at worst.

    ⚠⚠ **THE MARKER IS `"do"`, NOT `"dropout"`, AND THAT IS FORCED BY `sdOn` ABOVE.** `"dropout"`
    CONTAINS `"drop"`, so a dropout-only variant would read as a stochastic-depth one and the
    driver would try to pack nine mask slots the graph does not have. That is collision #3, and it
    is the first one that was predicted rather than discovered — because this file existed. -/
private def cdOn  (v : String) : Bool := (v.splitOn "do").length > 1

/-- ⭐⭐ GRADIENT ACCUMULATION (`planning/next_session_pipeline_then_r50.md` §4, 2026-08-05) — a
    FIFTH axis, and the second one after `ema` that changes the number of blob REGIONS: `acc<k>x<B>`
    carries `[θ|m|v|G]`, 4 regions and 5 scalars. A misfire misaligns every parameter.

    ⚠ It is a PREFIX test, like `emaOn` and for the same reason — the marker leads, so there is no
    concatenation that can spell it accidentally. ⚠⚠ And that is also its known limit, defect #1's
    shape exactly: a hypothetical `emaacc4x64` would read as EMA and NOT as accumulation. The
    driver closes it by REFUSING the combination outright (both want the fourth region), which is
    the only correct answer anyway; `accOn "emaacc4x64" == false` is pinned below so that limit is
    recorded rather than rediscovered. -/
private def accOn (v : String) : Bool := v.startsWith "acc"

/-- `k`, read back out of the name. ⚠⚠ **This parse is load-bearing and it is the reason `k` is in
    the name at all.** The graph has `1/k` BAKED into `%ob1`/`%ob2`; the driver decides the apply
    cadence. A disagreement does not fail — it trains at a silently wrong effective learning rate.
    `ResNet50RenderB` `#guard`s the round trip on the producing side; this is the consuming side. -/
private def accK (v : String) : Nat :=
  if accOn v then (((v.drop (if v.startsWith "accdp" then 5 else 3)).takeWhile (· != 'x')).toNat?).getD 0
  else 1

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
private def accumSpellings : List String := ["acc4x64", "accdp8x64", "acc2x128"]
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
-- ⚠ THE KNOWN LIMIT, pinned rather than rediscovered: `accOn` is a prefix test, so a two-axis name
-- with `ema` leading would miss it. The driver refuses `ema` × `acc` outright — they want the same
-- fourth region — so this is the correct answer, not a latent bug. Defect #1's exact shape.
#guard accOn "emaacc4x64" == false
#guard emaOn "emaacc4x64" == true
-- ⚠⚠ `k` ROUND-TRIPS, both spellings, including the two-digit case where `accdp` must be stripped
-- as 5 characters and `acc` as 3. Getting this off by one gives k = 8 read as 0 or as 88.
#guard accK "acc4x64" == 4
#guard accK "accdp8x64" == 8
#guard accK "acc2x128" == 2
#guard accK "acc16x32" == 16
#guard accK "accdp16x64" == 16
#guard accK "adamdp64" == 1        -- non-accumulation variants read k = 1, i.e. no accumulation
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

#eval do
  IO.println "── variant predicates ──"
  for (v, e, r, s) in table do
    let regions := if e || accOn v then 4 else 3
    let kNote := if accOn v then s!" k={accK v}" else ""
    IO.println s!"  {v} — ema={e} rms={r} drop={s} dropout={cdOn v} accum={accOn v}{kNote} \
→ {regions} blob regions"
  IO.println s!"✓ {table.length} variant spellings, 5 axes, no collision"
