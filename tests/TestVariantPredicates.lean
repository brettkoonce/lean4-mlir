import LeanMlir.VerifiedNets

/-! # The variant-string predicates, run rather than reasoned about

`LEAN_MLIR_VARIANT` now encodes **four independent axes**, and `trainAdamSched` recovers each with
a string test on the name:

| axis | predicate | what it decides |
|---|---|---|
| EMA | `variant.startsWith "ema"` | an EXTRA `[…\|ema]` blob region and two more scalars |
| RMSProp | `"rms"` substring | the mean-square slot initialises to **1.0**, not 0 |
| stochastic depth | `"drop"` substring | N extra `tensor<Bxf32>` scale inputs |
| classifier dropout | `"do"` substring | ONE extra `tensor<Bx1280xf32>` mask input |

Every one of those is a SILENT wrong answer if it misfires: a 3-region blob fed to a 4-region graph
misaligns every parameter, a zero-initialised mean-square is a different optimizer, and a spurious
drop-scale block is an arity error at best.

⚠⚠ **EMA and accumulation are INDEPENDENT as of 2026-08-27, so the region count is 3, 4 or 5** —
`verified_side_quest_counterparts.md` §6a. The driver used to REFUSE the pairing, and this file's
region guard was a two-way partition that would have gone green on the composed spelling it had
never seen. Both are fixed below; the three-way partition is checked for being *populated* as well
as true, which is the part a count alone does not give you.

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
    -- ▶▶ **EMA × ACCUMULATION — the FIVE-region spellings** (`verified_side_quest_counterparts.md`
    -- §6a). These were UNSPELLABLE until 2026-08-27: `trainAdamSched` threw on the pairing because
    -- both features claimed the fourth region, which is what stopped RSB-A2/A1 from being rendered
    -- faithfully. They are in this table BEFORE the driver can produce one, deliberately — a
    -- partition guard that has only ever seen a two-way split goes green on a graph it has never
    -- seen, and this is the axis where "green" means "every parameter aligned".
    -- ⚠ The `ema` marker LEADS, because `emaOn` is a PREFIX test and `accOn` is a substring one.
    -- So `lamb` ++ `acc` stays in the middle and `ema` goes in front of the whole optimizer name.
  , ("emalambaccdp8x64wxclipbce", true, false, false)
  , ("emalambacc8x64wxclipbce", true, false, false)
  , ("emalambaccdp8x64wxclipbcewd001", true, false, false)
  , ("emaaccdp4x64", true, false, false), ("emaacc4x64", true, false, false)
    -- ▶▶ **…AND WITH STOCHASTIC DEPTH — RSB-A2/A1 COMPLETE** (2026-08-27). Six markers on one
    -- name, and `drop` lands in the MIDDLE (N3's `[clip][drop][bce]`), so it has neighbours on both
    -- sides where `ema` has only one. ⚠ The two new adjacencies are `clip`++`drop` and
    -- `drop`++`bce`; neither spells `"do"` (`dr`, not `do`), which is the collision class that has
    -- already fired three times in this naming.
  , ("emalambaccdp8x64wxclipdropbce", true, false, true)
  , ("emalambacc8x64wxclipdropbce", true, false, true)
  , ("emalambaccdp8x64wxclipdropbcewd001", true, false, true)
  , ("emalambacc8x64wxclipdropbcewd001", true, false, true)
    -- ⭐ …at the REFERENCE's own factorisation of 2048: `k = 4` x 128 per device. ⚠ `k` is one
    -- digit and the batch is three, the reverse of `acc8x64`'s shape, so the `takeWhile (!= 'x')`
    -- parse is exercised from the other side.
  , ("emalambaccdp4x128wxclipdropbcebf16", true, false, true)
  , ("emalambacc4x128wxclipdropbcebf16", true, false, true)
  , ("emalambaccdp4x128wxclipdropbcewd001bf16", true, false, true)
  , ("emalambacc4x128wxclipdropbcewd001bf16", true, false, true)
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
    -- ▶▶ D1 (2026-08-14): the same composed optimizer WITH the global-norm clip, which is the
    -- optimizer timm's `Lamb` actually is (`max_grad_norm = 1.0` by default). ⚠⚠ This is a FIFTH
    -- placement for `clip`: on ConvNeXt it trails everything, on ViT it meets a digit, and here it
    -- is BRACKETED — `wx` before it and `bce` after — inside a name that also carries `acc` and its
    -- `k`. The load-bearing question is not whether `clip` reads as an axis (it is not one) but
    -- whether inserting it between `wx` and `bce` disturbs the `k` PARSE, which is a substring
    -- search followed by a `takeWhile` and is the thing defect #4 already got wrong once.
  , ("lambacc8x64wxclipbce", false, false, false)
  , ("lambaccdp8x64wxclipbce", false, false, false)
    -- ▶▶ The `wdStr` axis (2026-08-14): RSB-A1's decay 0.01 against A3's 0.02. ⚠ `%wd` is a BAKED
    -- constant, so the VALUE has to reach the name or two graphs share one path — `wdVariantMark`.
    -- ⚠⚠ A SIXTH placement, and the first marker that carries DIGITS OF ITS OWN after the batch's:
    -- `…64wxclipbcewd001` puts a second run of digits at the very end, and the `k` parse works by
    -- finding "acc"/"accdp" and taking the digits up to the "x". "The trailing digits cannot reach
    -- the leading ones" is exactly the reasoning defect #4 falsified, so it is RUN.
  , ("lambaccdp8x64wxclipbcewd001", false, false, false)
  , ("lambacc8x64wxclipbcewd001", false, false, false)
    -- ⚠⚠ **bf16, AND THIS TABLE HAD NONE OF IT UNTIL 2026-08-27.** Every bf16 render in the tree
    -- goes through these five predicates and not one bf16 spelling was listed here; the guards
    -- that existed were scattered through the renderer files, one net at a time. A SEVENTH marker
    -- placement, and the one that lands after everything else including the decay's digits — so
    -- `…bcewd001bf16` puts a THIRD token after the batch's digits, and `bf16` carries digits too.
    -- ▶ The specific risks, all run rather than reasoned about: `bf16` must not spell "do" (it
    -- does not, and neither does `drop` ++ `bf16` — "do" needs d-then-o and `drop` is d-r-o-p),
    -- must not start "ema", must not contain "rms", and must not disturb the `k` parse, which
    -- finds "acc"/"accdp" and reads the digits up to the "x".
  , ("lambaccdp8x64wxclipbcebf16", false, false, false)
  , ("lambacc8x64wxclipbcebf16", false, false, false)
  , ("lambaccdp8x64wxclipbcewd001bf16", false, false, false)
  , ("lambacc8x64wxclipbcewd001bf16", false, false, false)
    -- ⚠ and a NON-accumulating bf16 render, so the partition below is tested in both directions:
    -- this is R50's shipped `momdp64bf16`, which must read as 3 regions.
  , ("momdp64bf16", false, false, false)
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
   "lambacc8x64bce", "lambaccdp8x64bce", "lambacc4x64bce", "lambaccdp4x64",
   -- D1's clipped peers — same 4-region graph, one more trailing marker
   "lambacc8x64wxclipbce", "lambaccdp8x64wxclipbce",
   -- the A1-decay peers — same 4-region graph, one more trailing marker
   "lambaccdp8x64wxclipbcewd001", "lambacc8x64wxclipbcewd001",
   -- ⭐ the bf16 peers of both tiers (RSB-A2 and RSB-A1 at 224, 2026-08-27). Same 4-region graph;
   -- `bf16` is the outermost marker and must not reach the `k` parse. ⚠ `momdp64bf16` is
   -- deliberately NOT here — it is the other direction of the partition, a bf16 render with no
   -- accumulation, and listing it would make this check pass for the wrong reason.
   "lambaccdp8x64wxclipbcebf16", "lambacc8x64wxclipbcebf16",
   "lambaccdp8x64wxclipbcewd001bf16", "lambacc8x64wxclipbcewd001bf16",
   -- ⭐⭐ the FIVE-region peers (2026-08-27): accumulation composed with the EMA shadow, which is
   -- RSB-A2/A1's real recipe and was unspellable until the fifth region landed.
   "emalambaccdp8x64wxclipbce", "emalambacc8x64wxclipbce",
   "emalambaccdp8x64wxclipbcewd001", "emaaccdp4x64", "emaacc4x64",
   -- ⭐ and the same four with stochastic depth — RSB-A2/A1 complete
   "emalambaccdp8x64wxclipdropbce", "emalambacc8x64wxclipdropbce",
   "emalambaccdp8x64wxclipdropbcewd001", "emalambacc8x64wxclipdropbcewd001",
   -- ⭐ the ghost-BN-aligned pair, k = 4 x 128 per device
   "emalambaccdp4x128wxclipdropbcebf16", "emalambacc4x128wxclipdropbcebf16",
   "emalambaccdp4x128wxclipdropbcewd001bf16", "emalambacc4x128wxclipdropbcewd001bf16"]
#guard table.all (fun (v, _, _, _) => accOn v == accumSpellings.contains v)
#guard accumSpellings.all (fun v => table.any (fun (t, _, _, _) => t == v))
-- ⚠ and accumulation must disturb NONE of the other three axes it does not compose with. `acc4x64`
-- contains no "rms" and no "drop"; `accdp8x64` puts `dp` inside the prefix, which is a placement no
-- other marker uses, so it is run rather than reasoned about.
-- ⚠⚠ **`emaOn` CAME OUT OF THIS CONJUNCTION on 2026-08-27, and that is a real weakening of the
-- check, stated rather than quietly dropped.** It used to read `!emaOn v && !rmsOn v && …`, which
-- was true only because the driver REFUSED the pairing — the guard was recording a limitation as
-- if it were a naming fact. EMA × accumulation is now a five-region render (RSB-A2/A1), so the
-- pairing is legal and the ordering rule below is what replaces this half of the check.
-- ⚠⚠ **`sdOn` CAME OUT TOO, 2026-08-27, and for the same reason `emaOn` did**: RSB-A2/A1 render
-- accumulation WITH stochastic depth (`emalambaccdp8x64wxclipdropbce`), so a guard saying they
-- never co-occur was recording the absence of a render, not a fact about the naming.
#guard table.all (fun (v, _, _, _) => !accOn v || (!rmsOn v && !cdOn v))
-- ⭐ and the replacement, which is stronger than what it replaces: whenever BOTH fire, the `ema`
-- marker must LEAD. `emaOn` is a prefix test and `accOn` a substring one, so `lambaccema…` would
-- read as accumulation-only and pack four regions into a five-region graph.
#guard table.all (fun (v, _, _, _) => !(accOn v && emaOn v) || v.startsWith "ema")
#guard emaOn "lambaccdp8x64wxclipbceema" == false   -- the trailing spelling: silently 4 regions
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
-- ⚠⚠ **AND `k` SURVIVES D1's TWO EXTRA TRAILING MARKERS.** `wx` ++ `clip` sits between the batch
-- and `bce`, so `k`'s digits are now followed by `x64wxclipbce` rather than `x64bce`. The parse
-- takes digits between `acc`/`accdp` and the `x`, so neither marker can reach them — but "the
-- suffix cannot reach the prefix" is precisely the reasoning defect #4 falsified, so it is run.
#guard accK "lambaccdp8x64wxclipbce" == 8
#guard accK "lambacc8x64wxclipbce" == 8
#guard accK "lambacc8x64wxclipbce" != 64
#guard accOn "lambaccdp8x64wxclipbce" == true
-- ⚠ and the 4-region arithmetic, which is what a checkpoint's size is read against: adding `clip`
-- must not turn RSB-A3's 4-region graph into a 3-region read.
#guard nRegions "lambaccdp8x64wxclipbce" == 4
#guard nScalars "lambaccdp8x64wxclipbce" == 5
-- ⭐⭐ **AND THE SAME NAME WITH THE `ema` PREFIX IS FIVE REGIONS AND SEVEN SCALARS** — RSB-A2's
-- real composition. ⚠ `ema` must not disturb `k`, the optimizer, or any of the other four axes:
-- it is prepended to a string that already carries five markers, which is the outermost placement
-- this table has.
#guard nRegions "emalambaccdp8x64wxclipbce" == 5
#guard nScalars "emalambaccdp8x64wxclipbce" == 7
#guard emaOn "emalambaccdp8x64wxclipbce" == true
#guard accOn "emalambaccdp8x64wxclipbce" == true
#guard accK  "emalambaccdp8x64wxclipbce" == 8
#guard rmsOn "emalambaccdp8x64wxclipbce" == false
#guard sdOn  "emalambaccdp8x64wxclipbce" == false
#guard cdOn  "emalambaccdp8x64wxclipbce" == false
-- ⚠ and with the decay marker too, which is A1's spelling — `wd001` after `bce`, `ema` before all
-- of it, so `k` is bracketed by markers on BOTH sides for the first time in this table.
#guard nRegions "emalambaccdp8x64wxclipbcewd001" == 5
#guard accK "emalambaccdp8x64wxclipbcewd001" == 8
-- ⭐ THE REGION INDICES, which are what a checkpoint is SLICED at rather than merely sized by.
-- `G` before `E` is the ordering that keeps every previously-written blob readable: at `acc`
-- alone the accumulator is region 3, at `ema` alone the shadow is region 3, and only the
-- composition moves the shadow to 4. A reversed order re-homes every committed `ema*` file.
#guard emaRegion "emalambaccdp8x64wxclipbce" == some 4
#guard emaRegion "ema128" == some 3
#guard emaRegion "emarmsdp64" == some 3
#guard emaRegion "lambaccdp8x64wxclipbce" == none
#guard emaRegion "adamdp128x4wxclipdrop" == none
#guard emaScalarOff "emalambaccdp8x64wxclipbce" == 5
#guard emaScalarOff "ema128" == 3
-- ⭐⭐ **RSB-A2 COMPLETE: five regions, seven scalars, AND sixteen drop masks.** `drop` adds graph
-- INPUTS, never regions or scalars, so it must move neither count — that independence is the check.
#guard nRegions "emalambaccdp8x64wxclipdropbce" == 5
#guard nScalars "emalambaccdp8x64wxclipdropbce" == 7
#guard emaRegion "emalambaccdp8x64wxclipdropbce" == some 4
#guard sdOn   "emalambaccdp8x64wxclipdropbce" == true
#guard emaOn  "emalambaccdp8x64wxclipdropbce" == true
#guard accOn  "emalambaccdp8x64wxclipdropbce" == true
#guard accK   "emalambaccdp8x64wxclipdropbce" == 8
#guard cdOn   "emalambaccdp8x64wxclipdropbce" == false   -- ⚠ `dr`, not `do`
#guard rmsOn  "emalambaccdp8x64wxclipdropbce" == false
-- ⚠ and `drop` must not disturb `k` — it now sits between the batch and `bce`, so the digits `8`
-- are followed by `x64wxclipdropbce` rather than `x64wxclipbce`.
#guard accK "emalambaccdp8x64wxclipdropbcewd001" == 8
#guard accK "emalambaccdp8x64wxclipdropbcewd001" != 64
#guard nRegions "emalambaccdp8x64wxclipdropbcewd001" == 5
-- ⚠⚠ THE COUNTERFACTUAL FOR THE NEW ADJACENCIES, run rather than reasoned about — three collisions
-- in this naming have all lived in a PAIR of markers meeting, never in the marker alone.
#guard cdOn "wxclipdrop" == false
#guard cdOn "dropbce" == false
#guard sdOn "wxclipdropbce" == true
-- ⭐ `k = 4` with a THREE-DIGIT batch — the reverse shape of `acc8x64`, and the case where a parse
-- reading "the digits" rather than "the digits before the x" returns 4128 or 128 instead of 4.
#guard accK "emalambaccdp4x128wxclipdropbcebf16" == 4
#guard accK "emalambacc4x128wxclipdropbcebf16" == 4
#guard accK "emalambaccdp4x128wxclipdropbcebf16" != 128
#guard accK "emalambaccdp4x128wxclipdropbcebf16" != 4128
#guard nRegions "emalambaccdp4x128wxclipdropbcebf16" == 5
#guard emaRegion "emalambaccdp4x128wxclipdropbcebf16" == some 4
-- ⚠ the shadow is the LAST region in every spelling, which is what lets `scoreCheckpoint` bound
-- the slice at `nRegions` without a second index.
#guard table.all (fun (v, _, _, _) =>
  match emaRegion v with | some i => i + 1 == nRegions v | none => true)
-- ⚠ and `clip` in this bracketed placement must invent none of the four axes it is not
-- ⚠⚠ `k` SURVIVES THE DECAY MARKER TOO, and this is the one with digits in it. `wd001` appends a
-- second numeric run AFTER the batch, so if the parse ever became "the last digits" or "all the
-- digits" rather than "the digits between acc and x", these are what say so.
#guard accK "lambaccdp8x64wxclipbcewd001" == 8
#guard accK "lambacc8x64wxclipbcewd001" == 8
#guard accK "lambaccdp8x64wxclipbcewd001" != 1
#guard accK "lambaccdp8x64wxclipbcewd001" != 64
#guard accOn "lambaccdp8x64wxclipbcewd001" == true
#guard nRegions "lambaccdp8x64wxclipbcewd001" == 4
-- ⚠ and `wd` must invent none of the five axes. ⭐ The one worth running rather than reasoning
-- about: `wd001` ends in digits and begins with `w`, and `do` is the two-character marker that has
-- already collided once. `w`+`d` is not `d`+`o`, but that is the argument `rms`+`dp` ⊇ "sd" broke.
#guard cdOn  "lambaccdp8x64wxclipbcewd001" == false
#guard sdOn  "lambaccdp8x64wxclipbcewd001" == false
#guard rmsOn "lambaccdp8x64wxclipbcewd001" == false
#guard emaOn "lambaccdp8x64wxclipbcewd001" == false

#guard emaOn "lambaccdp8x64wxclipbce" == false
#guard rmsOn "lambaccdp8x64wxclipbce" == false
#guard sdOn  "lambaccdp8x64wxclipbce" == false
#guard cdOn  "lambaccdp8x64wxclipbce" == false
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
-- ⚠⚠ **A THREE-WAY PARTITION SINCE 2026-08-27, and it used to be two-way.** The guard read
-- `nRegions v == (if e || accOn v then 4 else 3)` — true of every spelling that existed, and
-- SILENTLY TRUE of a five-region one too, because no variant could spell both axes at once. That
-- is finding 5's shape exactly: a count does not notice a case that lands in neither bucket. The
-- form below is a SUM over the two independent axes, so a new axis cannot hide inside an `if`.
#guard table.all (fun (v, e, _, _) =>
  nRegions v == 3 + (if accOn v then 1 else 0) + (if e then 1 else 0))
-- and the scalar tail moves in lockstep: two slots per extra region, never one and never three.
#guard table.all (fun (v, _, _, _) => nScalars v == 3 + 2 * (nRegions v - 3))
-- ⭐ …and all THREE buckets are actually populated, so the two guards above are not vacuously
-- true of a table that happens to contain no five-region spelling. That is the check that would
-- have caught the two-way partition the day the fifth region landed.
#guard table.any (fun (v, _, _, _) => nRegions v == 3)
#guard table.any (fun (v, _, _, _) => nRegions v == 4)
#guard table.any (fun (v, _, _, _) => nRegions v == 5)
#guard nRegions "lambaccdp8x64bce" == 4    -- the spelling that falsified the prefix test
#guard nRegions "adamdp128x4wxclipdrop" == 3
-- ⭐ RSB-A2/A1 at 224 (2026-08-27), the outermost marker placement this table has. `k` must still
-- come back as 8 through TWO trailing markers carrying digits of their own — `wd001` then `bf16`.
#guard accK "lambaccdp8x64wxclipbcewd001bf16" == 8
#guard accK "lambacc8x64wxclipbcewd001bf16" == 8
#guard accK "lambaccdp8x64wxclipbcebf16" == 8
#guard nRegions "lambaccdp8x64wxclipbcewd001bf16" == 4
-- ⚠ and the other direction: bf16 alone must not invent the fourth region.
#guard nRegions "momdp64bf16" == 3

#eval do
  IO.println "── variant predicates ──"
  for (v, e, r, s) in table do
    let regions := nRegions v
    let kNote := if accOn v then s!" k={accK v}" else ""
    IO.println s!"  {v} — ema={e} rms={r} drop={s} dropout={cdOn v} accum={accOn v}{kNote} \
→ {regions} blob regions"
  IO.println s!"✓ {table.length} variant spellings, 5 axes, no collision"
