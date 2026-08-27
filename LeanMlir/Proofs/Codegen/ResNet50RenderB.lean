import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # ResNet-50 train step rendered from the verified AST, at the BATCHED index

R50 phase 2 (`planning/next_session_pipeline_then_r50.md` §3.2). The bottleneck peer of
`ResNet34RenderB.lean`, block for block: batch BN (`bnBatchF`, reduced over `[0,2,3]`), the whole
graph at `N := B`, the un-fused `*GradB` parameter gradients, and the proven AdamW /
heavy-ball tail — `optOne`/`optConstsB` are **imported, not copied**, so the optimizer has one
definition across both nets.

## The three block forms, and why the third exists

`Resnet50BlocksCertified.lean` carries the certified VJPs; this file renders them.

| renderer | block | where |
|---|---|---|
| `bnkIdFwdB` / `bnkIdBackGradB` | identity, `oc → mid → mid → oc` | 12 blocks |
| `bnkProjFwdB` / `bnkProjBackGradB` | ⭐ **stride-1** projection | **stage 1 block 0 only** |
| `bnkStridedFwdB` / `bnkStridedBackGradB` | strided projection | stages 2/3/4 block 0 |

⚠ **The stride is on the 3×3 (`W2`), not the leading 1×1** — v1.5 / torchvision, which is what
`jax/MainResnet50Imagenet.lean` trains. So in the strided block `conv1`, `bn1` and `relu1` all run
at the **input** resolution `2hh`, and only `conv2` decimates. That asymmetry is why the strided
renderer carries two sets of zero-vectors (`zIn`/`zMidIn` at `2hh`, `zMid`/`zOut` at `hh`) where
the other two carry one.

⚠ **`mid = oc / 4`**, so an identity block's `mid` is a QUARTER of its channel count, not a
multiple. `VerifiedSpec.bottleneckStageSpec` already computes it that way and already selects the
projecting form when `stride != 1 || ic != oc` — which is exactly how stage 1 (64→256 at stride 1)
gets a projection. This file's block sequence must agree with that spec's ORDER, because the
driver packs `[θ|m|v]` off `net.specs`; `r50SigList` below is the render's side of that contract
and `#guard`s pin the counts.

## ⚠ No conv biases

Every conv here is bias-free (`convBnNB` in the spec — 9 tensors per identity block, 12 per
projection block), matching R34's ImageNet render. The `convBias` plumbing R34's renderer carries
for its CIFAR-era artifacts is deliberately absent rather than threaded and passed `false`.

## ⚠⚠ There is no incumbent hand-written R50 render to tie against

§3.2. Every other net's swap onto the verified renderer was licensed by a bit-exact numeric tie
against the hand-written artifact it replaced. R50 has no such artifact, so that license does not
exist here and **must not be implied**. The substitutes are the layer-level VJP oracle
(`tests/vjp_oracle/run.sh`) and a keep-1 known-answer check, and whichever one is used has to be
named in the commit that ships a number off this render.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- The zero constant bound to every conv's bias operand. R50 is bias-free (`convBnNB`), but the
    proven conv ops still TAKE a bias, so it binds to `%zb{c}` and `zeroBiasPrelude` emits those
    constants once at the top of the body. Routed through `biasName` rather than spelled inline so
    the naming has one definition shared with R34's renderer. -/
private def zb (c : Nat) : String := biasName false "" c

-- ════════════════════════════════════════════════════════════════
-- § Signature lists — the render's side of the driver contract
-- ════════════════════════════════════════════════════════════════

/-- One bottleneck block's parameters, in `bneckIdBlk` order: `W1 g1 bt1 W2 g2 bt2 W3 g3 bt3`,
    plus `Wp gp btp` LAST when it projects. ⚠ The projection going last is not cosmetic — it is
    `bneckDownBlk`'s order and therefore the order the driver's `[θ|m|v]` walk assumes. -/
private def bnkSig (p : String) (cin mid oc : Nat) (proj : Bool) : List (String × List Nat) :=
  [(s!"%{p}W1", [mid,cin,1,1]), (s!"%{p}g1", [mid]), (s!"%{p}bt1", [mid]),
   (s!"%{p}W2", [mid,mid,3,3]), (s!"%{p}g2", [mid]), (s!"%{p}bt2", [mid]),
   (s!"%{p}W3", [oc,mid,1,1]),  (s!"%{p}g3", [oc]),  (s!"%{p}bt3", [oc])] ++
  (if proj then
     [(s!"%{p}Wp", [oc,cin,1,1]), (s!"%{p}gp", [oc]), (s!"%{p}btp", [oc])]
   else [])

/-- One bottleneck STAGE's parameters: block 0 projects (channels change, and in stage 1 that is
    the only thing that changes), the rest are identity blocks reading `oc`. -/
private def bnkStageSig (s : String) (ic oc count : Nat) : List (String × List Nat) :=
  let mid := oc / 4
  (bnkSig s!"{s}b0" ic mid oc true) ++
  ((List.range (count - 1)).flatMap (fun i => bnkSig s!"{s}b{i+1}" oc mid oc false))

/-- **The 161 R50 parameter inputs**, in func-arg order: stem (3), the `[3,4,6,3]` bottleneck
    stages, then the head. Single source for `%p`, `%pm`, `%pv` and the return order. -/
def r50ShapeList (nClasses : Nat) : List (String × List Nat) :=
  [("%sW", [64,3,7,7]), ("%sg", [64]), ("%sbt", [64])] ++
  bnkStageSig "s1"   64  256 3 ++
  bnkStageSig "s2"  256  512 4 ++
  bnkStageSig "s3"  512 1024 6 ++
  bnkStageSig "s4" 1024 2048 3 ++
  [("%Wd", [2048, nClasses]), ("%bd", [nClasses])]

/-- The same list as MLIR types. Derived, so the shapes have one definition. -/
def r50SigList (nClasses : Nat) : List (String × String) :=
  (r50ShapeList nClasses).map (fun (n, ds) => (n, ty ds))

/-- One block's BN running-stat slots, in BN-forward order `n1 n2 n3 [np]`. -/
private def bnkStatSig (p : String) (mid oc : Nat) (proj : Bool) : List (String × String) :=
  [(s!"%{p}n1mu", ty [mid]), (s!"%{p}n1var", ty [mid]),
   (s!"%{p}n2mu", ty [mid]), (s!"%{p}n2var", ty [mid]),
   (s!"%{p}n3mu", ty [oc]),  (s!"%{p}n3var", ty [oc])] ++
  (if proj then [(s!"%{p}npmu", ty [oc]), (s!"%{p}npvar", ty [oc])] else [])

private def bnkStageStatSig (s : String) (oc count : Nat) : List (String × String) :=
  let mid := oc / 4
  (bnkStatSig s!"{s}b0" mid oc true) ++
  ((List.range (count - 1)).flatMap (fun i => bnkStatSig s!"{s}b{i+1}" mid oc false))

/-- **The 106 running-stat inputs** — 53 BN layers × (μ, var), μ and var interleaved per layer,
    which is how the driver packs `runningBnStats` off `bnChannels`. -/
def r50StatSigList : List (String × String) :=
  [("%stnmu", ty [64]), ("%stnvar", ty [64])] ++
  bnkStageStatSig "s1"  256 3 ++
  bnkStageStatSig "s2"  512 4 ++
  bnkStageStatSig "s3" 1024 6 ++
  bnkStageStatSig "s4" 2048 3

-- 3 stem + (12+9+9) + (12+27) + (12+45) + (12+18) + 2 head = 161 parameter tensors.
#guard (r50SigList 1000).length == 161
-- 53 BN layers ⇒ 106 stat slots. stem 1 + s1 (4+3+3=10) + s2 13 + s3 19 + s4 10 = 53.
#guard r50StatSigList.length == 106

-- ════════════════════════════════════════════════════════════════
-- § Bottleneck block forwards (batch BN)
-- ════════════════════════════════════════════════════════════════

/-- Saved forward SSA names the bottleneck's backward + gradient passes reference. The 3-conv peer
    of `BFwdB`: two interior ReLUs (`r1`, `r2`) where the basic block has one. -/
structure BNFwd where
  code : String
  xin : String
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum
  c1 : String        -- conv1 out (= BN1 in)
  n1 : String        -- BN1 out (= relu1 pre-activation)
  r1 : String        -- relu1 out (= conv2 in)
  c2 : String        -- conv2 out (= BN2 in)
  n2 : String        -- BN2 out (= relu2 pre-activation)
  r2 : String        -- relu2 out (= conv3 in)
  c3 : String        -- conv3 out (= BN3 in)
  cp : String        -- projection conv out ("" for identity)
deriving Inhabited

/-- **R50's stochastic-depth site count** — one per bottleneck block, `[3,4,6,3] = 16`.

    ⚠ It is the count the RAMP's denominator is read against (`totalDrop − 1 = 15`), so it is a
    definition rather than a literal 16 sprinkled at three sites. -/
def r50DropTotal : Nat := 16

/-- The `%dp0 … %dp15` mask arguments, `tensor<Bxf32>` each — appended to the train step's
    signature only under `sd`. Mirrors `cnxDropSig`. -/
def r50DropSig (B : Nat) (sd : Bool) : String :=
  if sd then String.join ((List.range r50DropTotal).map (fun i => s!", {dpName i}: {ty [B]}"))
  else ""

/-- `some i` under `sd`, `none` otherwise — the per-block ramp index handed to the six block
    emitters. ⚠ **The index is the BLOCK index and the two coincide here**, because every R50
    bottleneck drops; EfficientNet's do not (its reference advances the ramp counter inside a skip
    guard), which is why `efficientnetVerified.dropKeeps` is a literal array and this is a range.
    ⚠ At `sd := false` this is `none` at every site and NOT ONE `pretty` call happens, so every
    committed R50 artifact re-renders byte-identically. -/
def dpAt (sd : Bool) (i : Nat) : Option Nat := if sd then some i else none

/-- Identity bottleneck forward: `1×1 → BN → relu → 3×3 → BN → relu → 1×1 → BN → (+x) → relu`,
    all at `hh`, channels `oc → mid → mid → oc`. -/
private def bnkIdFwdB (B mid oc hh : Nat) (epsStr p xName : String)
    (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BNFwd := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (zb mid) zk1 zm else .conv (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zOut))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMid))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (zb mid) zk2 zm else .conv (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMid))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W3" (zb oc) zk3 zo else .conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  -- ▶ STOCHASTIC DEPTH, on the residual branch and before the add. See this function's docstring
  -- for why the placement is the correctness question and why an all-ones gate cannot see it.
  let (cD, nD) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nN3 zOut))
    | none   => pure ("", nN3)
  let (cA,  nA)  ← pretty B (.addVB (.operand nD zOut) (.operand xName zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cD ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := "" }

/-- ⭐ **Stride-1 projection bottleneck forward** — R50 stage 1 block 0, and nowhere else.
    `cin → mid → mid → oc` with the resolution unchanged, so the projection is a plain `1×1`
    conv → BN, NOT the strided one `bnkStridedFwdB` uses. -/
private def bnkProjFwdB (B cin mid oc hh : Nat) (epsStr p xName : String)
    (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BNFwd := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn  : Vec (B*(cin*hh*ww)) := fun _ => 0
  let zOut : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (zb mid) zk1 zm else .conv (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zIn))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMid))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (zb mid) zk2 zm else .conv (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMid))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W3" (zb oc) zk3 zo else .conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  -- the projection: a STRIDE-1 1×1 conv. `.conv`, not `.convStrided` — the whole point of this form.
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}Wp" (zb oc) zkp zo else .conv (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo) (.operand xName zIn))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zo zo (.operand nCp zOut))
  -- ▶ STOCHASTIC DEPTH on the residual branch. ⚠⚠ **The PROJECTION is not dropped** — the
  -- reference scales `out` and leaves `shortcut` alone (`bottleneck_block_down`), so a render that
  -- dropped the sum would scale the skip too. That is a different function, and it still trains.
  let (cD, nD) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nN3 zOut))
    | none   => pure ("", nN3)
  let (cA,  nA)  ← pretty B (.addVB (.operand nD zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cD ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := nCp }

/-- **Strided projection bottleneck forward** — stages 2/3/4 block 0. `cin → mid → mid → oc`,
    `2hh → hh`.

    ⚠ `conv1`/`bn1`/`relu1` run at the **input** resolution `2hh`; only `conv2` (the 3×3) is
    strided. v1.5. -/
private def bnkStridedFwdB (B cin mid oc hh : Nat) (epsStr p xName : String)
    (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BNFwd := do
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn    : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zMidIn : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zMid   : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zOut   : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := 2*hh) (w := 2*ww) zrnd s!"%{p}W1" (zb mid) zk1 zm else .conv (h := 2*hh) (w := 2*ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zIn))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMidIn))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*(2*hh)*(2*ww))) (.operand nN1 zMidIn))
  -- ⚠⚠ THE STRIDE LIVES HERE, on the 3×3. Moving it to `W1` above is ResNet v1, a different net.
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (zb mid) zk2 zm else .convStrided (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMidIn))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W3" (zb oc) zk3 zo else .conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := hh) (w := ww) zrnd s!"%{p}Wp" (zb oc) zkp zo else .convStrided (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo) (.operand xName zIn))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zo zo (.operand nCp zOut))
  -- ▶ STOCHASTIC DEPTH on the residual branch. ⚠⚠ **The PROJECTION is not dropped** — the
  -- reference scales `out` and leaves `shortcut` alone (`bottleneck_block_down`), so a render that
  -- dropped the sum would scale the skip too. That is a different function, and it still trains.
  let (cD, nD) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nN3 zOut))
    | none   => pure ("", nN3)
  let (cA,  nA)  ← pretty B (.addVB (.operand nD zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cD ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Bottleneck block backwards + UN-FUSED parameter gradients
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck backward + its 9 parameter gradients.

    Cotangent chain: `da` (output relu) → `bn3` → `conv3` → `relu2` → `bn2` → `conv2` → `relu1`
    → `bn1` → `conv1`, then `dx = dc1 + da` (the identity skip carries `da` unchanged).
    ⚠ Each BN's γ/β gradient reads the cotangent at that BN's OUTPUT: `da` for bn3, `dr2` for
    bn2, `dr1` for bn1 — off by one and the gradient is silently wrong. -/
private def bnkIdBackGradB (B mid oc hh : Nat) (epsStr p : String) (f : BNFwd) (dyName : String)
    (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zbnO  : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM  : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  -- ▶▶ **THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK** (`Proofs.dropPath_vjp_is_self`):
  -- the map is diagonal, so it is its own transpose. What it costs is not an emitter — it is
  -- knowing WHICH cotangent each consumer takes.
  --
  -- ⚠⚠ **THE DROPPED COTANGENT FEEDS THE BRANCH; THE UNDROPPED ONE FEEDS THE SKIP.** The forward is
  -- `a = drop(n3) + skip`, so `∂a/∂n3` carries the mask and `∂a/∂skip` does not. Everything
  -- upstream of the last BN takes `nDd` — including bn3's OWN γ/β gradients, which read the
  -- cotangent at bn3's output; the residual `addVB` and the PROJECTION's whole backward keep `nDa`.
  -- ▶ Getting it wrong type-checks, trains and descends. It is the defect ConvNeXt's `bwdBlockB`
  -- records as *18 of 180 gradients wrong by a per-example factor, on the parameter stochastic
  -- depth is about* — here 3 of 161 per block, 48 across the net.
  let (cDd, nDd) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nDa zOut))
    | none   => pure ("", nDa)
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDd zbnO))
  let (cDc3, nDc3) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}W3" zk3 zo (.operand nDn3 zOut) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  let (cDc2, nDc2) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}W2" zk2 zm (.operand nDn2 zMid) else .convBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMid (.operand nDc2 zMid))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnM (.operand nDr1 zbnM))
  let (cDc1, nDc1) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := oc) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}W1" zk1 zm (.operand nDn1 zMid) else .convBackBatched (N := B) (ic := oc) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMid))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zOut) (.operand nDa zOut))
  -- parameter gradients, func-arg order: W1 g1 bt1 W2 g2 bt2 W3 g3 bt3
  let (cW1, nW1) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := oc) (oc := mid) (h := hh) (w := ww) zrnd xName zm zOut zk1 (.operand nDn1 zMid) else .convWeightGradB (ic := oc) (oc := mid) (h := hh) (w := ww) xName zm zOut zk1 (.operand nDn1 zMid))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnM (.operand nDr1 zbnM))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr1 zbnM))
  let (cW2, nW2) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd f.r1 zm zMid zk2 (.operand nDn2 zMid) else .convWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMid zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd f.r2 zo zMid zk3 (.operand nDn3 zOut) else .convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  -- ⚠ `nDd`, not `nDa` — bn3 is ON the dropped branch, so its own γ/β take the masked cotangent.
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDd zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDd zbnO))
  pure { code := cDa ++ cDd ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,oc,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩] }

/-- ⭐ **Stride-1 projection bottleneck backward** + its 12 parameter gradients — stage 1 block 0.
    Identical to the identity backward except the skip branch carries its own `bn→conv` backward
    (`dnp`/`dcp`) and `dx = dc1 + dcp`. -/
private def bnkProjBackGradB (B cin mid oc hh : Nat) (epsStr p : String) (f : BNFwd)
    (dyName : String) (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp   : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn   : Vec (B*(cin*hh*ww)) := fun _ => 0
  let zOut  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zbnO  : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM  : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  -- ▶▶ **THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK** (`Proofs.dropPath_vjp_is_self`):
  -- the map is diagonal, so it is its own transpose. What it costs is not an emitter — it is
  -- knowing WHICH cotangent each consumer takes.
  --
  -- ⚠⚠ **THE DROPPED COTANGENT FEEDS THE BRANCH; THE UNDROPPED ONE FEEDS THE SKIP.** The forward is
  -- `a = drop(n3) + skip`, so `∂a/∂n3` carries the mask and `∂a/∂skip` does not. Everything
  -- upstream of the last BN takes `nDd` — including bn3's OWN γ/β gradients, which read the
  -- cotangent at bn3's output; the residual `addVB` and the PROJECTION's whole backward keep `nDa`.
  -- ▶ Getting it wrong type-checks, trains and descends. It is the defect ConvNeXt's `bwdBlockB`
  -- records as *18 of 180 gradients wrong by a per-example factor, on the parameter stochastic
  -- depth is about* — here 3 of 161 per block, 48 across the net.
  let (cDd, nDd) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nDa zOut))
    | none   => pure ("", nDa)
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDd zbnO))
  let (cDc3, nDc3) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}W3" zk3 zo (.operand nDn3 zOut) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  let (cDc2, nDc2) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}W2" zk2 zm (.operand nDn2 zMid) else .convBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMid (.operand nDc2 zMid))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnM (.operand nDr1 zbnM))
  let (cDc1, nDc1) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := cin) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}W1" zk1 zm (.operand nDn1 zMid) else .convBackBatched (N := B) (ic := cin) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMid))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDcp, nDcp) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}Wp" zkp zo (.operand nDnp zOut) else .convBackBatched (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" zkp zo (.operand nDnp zOut))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zIn) (.operand nDcp zIn))
  let (cW1, nW1) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := cin) (oc := mid) (h := hh) (w := ww) zrnd xName zm zIn zk1 (.operand nDn1 zMid) else .convWeightGradB (ic := cin) (oc := mid) (h := hh) (w := ww) xName zm zIn zk1 (.operand nDn1 zMid))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnM (.operand nDr1 zbnM))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr1 zbnM))
  let (cW2, nW2) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd f.r1 zm zMid zk2 (.operand nDn2 zMid) else .convWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMid zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd f.r2 zo zMid zk3 (.operand nDn3 zOut) else .convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  -- ⚠ `nDd`, not `nDa` — bn3 is ON the dropped branch, so its own γ/β take the masked cotangent.
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDd zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDd zbnO))
  let (cWp, nWp) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := cin) (oc := oc) (h := hh) (w := ww) zrnd xName zo zIn zkp (.operand nDnp zOut) else .convWeightGradB (ic := cin) (oc := oc) (h := hh) (w := ww) xName zo zIn zkp (.operand nDnp zOut))
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbnO (.operand nDa zbnO))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  pure { code := cDa ++ cDd ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++
                 cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3 ++ cWp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,cin,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩,
                ⟨s!"{p}Wp", nWp, [oc,cin,1,1]⟩, ⟨s!"{p}gp", ngp, [oc]⟩, ⟨s!"{p}btp", ntp, [oc]⟩] }

/-- **Strided projection bottleneck backward** + its 12 parameter gradients — stages 2/3/4 block 0.

    ⚠ `dc2` is the STRIDED conv backward, so it takes the cotangent from `hh` back up to `2hh`;
    everything upstream of it (`dr1`, `dn1`, `dc1`, `W1`'s grad) lives at `2hh`. -/
private def bnkStridedBackGradB (B cin mid oc hh : Nat) (epsStr p : String) (f : BNFwd)
    (dyName : String) (bf16 : Bool := false) (drop : Option Nat := none) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp   : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn     : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zMidIn  : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zbnMIn  : Vec (B*(mid*((2*hh)*(2*ww)))) := fun _ => 0
  let zMid    : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zOut    : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zbnO    : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM    : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  -- ▶▶ **THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK** (`Proofs.dropPath_vjp_is_self`):
  -- the map is diagonal, so it is its own transpose. What it costs is not an emitter — it is
  -- knowing WHICH cotangent each consumer takes.
  --
  -- ⚠⚠ **THE DROPPED COTANGENT FEEDS THE BRANCH; THE UNDROPPED ONE FEEDS THE SKIP.** The forward is
  -- `a = drop(n3) + skip`, so `∂a/∂n3` carries the mask and `∂a/∂skip` does not. Everything
  -- upstream of the last BN takes `nDd` — including bn3's OWN γ/β gradients, which read the
  -- cotangent at bn3's output; the residual `addVB` and the PROJECTION's whole backward keep `nDa`.
  -- ▶ Getting it wrong type-checks, trains and descends. It is the defect ConvNeXt's `bwdBlockB`
  -- records as *18 of 180 gradients wrong by a per-example factor, on the parameter stochastic
  -- depth is about* — here 3 of 161 per block, 48 across the net.
  let (cDd, nDd) ← match drop with
    | some i => pretty B (.dropPathB (N := B) (n := oc*hh*ww) (dpName i) (fun _ => 0 : Vec B)
                            (.operand nDa zOut))
    | none   => pure ("", nDa)
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDd zbnO))
  let (cDc3, nDc3) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}W3" zk3 zo (.operand nDn3 zOut) else .convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  -- the strided conv backward: hh → 2hh
  let (cDc2, nDc2) ← pretty B (if bf16 then .convStridedBackBatchedBf16 (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd s!"%{p}W2" zk2 zm (.operand nDn2 zMid) else .convStridedBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMidIn (.operand nDc2 zMidIn))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnMIn (.operand nDr1 zbnMIn))
  let (cDc1, nDc1) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) zrnd s!"%{p}W1" zk1 zm (.operand nDn1 zMidIn) else .convBackBatched (N := B) (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMidIn))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDcp, nDcp) ← pretty B (if bf16 then .convStridedBackBatchedBf16 (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) zrnd s!"%{p}Wp" zkp zo (.operand nDnp zOut) else .convStridedBackBatched (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" zkp zo (.operand nDnp zOut))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zIn) (.operand nDcp zIn))
  let (cW1, nW1) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) zrnd xName zm zIn zk1 (.operand nDn1 zMidIn) else .convWeightGradB (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) xName zm zIn zk1 (.operand nDn1 zMidIn))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnMIn (.operand nDr1 zbnMIn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) (.operand nDr1 zbnMIn))
  let (cW2, nW2) ← pretty B (if bf16 then .convStridedWeightGradBBf16 (ic := mid) (oc := mid) (h := hh) (w := ww) zrnd f.r1 zm zMidIn zk2 (.operand nDn2 zMid) else .convStridedWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMidIn zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (if bf16 then .convWeightGradBBf16 (ic := mid) (oc := oc) (h := hh) (w := ww) zrnd f.r2 zo zMid zk3 (.operand nDn3 zOut) else .convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  -- ⚠ `nDd`, not `nDa` — bn3 is ON the dropped branch, so its own γ/β take the masked cotangent.
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDd zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDd zbnO))
  let (cWp, nWp) ← pretty B (if bf16 then .convStridedWeightGradBBf16 (ic := cin) (oc := oc) (h := hh) (w := ww) zrnd xName zo zIn zkp (.operand nDnp zOut) else .convStridedWeightGradB (ic := cin) (oc := oc) (h := hh) (w := ww) xName zo zIn zkp (.operand nDnp zOut))
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbnO (.operand nDa zbnO))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  pure { code := cDa ++ cDd ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++
                 cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3 ++ cWp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,cin,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩,
                ⟨s!"{p}Wp", nWp, [oc,cin,1,1]⟩, ⟨s!"{p}gp", ngp, [oc]⟩, ⟨s!"{p}btp", ntp, [oc]⟩] }

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched train step
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- Everything the whole-net render needs out of ONE forward traversal of ResNet-50: the emitted
    code, the logits, and every saved activation the backward reads.

    ⭐⭐ **This exists so `@resnet50_fwd` and `@resnet50_adam_train_step` cannot be different nets.**
    They were: `resnet50FwdFaithfulV` built its forward from the PER-EXAMPLE chain (`r50FwdChain`,
    which reaches `bnSite` and therefore `bnPerChannelF`, reduce `[2,3]`, divisor `H·W`) while the
    train step is batch BN (reduce `[0,2,3]`, divisor `B·H·W`). Its own docstring claimed "the same
    forward the train step differentiates", which was the invariant that did not hold.
    `scripts/regen_verified_mlir.sh check` could not see it: it only ever paired a forward with the
    SGD train step, and R50 has none (`planning/mnv4_verified.md` §3d(b)).

    ⚠ The eval forward is deliberately NOT moved onto this chain. `bnEval` reads frozen per-channel
    statistics as graph inputs, so it is the same arithmetic in either vocabulary and has no
    BN-world to disagree about — and it is the artifact that actually scores, so leaving its bytes
    untouched keeps 89.86% (`runs/r50_imagenette_adam_80ep.log`) exactly where it is. -/
structure R50FwdRecB where
  code : String
  logits : String
  stc : String            -- stem conv out (= stem-BN input)
  stn : String            -- stem BN out   (= stem-relu pre-activation)
  str : String            -- stem relu out (= maxpool input)
  stp : String            -- maxpool out
  gap : String            -- GAP out (= dense input)
  b : Array BNFwd         -- the 16 bottleneck blocks, in forward order
deriving Inhabited

set_option maxRecDepth 4000000 in
/-- **The ResNet-50 forward chain at the BATCHED index** — one traversal, consumed by both
    `@resnet50_fwd` and the train step that differentiates it. -/
def r50FwdChainB (B nClasses : Nat) (epsStr : String) (q : Nat := 7)
    -- ▶ TRAILING and defaulted, so `@resnet50_fwd` and every committed train step that does
    -- not ask for bf16 re-render byte-identical (gate 1).
    (bf16 : Bool := false)
    -- ▶ `sd` = stochastic depth, RSB-A2/A1's `dropPath := 0.05`. TRAILING and defaulted, so the
    -- committed `@resnet50_fwd` and every drop-free train step re-render byte-identically.
    (sd : Bool := false) :
    StateM Nat R50FwdRecB := do
  -- The ladder, bottom-up. At q = 7: 5→…  no — at q = 7 these are 7, 14, 28, 56, 112 and the
  -- input is 224; at q = 5 they are 5, 10, 20, 40, 80 and the input is 160.
  let q5 := q            -- stage 4 / the GAP window
  let q4 := 2 * q5       -- stage 3
  let q3 := 2 * q4       -- stage 2
  let q2 := 2 * q3       -- stage 1, and the max-pool's output
  let q1 := 2 * q2       -- the stem conv's output
  -- ═══ stem: 7×7/s2 conv → batch BN → relu → He et al.'s 3×3/s2 pool (img→img/2→img/4) ═══
  -- ⚠ `2*q1` rather than a name, because `convStrided (h := q1)` demands its operand at exactly
  -- `Vec (B*(3*(2*q1)*(2*q1)))` and any other spelling of the same number is a different TERM.
  let zx    : Vec (B*(3*(2*q1)*(2*q1))) := fun _ => 0
  let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64   : Vec 64 := fun _ => 0
  let z112  : Vec (B*(64*q1*q1)) := fun _ => 0
  let z112b : Vec (B*(64*(q1*q1))) := fun _ => 0
  let z56   : Vec (B*(64*q2*q2)) := fun _ => 0
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk*`/`zb`/`zOut` are: the render
  -- produces TEXT, and `skel` erases every ℝ payload before a token is emitted. The
  -- rounding-bearing `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let (cStc, nStc) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := q1) (w := q1) zrnd "%sW" (zb 64) zSk z64 else .convStrided (h := q1) (w := q1) "%sW" (zb 64) zSk z64) (.operand "%x" zx))
  let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 64) (h := q1) (w := q1) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*q1*q1)) (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool3s2 (c := 64) (h := q2) (w := q2)) (.operand nStr z112))
  -- ═══ 16 bottleneck blocks, [3,4,6,3] ═══
  let f1  ← bnkProjFwdB    B   64  64  256 q2 epsStr "s1b0" nStp bf16 (dpAt sd 0)   -- ⭐ the stride-1 projection
  let f2  ← bnkIdFwdB      B       64  256 q2 epsStr "s1b1" f1.o bf16 (dpAt sd 1)
  let f3  ← bnkIdFwdB      B       64  256 q2 epsStr "s1b2" f2.o bf16 (dpAt sd 2)
  let f4  ← bnkStridedFwdB B  256 128  512 q3 epsStr "s2b0" f3.o bf16 (dpAt sd 3)
  let f5  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b1" f4.o bf16 (dpAt sd 4)
  let f6  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b2" f5.o bf16 (dpAt sd 5)
  let f7  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b3" f6.o bf16 (dpAt sd 6)
  let f8  ← bnkStridedFwdB B  512 256 1024 q4 epsStr "s3b0" f7.o bf16 (dpAt sd 7)
  let f9  ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b1" f8.o bf16 (dpAt sd 8)
  let f10 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b2" f9.o bf16 (dpAt sd 9)
  let f11 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b3" f10.o bf16 (dpAt sd 10)
  let f12 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b4" f11.o bf16 (dpAt sd 11)
  let f13 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b5" f12.o bf16 (dpAt sd 12)
  let f14 ← bnkStridedFwdB B 1024 512 2048  q5 epsStr "s4b0" f13.o bf16 (dpAt sd 13)
  let f15 ← bnkIdFwdB      B      512 2048  q5 epsStr "s4b1" f14.o bf16 (dpAt sd 14)
  let f16 ← bnkIdFwdB      B      512 2048  q5 epsStr "s4b2" f15.o bf16 (dpAt sd 15)
  -- ═══ head: GAP(7×7) → dense(2048→nClasses) ═══
  let zL    : Vec (B*(2048*q5*q5)) := fun _ => 0
  let z2048 : Vec (B*2048) := fun _ => 0
  let zWd   : Mat 2048 nClasses := fun _ _ => 0
  let zNC   : Vec nClasses := fun _ => 0
  let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
  let zNCp  : Vec (B*nClasses) := fun _ => 0
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 2048) (h := q5) (w := q5)) (.operand f16.o zL))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z2048))
  pure { code := cStc ++ cStn ++ cStr ++ cStp ++
                 f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
                 f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
                 f15.code ++ f16.code ++ cGap ++ cLog,
         logits := nLog, stc := nStc, stn := nStn, str := nStr, stp := nStp, gap := nGap,
         b := #[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16] }

/-- **ResNet-50 `[3,4,6,3]` bottleneck train step, batch-BN, rendered at `N := B`.**

    161 θ / 161 m / 161 v, `%lr`/`%bc1`/`%bc2`, 106 running-stat slots and `%onehot` in;
    161 θ' / 161 m' / 161 v', `%loss`/`%bc1`/`%bc2` and 106 batch stats out. Parameter ORDER comes
    from `r50SigList`, the same single source the signature and the return list use, so the
    arity/order contract cannot drift within this file — and it is written to agree with
    `VerifiedSpec.bottleneckStageSpec`, which is what the DRIVER walks.

    ⚠ The block sequence is `[3,4,6,3]` with block 0 of every stage projecting. Stage 1's projects
    at **stride 1** (`bnkProjFwdB`); stages 2/3/4 project strided. -/
def resnet50TrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (opt : R34Opt := .adamw) (slug : String := "resnet50in")
    -- ⚠ TRAILING, per §2m: a parameter inserted mid-list captures an existing positional argument.
    -- `bce` swaps the LOSS (and therefore the cotangent) for BCE-with-logits — RSB-A2/A3's, and the
    -- reason the recipe's lr is what it is. It also names the artifact: `r34AdamVariant` appends
    -- the `bce` marker, so the entry point, the path and `LEAN_MLIR_VARIANT` stay ONE string
    -- derived from ONE flag.
    -- ⚠⚠ `vSuffix` IS GONE (`a3_paper_fidelity.md` §3.3, closed 2026-08-14). It was a `String` the
    -- caller had to spell `"bce"` by hand alongside this `Bool`, i.e. two writers for one fact on
    -- the artifact the 77.43% run depends on — the name and the loss could disagree and nothing
    -- would notice. `r34AdamVariant` now derives the marker from this flag; read its `bce` note for
    -- the measurement that licensed the removal (12 sites, 12 suffixes, no other value ever).
    (bce : Bool := false)
    -- ▶▶ **A NON-DEFAULT WEIGHT DECAY** (`planning/verified_optimizer_parity.md` §3, the `wdStr`
    -- item): empty means this optimizer's own value — `optWdDefault`, 1e-4 for the AdamW family and
    -- 0.02 for LAMB's — so every committed artifact keeps its bytes. RSB-**A1** wants 0.01 where A3
    -- wants 0.02, and this is what makes that a re-render rather than a new op.
    -- ⚠⚠ It is a BAKED `stablehlo.constant`, not a runtime operand: unlike `%lr`, the decay is not
    -- schedulable and changing it means re-rendering. ⚠ It reaches `r34AdamVariant`, because two
    -- renders differing only in a baked constant must not share a path — see `wdVariantMark`.
    (wdStr : String := "")
    -- ⭐⭐ RESOLUTION, and it is parameterised by the FINAL feature size rather than by the input.
    -- `q = 7` is 224 (every committed artifact, byte for byte); `q = 5` is RSB-A3's **160**.
    --
    -- ⚠⚠ **Deriving UPWARD by doubling is not a style choice, it is what makes the dependent types
    -- work.** Every size relation this graph needs is of the form "the input to a stride-2 op is
    -- twice its output", and `Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w)` states that IN THE TYPE. Written
    -- downward (`imgSize/2`, `imgSize/4`, …) Lean would have to prove `2*(imgSize/4) = imgSize/2`,
    -- which is false for odd inputs and not definitional for a variable. Written as `q4 := 2*q5`,
    -- `q3 := 2*q4`, … every such equation holds by zeta-reduction and nothing needs a proof.
    (q : Nat := 7)
    -- ▶ **`wdExcludeNormBias` — timm's `no_weight_decay` skip-list** (`a3_paper_fidelity.md` §2.1),
    -- the largest recipe delta the RSB-A3 run shipped with and a REQUIREMENT of RSB-A2. It excludes
    -- every 1-D parameter — BN γ, BN β, every bias — from decay, decaying only ≥2-D weights.
    --
    -- ⚠⚠ THE A3 RUN DID NOT HAVE IT. Its reference (`resnet50ImagenetConfigRSBFaithful`) sets
    -- `wdExcludeNormBias := true`; the live artifact has zero `%wdz`. So 77.43% was reached while
    -- decaying BN γ/β at wd = 0.02. Decay on a pre-BN conv weight is renormalised away by BN and
    -- acts only as an effective-LR control; decay on γ is not, because γ scales the layer's output
    -- directly — and the effect concentrates at low LR, i.e. in the cosine endgame.
    --
    -- ⭐ It needs NO new op and NO driver change: the decay is an operand NAME on every optimizer
    -- (`adamWParamF`, `lambDirF`, `momVNextF`), so excluding a parameter binds that name to a zero
    -- constant. Same arity, same types, same regions — exactly how ConvNeXt and ViT do it.
    -- ⚠ TRAILING, and it must reach `r34AdamVariant` too: this net DERIVES its entry name from the
    -- variant, so a flag that reaches the renderer but not the name produces an artifact whose
    -- declared entry disagrees with its own path.
    (wdExclude : Bool := false)
    -- ▶▶ **`gradClip` — timm's `Lamb.max_grad_norm`, D1** (`planning/recipe_fidelity_diffs.md`,
    -- `planning/verified_optimizer_parity.md` §2). `timm.optim.lamb.Lamb.__init__` DEFAULTS it to
    -- `1.0` and clips the global gradient norm inside the optimizer on every step, so a LAMB render
    -- without it is not the optimizer the recipe names — read from source, not assumed:
    --
    -- ```python
    -- clip_global_norm = (global_norm / max_grad_norm).clamp_(min=1.0)   # _get_clip_grad_norm
    -- grad.div_(clip_global_norm)                                        # step()
    -- ```
    --
    -- which is `g · min(1, C/‖g‖)` — algebraically `clipScaleF`, and the reference side's
    -- `g * jnp.minimum(1.0, C / (gn + 1e-6))`.
    --
    -- ⚠⚠ **THE NORM IS GLOBAL — ONE SCALAR ACROSS ALL 161 PARAMETERS.** That is the entire semantic
    -- content, and it is what a per-parameter check cannot see: a per-parameter clip compiles,
    -- renders, trains and descends (`Proofs.clipFactor_shared` against `Proofs.lambScale_not_shared`,
    -- which is LAMB's genuinely per-tensor ratio sitting three ops away in the same graph).
    --
    -- ⚠ TRAILING, and it reaches `r34AdamVariant` — this net DERIVES its entry name from the
    -- variant, and `cnxAdamVariant` records ConvNeXt shipping that exact defect twice.
    (gradClip : Bool := false)
    -- ⚠ The threshold is a BAKED constant, like `%wd` and unlike `%lr`: it lives in the artifact, so
    -- changing it is a re-render. Fine for timm's fixed `1.0`; worth knowing before anyone sweeps it.
    -- ⚠ A `Float` where `ConvNeXtRender`'s peer is a `String`, because this one is ARITHMETIC —
    -- under accumulation the baked threshold is `k·C` (`clipNormStr`, and read its note before
    -- touching this: the reference clips the MEAN accumulated gradient).
    (clipNorm : Float := 1.0)
    -- ⭐⭐ **bf16**, LAST, per §2m and for the same reason `clip` and `wx` are trailing.
    -- Every conv in this net — stem, the three bottleneck convs, the projection, both dgrads,
    -- every wgrad — becomes its bf16 twin: bf16 operands, a **bf16-TYPED** convolution result,
    -- then a convert back to f32. BN, the residual adds, the loss, the optimizer and the master
    -- weights all stay f32, which is what `jax/MainResnetImagenet.lean` does.
    -- ⚠⚠ It MUST reach `r34AdamVariant` below and not merely the block renderers: the entry
    -- NAME is derived from the variant, and a flag that reaches the emission but not the name
    -- writes `…bf16_train_step.mlir` declaring `@…_train_step` inside — the driver then refuses
    -- at load with an entry mismatch. ConvNeXt shipped that twice; R34's bf16 made it three.
    (bf16 : Bool := false)
    -- ▶▶▶ **`ema` — MODEL EMA, THE FIFTH REGION** (`verified_side_quest_counterparts.md` §6a,
    -- 2026-08-27). The blob becomes `[θ|m|v|G|E]` and the scalar tail `lr,bc₁,bc₂,aup,akeep,emad,oemad`.
    --
    -- ⚠⚠ **THIS IS THE PARAMETER RSB-A2 AND RSB-A1 COULD NOT BE RENDERED WITHOUT**, and the reason
    -- is worth keeping: `resnet50ImagenetConfigA2Accum` sets `useEMA := true` AND
    -- `gradAccumSteps := 4`, and until this landed the shadow and the accumulator were the SAME
    -- fourth region — `VerifiedTrain.lean` threw on the pairing rather than letting one win. A3 met
    -- neither obstacle because A3's own recipe sets `useEMA := false`, which is exactly why the
    -- limitation was invisible from A3's success.
    --
    -- ⭐ It costs ONE op per parameter and no new `SHlo` constructor — see `optOne`'s `ema` note.
    -- ⚠ `E` comes AFTER `G`, never instead of it: at `acc` alone the accumulator is still region 3
    -- and at `ema` alone the shadow is still region 3, so no previously-written blob moves.
    -- ⚠ It MUST reach `r34AdamVariant` — the `wx`/`clip`/`bf16` rule three parameters up, and here
    -- the stakes are higher than an entry mismatch: the marker is what tells the DRIVER to pack
    -- five regions, so a flag that stopped at the emission would produce a five-region graph the
    -- driver feeds four regions to.
    (ema : Bool := false)
    -- ▶▶ **`sd` — STOCHASTIC DEPTH, `dropPath := 0.05`** — RSB-A2/A1's remaining regulariser
    -- (`verified_side_quest_counterparts.md` §4a's second ⛔, 2026-08-27). Sixteen `tensor<Bxf32>`
    -- mask inputs, one per bottleneck, drawn on the HOST beside the augmentation seed.
    --
    -- ⚠⚠ **THE SITE IS ON THE RESIDUAL BRANCH AND THE OBVIOUS GATE CANNOT SEE THAT** — at an
    -- all-ones mask a site on the block OUTPUT is bit-identical. `bnkIdFwdB`'s docstring carries
    -- the argument and `scripts/misplace_drop_sites.py` is the control.
    -- ⚠ The BACKWARD is where it actually bites: the dropped cotangent feeds the branch and the
    -- UNDROPPED one feeds the skip. See `bnkIdBackGradB`.
    -- ⚠ It reaches `r34AdamVariant` — the `wx`/`clip`/`bf16` rule, and here the marker also tells
    -- the DRIVER to append 16 mask slots, so a flag that stopped at the emission would produce a
    -- graph the driver feeds none.
    (sd : Bool := false) : String :=
  let optLabel : String := match opt with
    | .adamw          => "AdamW"
    | .heavyBall      => "heavy-ball momentum + coupled L2"
    | .lamb           => "LAMB (per-tensor trust ratio)"
    | .adamwAccum k   => s!"AdamW over {k} ACCUMULATED micro-batches"
    | .lambAccum k    => s!"LAMB (per-tensor trust ratio) over {k} ACCUMULATED micro-batches"
  -- ⚠ TYPE-based, so unlike the driver's string predicate this one could not silently miss the new
  -- arm — adding a constructor without extending it is a non-exhaustive match, i.e. a build error.
  -- That asymmetry is exactly why the driver needs `TestVariantPredicates` and this line does not.
  let accOn := match opt with | .adamwAccum _ => true | .lambAccum _ => true | _ => false
  let go : StateM Nat String := do
    -- ═══ forward: THE SHARED CHAIN, not a second copy (see `r50FwdChainB`) ═══
    let fw ← r50FwdChainB B nClasses epsStr q bf16 sd
    let q5 := q
    let q4 := 2 * q5
    let q3 := 2 * q4
    let q2 := 2 * q3
    let q1 := 2 * q2
    let zx    : Vec (B*(3*(2*q1)*(2*q1))) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*q1*q1)) := fun _ => 0
    let z112b : Vec (B*(64*(q1*q1))) := fun _ => 0
    let z56   : Vec (B*(64*q2*q2)) := fun _ => 0
    let zL    : Vec (B*(2048*q5*q5)) := fun _ => 0
    let z2048 : Vec (B*2048) := fun _ => 0
    let zWd   : Mat 2048 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp  : Vec (B*nClasses) := fun _ => 0
    -- Placeholder rounding, as the `z*` zeros are placeholders — see `bnkIdFwdB`.
    let zrnd : ℝ → ℝ := fun r => r
    let nStc := fw.stc; let nStn := fw.stn; let nStr := fw.str; let nStp := fw.stp
    let nGap := fw.gap; let nLog := fw.logits
    let f1 := fw.b[0]!;  let f2 := fw.b[1]!;  let f3 := fw.b[2]!;  let f4 := fw.b[3]!
    let f5 := fw.b[4]!;  let f6 := fw.b[5]!;  let f7 := fw.b[6]!;  let f8 := fw.b[7]!
    let f9 := fw.b[8]!;  let f10 := fw.b[9]!; let f11 := fw.b[10]!; let f12 := fw.b[11]!
    let f13 := fw.b[12]!; let f14 := fw.b[13]!; let f15 := fw.b[14]!; let f16 := fw.b[15]!
    -- ═══ the loss cotangent ═══
    --
    -- ⭐⭐ **BCE-with-logits is the SAME SHAPE with one op swapped, and THREE ops instead of five.**
    -- Softmax-CE's cotangent is `(softmax(z) − t)/B` with `t` the smoothed target; BCE's is
    -- `(σ(z) − t)/(B·K)`. So `softmaxRow → sigmoidB`, and the two label-smoothing nodes disappear.
    --
    -- ⚠⚠ **THE DIVISOR IS `B·K`, NOT `B`, AND THAT IS NOT A DETAIL.** timm's `BinaryCrossEntropy`
    -- is `reduction='mean'` over B×C, not the mean of the per-example SUM over classes. The
    -- reference's own comment: *"that would be NC× larger and need an NC× smaller lr — RSB-A2's
    -- lr 5e-3 is tuned to this form."* At K = 1000 the two differ by 1000× on the effective step.
    --
    -- ⚠ **NO LABEL SMOOTHING on the BCE path**, and that is A3's recipe rather than an omission:
    -- timm's a3 arg string is `…-m0.1-sd0.0-d0.0-ls0.0-100`, i.e. `ls0.0`. The soft targets come
    -- from mixup/cutmix through `%onehot` (wire v2), not from a smoothing constant in the loss.
    let (cSm,  nSm)  ← if bce
      -- ⚠ `zNCp : Vec (B*nClasses)`, not `zNCb : Vec (B*(1*nClasses))`. The two indices are equal
      -- but NOT definitionally so for a variable `nClasses` (`Nat.mul` recurses on its second
      -- argument, so `1 * n` does not reduce), and `sigmoidB`'s `{N n}` unify against the bare
      -- product. The `if` still typechecks because both branches are `StateM Nat (String × String)`
      -- — the SHlo index never escapes the branch.
      then pretty B (.sigmoidB (N := B) (n := nClasses) (.operand nLog zNCp))
      else pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← if bce then pure ("", nD0)
      else pretty B (.scaleB "0.100000" 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← if bce then pure ("", nD0)
      else pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← if bce then pure ("", nD0)
      else pretty B (.shiftB s!"-{alphaOverK nClasses}" 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB (if bce then s!"{B * nClasses}.0" else s!"{B}.0") 0
                                  (.operand nD2 zNCb))
    -- ═══ head backward + dense grads ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) (a := 2048) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWd,  nWd)  ← pretty B (.denseWeightGradB (c := nClasses) nGap z2048 (.operand nDy zNCp))
    let (cbd,  nbd)  ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 2048) (h := q5) (w := q5) (.operand nDgi z2048))
    -- ═══ 16 block backwards, in reverse ═══
    let b16 ← bnkIdBackGradB      B      512 2048  q5 epsStr "s4b2" f16 nDgp bf16 (dpAt sd 15)
    let b15 ← bnkIdBackGradB      B      512 2048  q5 epsStr "s4b1" f15 b16.dx bf16 (dpAt sd 14)
    let b14 ← bnkStridedBackGradB B 1024 512 2048  q5 epsStr "s4b0" f14 b15.dx bf16 (dpAt sd 13)
    let b13 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b5" f13 b14.dx bf16 (dpAt sd 12)
    let b12 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b4" f12 b13.dx bf16 (dpAt sd 11)
    let b11 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b3" f11 b12.dx bf16 (dpAt sd 10)
    let b10 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b2" f10 b11.dx bf16 (dpAt sd 9)
    let b9  ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b1" f9  b10.dx bf16 (dpAt sd 8)
    let b8  ← bnkStridedBackGradB B  512 256 1024 q4 epsStr "s3b0" f8  b9.dx bf16 (dpAt sd 7)
    let b7  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b3" f7  b8.dx bf16 (dpAt sd 6)
    let b6  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b2" f6  b7.dx bf16 (dpAt sd 5)
    let b5  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b1" f5  b6.dx bf16 (dpAt sd 4)
    let b4  ← bnkStridedBackGradB B  256 128  512 q3 epsStr "s2b0" f4  b5.dx bf16 (dpAt sd 3)
    let b3  ← bnkIdBackGradB      B       64  256 q2 epsStr "s1b2" f3  b4.dx bf16 (dpAt sd 2)
    let b2  ← bnkIdBackGradB      B       64  256 q2 epsStr "s1b1" f2  b3.dx bf16 (dpAt sd 1)
    let b1  ← bnkProjBackGradB    B   64  64  256 q2 epsStr "s1b0" f1  b2.dx bf16 (dpAt sd 0)
    -- ═══ stem backward ═══
    let (cDmp, nDmp) ← pretty B (.maxPool3s2BackB (N := B) (c := 64) (h := q2) (w := q2) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 64) (h := q1) (w := q1) "%sg" nStc epsStr 0 z64 z112b (.operand nDsr z112b))
    let (csW, nsW) ← pretty B (if bf16 then .convStridedWeightGradBBf16 zrnd "%x" z64 zx zSk (.operand nDsn z112) else .convStridedWeightGradB "%x" z64 zx zSk (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaGradB nStc epsStr 0 z112b (.operand nDsr z112b))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 64) (h := q1) (w := q1) (.operand nDsr z112b))
    -- ═══ BN running statistics, from each BN layer's INPUT ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × String × String) := do
      let zbv : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zbv))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zbv))
      pure (cM ++ cV, nM, nV)
    -- identity + stride-1-projection blocks keep every BN at `hh`.
    let idStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat mid hh f.c1
      let (c2, m2, v2) ← bnStat mid hh f.c2
      let (c3, m3, v3) ← bnStat oc hh f.c3
      pure (c1 ++ c2 ++ c3, [m1, v1, m2, v2, m3, v3])
    let projStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (cb, ns) ← idStats mid oc hh f
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (cb ++ cp, ns ++ [mp, vp])
    -- ⚠ the STRIDED block's bn1 lives at `2*hh`; only bn2/bn3/bnp are at `hh`.
    let strStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat mid (2*hh) f.c1
      let (c2, m2, v2) ← bnStat mid hh f.c2
      let (c3, m3, v3) ← bnStat oc hh f.c3
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (c1 ++ c2 ++ c3 ++ cp, [m1, v1, m2, v2, m3, v3, mp, vp])
    let (cSt0, st0) ← bnStat 64 q1 nStc
    let (cSt1,  st1)  ← projStats  64  256 q2 f1
    let (cSt2,  st2)  ← idStats    64  256 q2 f2
    let (cSt3,  st3)  ← idStats    64  256 q2 f3
    let (cSt4,  st4)  ← strStats  128  512 q3 f4
    let (cSt5,  st5)  ← idStats   128  512 q3 f5
    let (cSt6,  st6)  ← idStats   128  512 q3 f6
    let (cSt7,  st7)  ← idStats   128  512 q3 f7
    let (cSt8,  st8)  ← strStats  256 1024 q4 f8
    let (cSt9,  st9)  ← idStats   256 1024 q4 f9
    let (cSt10, st10) ← idStats   256 1024 q4 f10
    let (cSt11, st11) ← idStats   256 1024 q4 f11
    let (cSt12, st12) ← idStats   256 1024 q4 f12
    let (cSt13, st13) ← idStats   256 1024 q4 f13
    let (cSt14, st14) ← strStats  512 2048 q5 f14
    let (cSt15, st15) ← idStats   512 2048 q5 f15
    let (cSt16, st16) ← idStats   512 2048 q5 f16
    -- ═══ the 161 parameter gradients in func-arg order ═══
    let stemPs : List PGrad := [⟨"sW", nsW, [64,3,7,7]⟩, ⟨"sg", nsg, [64]⟩, ⟨"sbt", nst, [64]⟩]
    let headPs : List PGrad := [⟨"Wd", nWd, [2048, nClasses]⟩, ⟨"bd", nbd, [nClasses]⟩]
    let allPs : List PGrad := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++
      b9.ps ++ b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ headPs
    -- ═══ the optimizer stage: D1's hoisted global-norm clip, then one proven triple per
    -- parameter. ⚠ `optAllParams` is IMPORTED from `ResNet34RenderB`, not written here — read its
    -- docstring for the two orderings the clip has to respect (after the all_reduce AND after the
    -- accumulation), and for why it is a function at all: `tests/TestOptStepFixtures.lean` drives
    -- the SAME call for `planning/verified_optimizer_parity.md` §5's one-step update gate, so the
    -- gate exercises this emission rather than a second copy of it. ═══
    let (adamCode, thetaN, mNames, vNames, aNames, eNames) ←
      optAllParams opt B replicas allPs wdExclude gradClip clipNorm ema
    let statCode := cSt0 ++ cSt1 ++ cSt2 ++ cSt3 ++ cSt4 ++ cSt5 ++ cSt6 ++ cSt7 ++ cSt8 ++
      cSt9 ++ cSt10 ++ cSt11 ++ cSt12 ++ cSt13 ++ cSt14 ++ cSt15 ++ cSt16
    let statNames := st0.1 :: st0.2 :: (st1 ++ st2 ++ st3 ++ st4 ++ st5 ++ st6 ++ st7 ++ st8 ++
      st9 ++ st10 ++ st11 ++ st12 ++ st13 ++ st14 ++ st15 ++ st16)
    -- `%loss` is REPORT-ONLY (logging), on no gradient path, and NOT `pretty` of an AST node — the
    -- same §5 carve-out R34's takes. SMOOTHED CE, matching the cotangent's soft target; a first cut
    -- on R34 computed plain CE here and only a numeric tie caught it.
    -- ⭐ **BCE-with-logits, in the stable form `softplus(z) − t·z`.** Expanding
    -- `t·softplus(−z) + (1−t)·softplus(z)` with the identity `softplus(−x) = softplus(x) − x`
    -- collapses the reference's two softplus calls to ONE, and the result never exponentiates a
    -- positive number: `softplus(z) = max(z,0) + log(1 + exp(−|z|))`. Report-only, like the CE
    -- loss beside it — the §5 carve-out — but the arithmetic still has to be right, because the
    -- epoch curve is how the run is judged against the reference (§2b's `%loss` bug).
    let lossCodeBce :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      "    // BCE-with-logits, mean over B x K: softplus(z) - t*z, softplus stable as\n" ++
      "    // max(z,0) + log(1 + exp(-|z|)). ⚠ mean over B*K, NOT mean of the per-example sum.\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %lzb = stablehlo.constant dense<0.0> : {ty [B, nClasses]}\n" ++
      s!"    %labs = stablehlo.abs {nLog} : {ty [B, nClasses]}\n" ++
      s!"    %lneg = stablehlo.negate %labs : {ty [B, nClasses]}\n" ++
      s!"    %lexp = stablehlo.exponential %lneg : {ty [B, nClasses]}\n" ++
      s!"    %lone = stablehlo.constant dense<1.0> : {ty [B, nClasses]}\n" ++
      s!"    %l1pe = stablehlo.add %lone, %lexp : {ty [B, nClasses]}\n" ++
      s!"    %llg = stablehlo.log %l1pe : {ty [B, nClasses]}\n" ++
      s!"    %lmax = stablehlo.maximum {nLog}, %lzb : {ty [B, nClasses]}\n" ++
      s!"    %lsp = stablehlo.add %lmax, %llg : {ty [B, nClasses]}\n" ++
      s!"    %ltz = stablehlo.multiply %onehot, {nLog} : {ty [B, nClasses]}\n" ++
      s!"    %lbce = stablehlo.subtract %lsp, %ltz : {ty [B, nClasses]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%lbce init: %lz) applies stablehlo.add across dimensions = [0, 1] : ({ty [B, nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B * nClasses}.0> : tensor<f32>\n" ++
      s!"    %loss = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n"
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %lomac = stablehlo.constant dense<0.900000> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses}> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let body := fw.code ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWd ++ cbd ++ cDgp ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := (r50StatSigList.map (·.2))
    -- ⚠ THE PACKED LAYOUT IS `[θ|m|v|(G)| scalars | BN stats]`, and the driver indexes it off
    -- `nRegions`/`nScalars` rather than literals — so the accumulator has to sit where the EMA
    -- shadow sits (fourth region) and the two accum scalars where EMA's two sit (slots 4 and 5).
    -- `%aup`/`%akeep` ride out as passthroughs so `#out = #in − 2` holds exactly as for `.adamw`.
    let accScalars := if accOn then ["%aup", "%akeep"] else []
    -- ⭐ The EMA pair rides out as a passthrough exactly as `%aup`/`%akeep` do, so `#out = #in − 2`
    -- holds at every combination of the two axes rather than only at one of them.
    -- ⚠⚠ **AND THE DECAY IS A RUNTIME SCALAR, WHICH IS WHAT MAKES THE ACCUMULATE PHASE EXPRESSIBLE
    -- WITHOUT A SECOND GRAPH.** The reference EMAs once per OPTIMIZER step, not once per
    -- micro-batch (`ema_update` follows the `train_step` call, and JAX's accumulation is INSIDE
    -- that call), so on an accumulate micro-batch this graph must leave the shadow exactly alone.
    -- The driver supplies `%emad = 1, %oemad = 0` there, which is `e' = 1·e + 0·θ' = e` — the same
    -- arithmetic-not-branching trick `%aup` plays on the moments. Baking the decay would have
    -- forced either a second artifact or an EMA running k times per optimizer step.
    let emaScalars := if ema then ["%emad", "%oemad"] else []
    let accScalarTys := (accScalars ++ emaScalars).map (fun _ => "tensor<f32>")
    let retVals := thetaN ++ mNames ++ vNames ++ aNames ++ eNames ++
      ["%loss", "%bc1", "%bc2"] ++ accScalars ++ emaScalars ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++ (if accOn then pTypes else []) ++
      (if ema then pTypes else []) ++
      ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ accScalarTys ++ statTypes
    pure <|
      (if replicas ≤ 1 then
        s!"    // ── ResNet-50 bottleneck batch-BN {optLabel} train step: every line is pretty(verified AST node) ──\n"
       else
        s!"    // ── ResNet-50 bottleneck batch-BN {optLabel} train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5).\n") ++
      zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ body ++ optConstsB opt wdStr ++
      wdzConst wdExclude ++ clipZeroConst gradClip ++ adamCode ++
      (if bce then lossCodeBce else lossCode) ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := r50SigList nClasses
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  -- The accumulator region `G`, named `<p>a`, present only under `.adamwAccum`. At every other
  -- optimizer this is the empty string, so the three committed R50 artifacts re-render byte for byte.
  let aSig := if accOn then ", " ++ String.intercalate ", "
                              (sigList.map (fun (n, t) => s!"{n}a: {t}")) else ""
  -- The shadow region `E`, named `<p>e`, present only under `ema`. ⚠ It follows `aSig` and never
  -- precedes it — the `[θ|m|v|G|E]` order the driver packs, and the order that leaves both
  -- single-axis layouts at the index they already occupy.
  -- ⚠ `{n}ema`, not `{n}e` — see `optOne`'s note: `%sg` + `e` is `%sge`, the maxpool backward's
  -- own block-local name, and the artifact does not parse. The ARGUMENT and the produced value have
  -- to move together or the region has no input.
  let eSig := if ema then ", " ++ String.intercalate ", "
                              (sigList.map (fun (n, t) => s!"{n}ema: {t}")) else ""
  let accSSig := (if accOn then ", %aup: tensor<f32>, %akeep: tensor<f32>" else "") ++
                 (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "")
  let statSig := String.intercalate ", " (r50StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  -- ⚠ The 16 drop masks go AFTER the BN stats and BEFORE `%onehot`, which is where the driver
  -- packs them (`dropSlots` trails `runningBnStats` in `trainAdamSched`'s `pbuf`). The labels ride
  -- separately, so `%onehot` stays last.
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    aSig ++ eSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++ accSSig ++ ", " ++ statSig ++
    r50DropSig B sd ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let accTy := (if accOn then ["tensor<f32>", "tensor<f32>"] else []) ++
               (if ema then ["tensor<f32>", "tensor<f32>"] else [])
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ (if accOn then pTy else []) ++ (if ema then pTy else []) ++
     ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ accTy ++ (r50StatSigList.map (·.2)))
  let inner : String := go.run' 0
  -- ⚠ Same `{slug}_{variant}_train_step` convention the shim checks; `r34AdamVariant` is reused as
  -- the single source for the variant name so R50's artifact names cannot drift from R34's rule.
  let fname := s!"{slug}_{r34AdamVariant B replicas opt wdExclude gradClip bce wdStr bf16 ema sd}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The forward chain — `@resnet50in_fwd` and `@resnet50in_fwd_eval`
-- ════════════════════════════════════════════════════════════════

/-! ⚠ These live here rather than in a `ResNet50Render.lean` peer of R34's split. That split is
historical — R34 has a per-example-BN net AND a batch-BN net, and they are different functions, so
they get different files. R50 has only the batch-BN net, so one file keeps `r50SigList` and every
consumer of it together.

⚠⚠ **The train and eval forwards MUST be the same chain with one switch**, which is why `bnSite`
(shared with R34, now public) takes the mode rather than each render spelling its BN. §2g's
`mobilenetv2_fwd` defect was exactly this drifting: a net trained with one pool and scored with
another, logits rel 1.86, silent. -/

/-- Identity bottleneck forward, per-example vocabulary. -/
private def bnkIdFwdV (B mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut : Vec (oc*hh*ww) := fun _ => 0
  let zMid : Vec (mid*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := oc) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zOut))
  let (cN1, nN1) ← bnSite B mid hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMid))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand xName zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cA ++ cO, nO)

/-- ⭐ Stride-1 projection bottleneck forward — stage 1 block 0. The projection is `.flatConvF`,
    NOT `.flatConvStridedF`. -/
private def bnkProjFwdV (B cin mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn  : Vec (cin*hh*ww) := fun _ => 0
  let zOut : Vec (oc*hh*ww) := fun _ => 0
  let zMid : Vec (mid*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := cin) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zIn))
  let (cN1, nN1) ← bnSite B mid hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMid))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cCp, nCp) ← pretty B (.flatConvF (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo (.operand xName zIn))
  let (cNp, nNp) ← bnSite B oc hh mode epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO, nO)

/-- Strided projection bottleneck forward. ⚠ `conv1`/`bn1`/`relu1` at `2hh`; only `conv2` strides. -/
private def bnkStridedFwdV (B cin mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm     : Vec mid := fun _ => 0
  let zo     : Vec oc := fun _ => 0
  let zk1    : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2    : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3    : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp    : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn    : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zMidIn : Vec (mid*(2*hh)*(2*ww)) := fun _ => 0
  let zMid   : Vec (mid*hh*ww) := fun _ => 0
  let zOut   : Vec (oc*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zIn))
  let (cN1, nN1) ← bnSite B mid (2*hh) mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMidIn))
  let (cC2, nC2) ← pretty B (.flatConvStridedF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMidIn))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cCp, nCp) ← pretty B (.flatConvStridedF (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo (.operand xName zIn))
  let (cNp, nNp) ← bnSite B oc hh mode epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO, nO)

set_option maxRecDepth 4000000 in
/-- The full R50 forward: stem → `[3,4,6,3]` bottlenecks → GAP(7×7) → dense(2048→nClasses).
    `mode` picks batch statistics (`.train`) or frozen running stats (`.eval`); ONE chain, so the
    two renders cannot describe different nets. -/
private def r50FwdChain (B nClasses : Nat) (mode : R34Bn) (epsStr : String)
    -- ⚠ TRAILING, and the SAME ladder the train step uses (`q = 7` is 224, `q = 5` is 160). It has
    -- to be the same derivation, not merely the same numbers: the §2g prefix audit only means
    -- anything if the forward and the train step are one chain at one resolution.
    (q : Nat := 7) : StateM Nat (String × String) := do
  let q5 := q; let q4 := 2 * q5; let q3 := 2 * q4; let q2 := 2 * q3; let q1 := 2 * q2
  let zx   : Vec (3*(2*q1)*(2*q1)) := fun _ => 0
  let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64  : Vec 64 := fun _ => 0
  let z112 : Vec (64*q1*q1) := fun _ => 0
  let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 64) (h := q1) (w := q1) "%sW" (zb 64) zSk z64 (.operand "%x" zx))
  let (cStn, nStn) ← bnSite B 64 q1 mode epsStr "%sg" "%sbt" "stn" nStc
  let (cStr, nStr) ← pretty B (.reluF (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.maxPool3s2F (c := 64) (h := q2) (w := q2) (.operand nStr z112))
  let (c1, n1)   ← bnkProjFwdV    B   64  64  256 q2 mode epsStr "s1b0" nStp
  let (c2, n2)   ← bnkIdFwdV      B       64  256 q2 mode epsStr "s1b1" n1
  let (c3, n3)   ← bnkIdFwdV      B       64  256 q2 mode epsStr "s1b2" n2
  let (c4, n4)   ← bnkStridedFwdV B  256 128  512 q3 mode epsStr "s2b0" n3
  let (c5, n5)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b1" n4
  let (c6, n6)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b2" n5
  let (c7, n7)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b3" n6
  let (c8, n8)   ← bnkStridedFwdV B  512 256 1024 q4 mode epsStr "s3b0" n7
  let (c9, n9)   ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b1" n8
  let (c10, n10) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b2" n9
  let (c11, n11) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b3" n10
  let (c12, n12) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b4" n11
  let (c13, n13) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b5" n12
  let (c14, n14) ← bnkStridedFwdV B 1024 512 2048  q5 mode epsStr "s4b0" n13
  let (c15, n15) ← bnkIdFwdV      B      512 2048  q5 mode epsStr "s4b1" n14
  let (c16, n16) ← bnkIdFwdV      B      512 2048  q5 mode epsStr "s4b2" n15
  let zL    : Vec (2048*q5*q5) := fun _ => 0
  let z2048 : Vec 2048 := fun _ => 0
  let zWd   : Mat 2048 nClasses := fun _ _ => 0
  let zNC   : Vec nClasses := fun _ => 0
  let (cGap, nGap) ← pretty B (.gapF (c := 2048) (h := q5) (w := q5) (.operand n16 zL))
  let (cLog, nLog) ← pretty B (denseF "%Wd" "%bd" zWd zNC (.operand nGap z2048))
  pure (cStc ++ cStn ++ cStr ++ cStp ++
        c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++
        c9 ++ c10 ++ c11 ++ c12 ++ c13 ++ c14 ++ c15 ++ c16 ++ cGap ++ cLog, nLog)

set_option maxRecDepth 4000000 in
/-- **`@resnet50in_fwd`** — 162 inputs (`%x` + 161 params), logits `[B, nClasses]`. Batch-statistic
    BN, i.e. the same forward the train step differentiates.

    ⚠⚠ **That sentence was FALSE until 2026-08-10 and is now enforced.** This built its forward from
    `r50FwdChain .train`, which reaches R34's shared `bnSite` and therefore `bnPerChannelF` — reduce
    `[2,3]`, divisor `H·W`, PER-EXAMPLE — while `resnet50TrainStepFaithfulB` is `bnBatchF`, reduce
    `[0,2,3]`, divisor `B·H·W`. Two different functions under one net's name (§3d(b)), and
    `regen_verified_mlir.sh check` could not see it because it only ever paired a forward with the
    SGD train step and R50 has none. It now renders from `r50FwdChainB` — literally the traversal
    the train step differentiates — so `check_adam_prefix` holds it as a byte prefix.

    ⚠ `r50FwdChain`'s `.train` branch is now UNUSED by R50 and must stay that way; it survives only
    because `.eval` shares the function. Rendering a forward from it reopens the split. -/
def resnet50FwdFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet50in") (q : Nat := 7) (vSuffix : String := "") : String :=
  let sigList := r50SigList nClasses
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let fwr := (r50FwdChainB B nClasses epsStr q).run' 0
  let (code, logits) := (fwr.code, fwr.logits)
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-50 forward: every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ code ++
  s!"    return {logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 4000000 in
/-- **`@resnet50in_fwd_eval`** — the inference forward, every BN site reading frozen running stats.
    161 params + 106 stat inputs + `%x` = **268 inputs**. This is what the driver scores through,
    so its BN order must match `r50StatSigList`, which it does by sharing the chain. -/
def resnet50FwdEvalFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet50in") (q : Nat := 7) (vSuffix : String := "") : String :=
  let sigList := r50SigList nClasses ++ r50StatSigList
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let (code, logits) := (r50FwdChain B nClasses .eval epsStr q).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-50 eval forward (running-stats BN): every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ code ++
  s!"    return {logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

end Proofs.StableHLO

-- ⚠ Slug `resnet50in`, distinct from any 10-class R50, for the reason §2k records for `resnet34in`:
-- the forwards carry no variant in their path, so a shared slug would silently overwrite a
-- different-arity artifact.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adam64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50in")

-- The 4-GPU data-parallel peer at `B := 64` PER REPLICA ⇒ global batch 256, matching R34's
-- ImageNet recipe and the reference's. One `#eval`: `optOne` already takes `replicas`.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adamdp64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.adamw "resnet50in")

-- ⭐ THE 2018 RECIPE, so R50 and R34 can be compared with ONLY the network swapped.
-- Heavy-ball momentum + coupled L2 at global batch 256, which is exactly
-- `resnet34in_mom256`'s configuration (`ResNet34RenderB.lean`) with the bottleneck backbone
-- substituted. Everything the two runs differ by is the architecture, which is what makes the
-- 2018-vs-A3 comparison in the book a recipe comparison rather than a confounded one.
--
-- ⚠ `mom256` is a SINGLE-device render at batch 256, matching R34's. The data-parallel peer is
-- `momdp64` (4 replicas x 64) if a 4-GPU run is wanted; both are the same global batch.
#eval IO.FS.writeFile "verified_mlir/resnet50in_mom256_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 256 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.heavyBall "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_momdp64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.heavyBall "resnet50in")

-- ⭐⭐ **The bf16 peer of the render above** — `momdp64bf16`, the same graph with every one of its
-- 53 convolutions replaced by its bf16 twin: bf16 operands, a **bf16-TYPED** convolution result,
-- then a convert back to f32. Deliberately the same config R34's bf16 arm uses (heavy-ball, 4×64,
-- global batch 256), so the two nets differ by the ARCHITECTURE and nothing else and the R34
-- 1.41× is a comparable number rather than a differently-configured one.
--
-- ⚠ The bf16-typed RESULT is load-bearing and is not cosmetic. A convolution with bf16 operands
-- and an f32 result has its converts folded away by XLA under excess precision — cuDNN then gets
-- f32 parameters and the graph runs entirely in fp32 while still *reading* as mixed precision.
-- Measured, not feared; `BatchableOp.convBf16` carries the note. ▶ Check it with
-- `scripts/bf16_gate2.py` on the OPTIMIZED HLO's operand SSA names, never by grepping the op line,
-- which shows only the result type.
--
-- ⚠ R50 exercises a shape R34 never did: the **stride-1 1×1** convolution, which is 34 of these 53
-- sites (`convBf16` at `kH = kW = 1`, so `pad = 0`). R34's only 1×1s are its strided projections.
-- That is the one genuinely new thing about this render and it is what gate 2 has to confirm.
#eval IO.FS.writeFile "verified_mlir/resnet50in_momdp64bf16_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.heavyBall "resnet50in" (bf16 := true))

-- ⭐ The variant spelling, and the wiring that actually breaks. `resnet50TrainStepFaithfulB`
-- derives its entry name from `r34AdamVariant`, so `bf16` has to reach THAT call and not merely
-- the block renderers — otherwise the artifact lands at `…momdp64bf16_train_step.mlir` while
-- declaring `@resnet50in_momdp64_train_step` inside and the driver refuses at load with an entry
-- mismatch. ConvNeXt shipped that defect twice, R34's bf16 a third time.
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.heavyBall
         false false false "" true == "momdp64bf16"
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.heavyBall == "momdp64"
-- ▶ And the slug must not trip the DRIVER's variant predicates, which read the same string to size
-- the checkpoint blob. `cdOn` is the dangerous one — a SUBSTRING test for "do", and a false
-- positive would silently add a dropout region to the layout with no error anywhere.
#guard ("momdp64bf16".splitOn "do").length == 1
#guard ("momdp64bf16".splitOn "acc").length == 1
#guard !"momdp64bf16".startsWith "ema"

-- ⭐⭐ GRADIENT ACCUMULATION — `planning/next_session_pipeline_then_r50.md` §4's blocker.
--
-- A FOURTH parameter region `G` and two runtime scalars, so one graph is both phases. `k` is in the
-- artifact name because the driver must agree with the `1/k` baked into `%ob1`/`%ob2`, and a
-- disagreement is silent (a wrong effective learning rate, no error anywhere).
--
-- ⚠ THE TWO ARE FOR DIFFERENT JOBS AND ONLY ONE IS THE RECIPE:
--   * `acc4x64`   — 1 replica, micro-batch 64, k = 4 ⇒ effective **256**. This is what
--     `lake build r50-accum-tie` gates, because the tie needs a SINGLE-DEVICE peer to compare
--     against and the DP render's `%loss` is replica-local (`tests/r50_dp_render_tie.py`).
--   * `accdp8x64` — 4 replicas, micro-batch 64, k = 8 ⇒ effective **2048**, which is RSB-A3's
--     design batch and LAMB's. ⚠ It is the batch, NOT the recipe: `planning/rsb_a3_r50_verified.md`
--     §2.3's LAMB and BCE-with-logits are still absent, so this is AdamW at bs2048 and must not be
--     described as `rsb-faithful`.
#eval IO.FS.writeFile "verified_mlir/resnet50in_acc4x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.adamwAccum 4) "resnet50in")

-- ⚠ `accdp4x64` exists for ONE reason: it is `acc4x64`'s data-parallel peer at the SAME k, which is
-- what `tests/r50_dp_render_tie.py` needs to carry `r50-accum-tie`'s verdict onto the DP
-- accumulation path. Without a matched-k pair that tie has nothing to diff. It is not a recipe.
#eval IO.FS.writeFile "verified_mlir/resnet50in_accdp4x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.adamwAccum 4) "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_accdp8x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.adamwAccum 8) "resnet50in")

-- ⭐⭐ BCE-WITH-LOGITS — RSB-A2/A3's loss (`planning/next_session_pipeline_then_r50.md` §4).
--
-- Same three regions and the same signature as the CE renders: the loss is not state, so nothing
-- in the driver moves. ⚠ `adam64bce` exists so `lake build r50-bce-tie` can recover the cotangent
-- from AdamW's `m' = 0.1·g` (LAMB's trust ratio is in the way); `lamb64bce` is the recipe-facing
-- pair. Both share ONE `lossCodeBce`, so a defect in the loss cannot be present in one and not the
-- other.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adam64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50in" (bce := true))

#eval IO.FS.writeFile "verified_mlir/resnet50in_lamb64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in" (bce := true))

-- ⭐⭐ LAMB — RSB-A3's optimizer (`planning/rsb_a3_r50_verified.md` §2.3). THREE regions, same
-- `[θ|m|v]` signature as `adam64`, because the trust ratio is computed inside the graph from θ and
-- the direction and needs no extra state. So the driver is byte-identical across `adam64` and
-- `lamb64` — the only per-net fact is which file it opens.
--
-- ⚠⚠ THIS IS LAMB, NOT `rsb-faithful`. That recipe is LAMB **at effective batch 2048** with
-- BCE-with-logits and a 160/224 resolution split; `planning/rsb_a2_resnet50.md` records LAMB at
-- bs512 giving 40.8% against 78.1%, so the batch is not a detail. Composing this with the
-- accumulation render (`.lambAccum`) is not built — see the handoff.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lamb64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in")

-- Pin the literal artifact paths above against the name the renderer emits, so a rename fails at
-- `lake build` rather than at run time as a shim "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 64 1 == "adam64"
#guard Proofs.StableHLO.r34AdamVariant 64 4 == "adamdp64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 Proofs.StableHLO.R34Opt.lamb == "lamb64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.adamwAccum 4) == "acc4x64"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.adamwAccum 8) == "accdp8x64"
-- ▶ the `wx` spellings. ⚠ `wx` trails the BATCH and precedes the `bce` suffix the caller appends,
-- so the shipping A3-fidelity name is `lambaccdp8x64wxbce`. Pinned because this net DERIVES its
-- entry name from the variant: a marker that moved would produce an artifact whose declared entry
-- disagrees with its own path, which the shim refuses outright rather than running the wrong graph.
#guard Proofs.StableHLO.r34AdamVariant 64 1 Proofs.StableHLO.R34Opt.lamb true == "lamb64wx"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) true
         == "lambaccdp8x64wx"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8) true
         == "lambacc8x64wx"
-- …and the flag OFF must still spell exactly what every committed artifact is named.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) false
         == "lambaccdp8x64"
-- The rank test, spelled out on the shapes it actually decides.
#guard Proofs.StableHLO.r34WdDecays "conv1w" [64,3,7,7] == true      -- conv weight: decayed
#guard Proofs.StableHLO.r34WdDecays "bn1g" [64] == false             -- BN γ: excluded
#guard Proofs.StableHLO.r34WdDecays "bn1b" [64] == false             -- BN β: excluded
#guard Proofs.StableHLO.r34WdDecays "fcb" [1000] == false            -- dense bias: excluded
#guard Proofs.StableHLO.r34WdDecays "fcw" [2048,1000] == true        -- dense weight: decayed
-- ⚠ The driver reads `k` back OUT of this string (`VerifiedTrain`'s `accK`). Pin the round trip
-- here, where the name is produced, rather than trusting two parsers to agree.
#guard ((Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.adamwAccum 8)).drop 5
          |>.takeWhile (· != 'x')) == "8"

-- Batch 256 on the forwards, matching the shim's val batch: 195 × 256 = 49,920 after tfds
-- drop_remainder, the count the reference reports scoring. ⚠ `evalBs` in the driver is read OFF
-- the artifact, so this need not equal the train batch.
-- ⭐⭐ RSB-A3's RESOLUTION SPLIT — train @160, eval @224 (`next_session_pipeline_then_r50.md` §4).
--
-- Slug `resnet50in160`, so the resolution is in the name the driver opens rather than in a suffix
-- some call site has to remember. ⚠ THE EVAL FORWARD IS DELIBERATELY AT A DIFFERENT RESOLUTION
-- FROM ITS OWN TRAIN STEP — that is the recipe, not an oversight: `q = 5` (160) for training,
-- `q = 7` (224) for scoring, one renderer, two instantiations.
--
-- ⚠ These are the RENDER half. They have no data path yet: `resnet50ImagenetVerified` is a 224 net
-- (`d0 = 3·224·224`) and the tfds shim produces 224 crops, so a 160 `VerifiedNetSpec` and a 160
-- shim are still owed before any of this can run. Rendering them now is what type-checks the
-- resolution parameterisation at `q = 5` — the part with the dependent-type risk.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lamb64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in160" (bce := true) (q := 5))

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- ⭐⭐⭐ **RSB-A3 ITSELF — LAMB × BCE × ACCUMULATION × 4 REPLICAS, AT 160.** The composition
-- `planning/next_session_rsb_a3.md` was written to reach, rendered 2026-08-06.
--
-- `lambaccdp8x64bce` = LAMB, BCE-with-logits, k = 8 accumulated micro-batches, 4 replicas × bs64
-- ⇒ effective batch **2048** — RSB-A3's design batch and the one LAMB was built for
-- (`planning/rsb_a2_resnet50.md` records LAMB at bs512 giving **40.8% against 78.1%**, so the batch
-- is not a detail). At `q = 5`, i.e. A3's 160² train resolution, scoring through
-- `resnet50in160_fwd_eval` at 224².
--
-- ⚠⚠ **WHAT THIS STILL IS NOT.** `wdExcludeNormBias` remains absent (§0), so BN γ/β and biases are
-- decayed where timm excludes them; and the BN regime is Ghost-BN over 64-image micro-batches, not
-- a genuine bs2048 forward. Quote it against §4.1's delta list, never as "RSB-A3 reproduced".
--
-- ⚠ The 1-replica peer below is NOT a second recipe — it is what the accumulation gates need. Both
-- `r50-accum-tie` and `r50-accum-shard-tie` compare against a SINGLE-DEVICE peer, so a DP-only
-- render would be ungateable. Same k, same loss, same resolution; only `replicas` differs.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambaccdp8x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5))

#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc8x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5))

-- ── ▶ `wx` — timm `no_weight_decay`, CLOSING `a3_paper_fidelity.md` §2.1 (2026-08-14) ──────────
-- The A3 recipe's LARGEST open delta, and the one its own ledger ranked "most likely to move the
-- final number". `resnet50ImagenetConfigRSBFaithful` sets `wdExcludeNormBias := true`; the render
-- above does not, so the 77.43% run decayed all 161 parameters — BN γ, BN β and every bias
-- included — at wd = 0.02.
--
-- ⚠⚠ **THIS IS A DIFFERENT FUNCTION, NOT A FIXED ONE, WHICH IS WHY IT GETS ITS OWN NAME.** The
-- `wx` renders are new artifacts beside the old ones rather than in place of them: the 77.43%
-- result belongs to the graph that produced it, and silently re-pointing that slug at a graph with
-- different decay semantics would make an already-quoted number unreproducible. §2a's
-- last-writer-wins race, in the one place where the loser is a finished 34-hour run.
--
-- ⭐ The exclusion is the PLAIN RANK TEST (`r34WdDecays`, `ds.length ≥ 2`), so it needs no name
-- carve-out: ResNet has no positional parameter for ViT's `pos` rule to apply to. What it excludes
-- here is every BN γ/β and every conv/dense bias.
--
-- ⚠ NOT YET RUN, and the ledger's warning applies in full: this changes the trajectory, so it can
-- only be compared to A3 by a fresh run, never by resuming one.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambaccdp8x64wxbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5)
    (wdExclude := true))
-- Its single-device peer, for the same reason the non-`wx` pair has one: `r50-accum-tie` and
-- `r50-accum-shard-tie` both compare against a 1-replica render, so a DP-only `wx` would be
-- ungateable — and `wx` is exactly the axis worth gating, since it changes 105 of 161 decay
-- operands and nothing about the arity.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc8x64wxbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5)
    (wdExclude := true))

-- ⭐⭐ **D1 — `wx` PLUS THE GLOBAL-NORM GRADIENT CLIP**, and this pair is the first R50 render that
-- is the optimizer timm's arg string actually names. `timm.optim.lamb.Lamb` defaults
-- `max_grad_norm = 1.0` and clips every step; every LAMB artifact above this line does not clip, so
-- each of them is `Lamb(max_grad_norm=None)` — a different optimizer, run by a real 34-hour job.
--
-- ⚠⚠ **A NEW ARTIFACT BESIDE THE OLD ONES, NOT A FIX OF THEM** — the `wx` renders' own reason, one
-- delta on: the 77.43% result belongs to the graph that produced it, and re-pointing a slug at a
-- graph with different gradient semantics makes an already-quoted number unreproducible.
--
-- ⚠⚠ **THE CLIP IS ON THE MEAN ACCUMULATED GRADIENT**, which is why the threshold this bakes is
-- `k·C = 8.0` and not `1.0`. The reference (`jax/Jax/Codegen.lean:2439`) forms `_gsum / _K` and
-- clips THAT; this graph never materialises the mean, so the fold runs on `Gt` and the threshold
-- moves with it — `min(1, kC/(‖Gt‖ + kε)) = min(1, C/(‖Gt‖/k + ε))`, equal by algebra, no new op.
-- ▶ Read `clipNormStr` before changing either constant.
--
-- ⚠ NOT YET RUN. Like `wx` before it this changes the trajectory, so it can only be compared to A3
-- by a fresh run, never by resuming one.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambaccdp8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5)
    (wdExclude := true) (gradClip := true))
-- Its single-device peer, for the reason the `wx` pair has one: `r50-accum-tie` and
-- `r50-accum-shard-tie` both compare against a 1-replica render, so a DP-only clip would be
-- ungateable — and the clip is exactly the axis worth gating, because the ONE thing that
-- distinguishes it from a per-parameter clip (`Proofs.clipFactor_shared`) is invisible to every
-- check that only asks whether the gradients got smaller.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5)
    (wdExclude := true) (gradClip := true))

-- The **2-GPU** peer of the 4-replica render above: `B := 128` per replica at the same `k = 8`, so
-- the global batch stays 128×2×8 = 2048 and the recipe is untouched. ⚠ Note what does NOT stay the
-- same: Ghost-BN normalises per micro-batch, so this run's BN regime is 128-image ghosts against
-- the 4×64 render's 64-image ones. For a wall-clock measurement that is immaterial; for any
-- accuracy claim it is a different regime and must be said out loud (§4.1's delta list).
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambaccdp8x128bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 128 1000 "1.0e-05" 2
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (q := 5))

-- ⚠ `k = 4` at ONE replica, the peer `r50-accum-tie` actually runs against (its gate is written at
-- k = 4). Rendering only k = 8 would leave the composed optimizer's accumulation ungated at the k
-- the gate uses.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc4x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 4) "resnet50in160" (bce := true) (q := 5))

-- ⚠ `lambdp64bce` — LAMB + BCE at 4 replicas, NO accumulation. It exists for ONE reason: it is the
-- peer `r50-accum-shard-tie` needs. That gate's identity is `acc(x1..xk) == DP([x1|..|xk])`, so the
-- right-hand side must be the DATA-PARALLEL render of the SAME optimizer and loss — `adamdp64` would
-- compare LAMB to AdamW and BCE to CE and fail for three reasons with one number to read.
-- ▶ §3's "needs a `lambdp64` peer rendered", at the composition's resolution.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambdp64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.lamb "resnet50in160" (bce := true) (q := 5))

-- ⚠⚠ THE NAME ROUND TRIP FOR THE COMPOSED VARIANT, pinned on the PRODUCING side — the driver reads
-- `k` back out of this exact string with a substring parse, and a disagreement is silent (a wrong
-- effective learning rate, no error anywhere). `tests/TestVariantPredicates.lean` pins the
-- consuming side, including the counterfactual that the OLD `startsWith "acc"` test MISSED this.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) == "lambaccdp8x64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8) == "lambacc8x64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 4) == "lambacc4x64"
-- ⚠ and the marker is NOT leading here, which is the whole defect: pin that fact so the spelling
-- cannot drift back to something the prefix test would have accepted by accident.
#guard ("lambaccdp8x64bce".startsWith "acc") == false
#guard (("lambaccdp8x64bce".splitOn "acc").length > 1) == true

-- ⚠⚠ **D1's spelling, pinned on the PRODUCING side.** `clip` TRAILS `wx` and both precede the
-- `bce` the R50 caller appends — `lambaccdp8x64wxclipbce`. The order is a CHOICE (ConvNeXt's,
-- reused rather than re-decided) and these are what make it a fixed one; `cnxAdamVariant` learned
-- twice that a marker's POSITION is as load-bearing as its presence.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) true true
         == "lambaccdp8x64wxclip"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8) true true
         == "lambacc8x64wxclip"
-- ⚠ The two flags are INDEPENDENT axes, so pin `clip` without `wx` too — otherwise nothing stops
-- the spelling from becoming "wxclip" as a single fused marker that only ever appears together.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) false true
         == "lambaccdp8x64clip"
#guard Proofs.StableHLO.r34AdamVariant 64 1 Proofs.StableHLO.R34Opt.lamb false true == "lamb64clip"
-- ⚠ And the inertness of the flag on the NAME, which is the half of §2c's defect that the entry
-- point sees: clip off must reproduce the committed spelling exactly.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) true false
         == "lambaccdp8x64wx"
-- ⚠⚠ THE ROUND TRIP THE DRIVER ACTUALLY MAKES: it parses `k` back out of this string, and the new
-- trailing markers must not disturb that parse. `lambaccdp8x64wxclipbce` still splits on "acc" and
-- still carries `8x` between "accdp" and the batch, exactly as the un-clipped spelling does.
#guard ("lambaccdp8x64wxclipbce".startsWith "acc") == false
#guard (("lambaccdp8x64wxclipbce".splitOn "acc").length > 1) == true

-- ⭐⭐ **§3.3 CLOSED: the `bce` marker is DERIVED, and these are what make that mean something.**
-- It used to be a `vSuffix : String` the caller spelled by hand beside a `bce : Bool` that swapped
-- the loss — two writers for one fact, on the artifact the 77.43% run trained on, with nothing
-- checking they agreed. Now one flag produces both, so the ONLY thing left to pin is the spelling.
-- ⚠ These are the full four-marker compositions, in order: optimizer, k, batch, `wx`, `clip`, `bce`.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true == "lambaccdp8x64wxclipbce"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true == "lambacc8x64wxclipbce"
-- ⚠ The RSB-A3 run's own artifact, pinned character for character: no `wx`, no `clip`, `bce` only.
-- This is the name whose graph produced 77.43%, and it must not move.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         false false true == "lambaccdp8x64bce"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         false false true == "lambacc8x64bce"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 4)
         false false true == "lambacc4x64bce"
#guard Proofs.StableHLO.r34AdamVariant 128 2 (Proofs.StableHLO.R34Opt.lambAccum 8)
         false false true == "lambaccdp8x128bce"
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.lamb
         false false true == "lambdp64bce"
-- ⚠ And `bce` OFF must leave every non-BCE spelling untouched — the half that says the new
-- parameter is inert, which is what let it be added without moving a byte.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         false false false == "lambaccdp8x64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 Proofs.StableHLO.R34Opt.lamb false false false == "lamb64"
-- ⚠⚠ `bce` LAST, and this is the counterfactual: the marker must not land before `wx`/`clip`.
-- `lambaccdp8x64bcewx` is what a wrong order produces, and the shim would refuse the call with
-- nothing but "entry mismatch" to say why.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true != "lambaccdp8x64bcewxclip"

-- ⭐⭐ **THE `wdStr` AXIS** (`verified_optimizer_parity.md` §3, RSB-A1's 0.01 against A3's 0.02).
-- ⚠ `%wd` is a BAKED constant, so two renders differing only in it must not share an artifact path
-- — `wdVariantMark` is what makes that collision unspellable rather than merely detectable by the
-- writer audit. These pin the spelling and, more importantly, both directions of "default".
#guard Proofs.StableHLO.optWdStr Proofs.StableHLO.R34Opt.lamb == "0.02"
#guard Proofs.StableHLO.optWdStr Proofs.StableHLO.R34Opt.adamw == "0.0001"
#guard Proofs.StableHLO.optWdStr (Proofs.StableHLO.R34Opt.lambAccum 8) "0.01" == "0.01"
-- ⚠ The DEFAULT, whether reached by omission or by the caller spelling it out, must produce NO
-- marker — otherwise `lambaccdp8x64bce` and `lambaccdp8x64bcewd002` would be the same graph under
-- two names, which is the collision in the other direction.
#guard Proofs.StableHLO.wdVariantMark (Proofs.StableHLO.R34Opt.lambAccum 8) == ""
#guard Proofs.StableHLO.wdVariantMark (Proofs.StableHLO.R34Opt.lambAccum 8) "0.02" == ""
#guard Proofs.StableHLO.wdVariantMark Proofs.StableHLO.R34Opt.adamw "0.0001" == ""
-- ⚠ …and a non-default one must produce a marker that carries the VALUE, not merely a flag: the
-- point is that 0.01 and 0.005 land on different paths, so `wd` alone would not do.
#guard Proofs.StableHLO.wdVariantMark (Proofs.StableHLO.R34Opt.lambAccum 8) "0.01" == "wd001"
#guard Proofs.StableHLO.wdVariantMark (Proofs.StableHLO.R34Opt.lambAccum 8) "0.005" == "wd0005"
#guard Proofs.StableHLO.wdVariantMark Proofs.StableHLO.R34Opt.adamw "0.02" == "wd002"
-- ⚠ AND THE SAME VALUE MEANS DIFFERENT THINGS TO THE TWO FAMILIES: 0.02 is LAMB's default (no
-- marker) and a 200x override for AdamW (marker). The mark is per-optimizer for that reason.
#guard Proofs.StableHLO.wdVariantMark Proofs.StableHLO.R34Opt.lamb "0.02" == ""
-- ▶ The full A1 spelling, end to end.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" == "lambaccdp8x64wxclipbcewd001"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.02" == "lambaccdp8x64wxclipbce"

-- ═══════════════════════════════════════════════════════════════════════════
-- § RSB-**A2** and RSB-**A1**, at 224² — the other two tiers of `sec:r50_a2_a1_cost`.
--
-- ⭐ These are A3's composition at `q := 7` instead of `q := 5`, on the 224 slug. LAMB × BCE ×
-- accumulation k=8 × wx × clip is unchanged; the resolution and (for A1) the baked decay are the
-- only knobs that move, which is why this is eight `#eval`s and not a proof chain. The renderer
-- already took every parameter needed.
--
-- ⚠⚠⚠ **AND THEY ARE NOT FAITHFUL A2/A1. READ THIS BEFORE QUOTING A NUMBER OFF THEM.**
-- `planning/verified_side_quest_counterparts.md` §4a called this "four `#eval` lines … it closes a
-- whole book section". The `#eval`s were four lines; the section does not close, because A2's
-- reference carries two regularisers this path cannot express. Checked against
-- `jax/MainResnet50Imagenet.lean`'s `resnet50ImagenetConfigA2Accum`, field by field:
--
--   ⛔ **MODEL EMA (`useEMA := true`, `emaDecay := 0.9999`) — STRUCTURALLY IMPOSSIBLE HERE, not
--      merely unrendered.** The EMA shadow and the gradient accumulator are THE SAME fourth region
--      of `[θ|m|v|·]` — see the packed-layout note above `accScalars` in this file — and
--      `VerifiedTrain.lean:1156` throws on the combination rather than letting one win:
--      *"variant selects BOTH the EMA shadow and gradient accumulation, and they occupy the same
--      fourth region … Render one or the other."*
--      ▶ And accumulation is not optional for A2: the recipe's effective batch is 2048, and at
--      224² there is no other way to reach it on 16 GB cards (the reference itself uses 4× accum).
--      So this is a REGION-LAYOUT limitation, and lifting it means a fifth region in the driver's
--      pack/unpack and in every optimizer's return list — not a flag.
--      ⚠ A3 did not hit this because A3's own recipe sets `useEMA := false`. A2 and A1 both set it.
--   ⛔ **STOCHASTIC DEPTH (`dropPath := 0.05`).** `LeanMlir/Proofs/Codegen/DropPath.lean` exists and
--      EfficientNet and ConvNeXt render `drop` variants off it, but neither `ResNet34RenderB` nor
--      `ResNet50RenderB` imports it (0 hits in both), and `r34AdamVariant` has no `drop` marker to
--      ask for it with. That is renderer work on the residual family, not a flag.
--      ⚠ A3 sets `dropPath := 0.0`, which is again why A3 never needed it.
--   ⚠ **GHOST-BN GROUP.** These render 8×64, i.e. 64-image ghosts. The reference reaches 2048 as
--      4 accumulation steps × 512 global (128 per device on four cards), i.e. 128-image ghosts.
--      Immaterial to a wall clock; a different regime for any accuracy claim.
--   ✅ Repeated augmentation 3× IS present — it rides the shim (`repeat(3)`), which is shared with
--      the `default` recipe. ⚠ The shim's own comment calls it a stream-level APPROXIMATION.
--
-- ⭐ **What these artifacts ARE good for**: they are A2's graph minus two regularisers, so they
-- price the tier honestly (the step cost is unaffected by EMA and by sd 0.05) and they are the
-- thing to extend once the fifth region and the `drop` marker exist. What they are not is an
-- 79.8%-target run. Quote them the way A3's deltas are quoted, never as "RSB-A2 reproduced".
--
-- ⭐⭐ **THE SHIM QUESTION, SETTLED BY MEASUREMENT AND NOT BY READING.** §4a warned to "check the
-- A2 shim rather than assuming it". Generated and diffed, 2026-08-27:
--   `generated_resnet50_imagenet_shim.py` (recipe `default`) and
--   `generated_resnet50_imagenet_a2accum_shim.py` are **byte-identical**, md5 d42c412beb4f…
-- — because `resnet50ImagenetConfigA2Accum` differs from `default` in `learningRate`,
-- `gradAccumSteps` and `wdExcludeNormBias`, all three optimizer-side. So **A2 needs no new
-- `VerifiedNetSpec` and no new shim**: it is `resnet50ImagenetVerified` under a new variant string.
-- ⚠ A1 is the opposite and it is a ONE-LINE difference: `_MIX_A` 0.100000 → 0.200000. Mixup is
-- data-side, so A1 gets its own shim and its own spec. See `scripts/gen_shims.sh`.

-- ── RSB-A2 @ 224, fp32. The 4-replica artifact a run would use, and its 1-replica peer. ─────────
-- ⚠ The 1-replica peer is not optional: `r50-accum-tie` and `r50-accum-shard-tie` both compare a
-- DP render against a single-device one, so a DP-only render is ungateable — the same reason the
-- 160 family carries `lambacc8x64wxclipbce` beside `lambaccdp8x64wxclipbce`.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambaccdp8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambacc8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true))

-- ── RSB-A2 @ 224, bf16. `resnet50in_momdp64bf16` already proves the bf16 path renders for this
-- net at this resolution, so this is a flag rather than an investigation. ─────────────────────
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambaccdp8x64wxclipbcebf16_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (bf16 := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambacc8x64wxclipbcebf16_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (bf16 := true))

-- ── RSB-A1 @ 224 — **A DISTINCT ARTIFACT, NOT "the same render as A2"**. ────────────────────────
-- ⚠⚠ The book's `sec:r50_a2_a1_cost` table said A1's renderer work was "same render as A2". It is
-- not, and this file already says why: `%wd` is a BAKED `stablehlo.constant`, not a runtime
-- operand, so A1's 0.01 against A2's 0.02 is a re-render. `wdVariantMark` appends `wd001`, which is
-- what stops the two from colliding on one path — the collision being unspellable is the point.
-- ▶ A1's other two deltas: epochs 600 (a driver knob, free) and Mixup α 0.2 (data-side, its shim).
-- ⭐ Its optimizer arm was gated BEFORE this render existed: `scripts/opt_step_tie.py` carries
-- `("lambacc8wxclipwd001", "generated_resnet50_imagenet_a1.py", 8, True)`, checked against an
-- emitted A1 trainer that bakes `WD = 0.010000`. So "the string reaches the constant block" is a
-- measurement here, not a code-reading claim.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambaccdp8x64wxclipbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambacc8x64wxclipbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true))

-- ── RSB-A1 @ 224, bf16. Rendered now rather than later for the reason ConvNeXt-S's MISSING bf16
-- twin is a `planning/verified_side_quest_counterparts.md` §4c item: a size or tier that ships
-- without its precision peer leaves a gap that reads as a decision and is really an accident of
-- ordering. Both tiers get the full (precision × replicas) square in one pass. ────────────────
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambaccdp8x64wxclipbcewd001bf16_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (bf16 := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_lambacc8x64wxclipbcewd001bf16_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (bf16 := true))

-- ══════════════════════════════════════════════════════════════════════════════
-- § RSB-A2 and A1 **WITH THE MODEL-EMA SHADOW** — the five-region renders, 2026-08-27.
--
-- ⭐⭐⭐ **THIS IS WHAT THE `⛔ MODEL EMA … STRUCTURALLY IMPOSSIBLE HERE` NOTE ABOVE WAS OWED.**
-- It read: *"lifting it means a fifth region in the driver's pack/unpack and in every optimizer's
-- return list — not a flag."* That is exactly what landed. The shadow and the accumulator are now
-- two regions rather than one slot two features claim, `[θ|m|v|G|E]`, and `emaOn`/`accOn` are
-- independent axes of `VerifiedVariant.nRegions` (3, 4 or 5).
--
-- ⚠⚠ **THE `ema` MARKER LEADS, and that is forced rather than chosen.** `VerifiedVariant.emaOn`
-- is a PREFIX test while `accOn` is a substring one, so `lambaccdp8x64wxclipbceema` would read as
-- accumulation-only — the driver would pack four regions into a five-region graph, misaligning
-- every parameter with no error anywhere. `tests/TestVariantPredicates.lean` pins both directions.
--
-- ⭐ **The composition is MEASURED, not merely rendered.** `scripts/opt_step_tie.py`'s
-- `emalambacc8wxclip` row runs `generated_resnet50_imagenet_a2accum.py`'s OWN `ema_update` against
-- the `E` slot this emits: **1.20e-07** against rtol 2e-6. Its controls, run the same session
-- (`runs/2026-08-27-r50-a2-a1-ema-fifth-region/`): a shadow reading the INCOMING θ rather than θ′
-- reads 1.14e-02, and the swapped region order reads 1.9e+01. Both are the defects a code reading
-- cannot rule out.
--
-- ⚠ **ONE A2 DELTA REMAINS AND IT IS NOT THIS ONE**: stochastic depth `dropPath := 0.05` still has
-- no importer on the residual family (neither `ResNet34RenderB` nor `ResNet50RenderB` imports
-- `DropPath.lean`, and `r34AdamVariant` has no `drop` marker). So these are A2's graph minus ONE
-- regulariser rather than minus two. Ghost-BN group 64-vs-128 is unchanged. Quote them the way
-- A3's deltas are quoted.
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambaccdp8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true))
-- ⚠ The 1-replica peer, for `r50-accum-tie`/`r50-accum-shard-tie`'s reason: a DP-only render is
-- ungateable, because both gates compare a DP render against a single-device one.
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambacc8x64wxclipbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true))

-- ── …and RSB-A1's pair, which differs in the BAKED `%wd` alone (0.01 against A2's 0.02). ───────
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambaccdp8x64wxclipbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambacc8x64wxclipbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true))

-- ⚠⚠ **NO bf16 TWINS HERE, AND THAT IS A DECISION RATHER THAN AN OMISSION.** §4c's own complaint
-- is that a tier shipping without its precision peer "reads as a decision and is really an accident
-- of ordering", and the A2/A1 fp32×bf16 square above was rendered in one pass for exactly that
-- reason. The EMA pair is different: the shadow's arithmetic is `e' = d·e + (1−d)·θ′` on MASTER
-- weights, which stay f32 in every bf16 render in this tree — so a bf16 twin of these would differ
-- from the fp32 twin in no line of the EMA block at all, while doubling the artifacts a coverage
-- gate has to carry. ▶ Render them the day a bf16 A2 run is actually scheduled, not before.

-- ══════════════════════════════════════════════════════════════════════════════
-- § RSB-A2 and A1 **COMPLETE** — EMA shadow AND stochastic depth, 2026-08-27.
--
-- ⭐⭐⭐ **THESE CLOSE `sec:r50_a2_a1_cost`'s TWO ⛔ ROWS.** §4a found the eight A2/A1 renders it had
-- just landed were not faithful A2/A1, because two of A2's regularisers had no expression here:
-- model EMA (the fourth-region collision, now a fifth region) and stochastic depth `dropPath :=
-- 0.05` (no importer on the residual family, now sixteen sites). Both are in these four.
--
-- ⚠⚠ **AND THE sd HALF NEEDED A FIX ON THE *REFERENCE* SIDE FIRST.** `bottleneck_block` drew
-- `jax.random.bernoulli(drop_key, keep_prob)` with NO shape argument — a SCALAR shared by the whole
-- batch, so the branch was dropped for every example or for none. timm 1.0.28's `drop_path` is
-- explicitly *"per sample"*, and this render is too. Nine emitters across `jax/Jax/Codegen.lean`
-- now call the per-example `_drop_branch` that file already contained for the transformer family.
-- Measured, because the two forms have the SAME EXPECTATION: at batch 64 and keep 0.95 the old form
-- dropped all 64 or none on **200/200** steps and the new one drops a mean of 3.23.
-- See `runs/2026-08-27-r50-a2-a1-ema-fifth-region/droppath_shape.log` and
-- `planning/verified_side_quest_counterparts.md` §6b.
--
-- ⚠ **WHAT IS STILL NOT A2**: ghost-BN group. These render 8×64, i.e. 64-image ghosts, against the
-- reference's 4 × 512-global (128 per device on four cards). Immaterial to a wall clock; a
-- different regime for any accuracy claim. That is now the ONLY delta, and it is the one §4a
-- listed as ⚠ rather than ⛔.
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambaccdp8x64wxclipdropbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true) (sd := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambacc8x64wxclipdropbce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true) (sd := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambaccdp8x64wxclipdropbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true) (sd := true))
#eval IO.FS.writeFile "verified_mlir/resnet50in_emalambacc8x64wxclipdropbcewd001_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in" (bce := true) (wdStr := "0.01") (q := 7)
    (wdExclude := true) (gradClip := true) (ema := true) (sd := true))

-- ⚠ THE FOUR sd SPELLINGS. `drop` is the SIXTH marker on these names and it goes in the MIDDLE —
-- between `clip` and `bce`, N3's grammar — so unlike `ema` it has neighbours on both sides.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" false true true == "emalambaccdp8x64wxclipdropbce"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" false true true == "emalambacc8x64wxclipdropbce"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" false true true == "emalambaccdp8x64wxclipdropbcewd001"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" false true true == "emalambacc8x64wxclipdropbcewd001"
-- ⚠⚠ **THE CONCATENATIONS, not the marker.** Three collisions have already shipped in this naming
-- and every one lived in a PAIR of markers meeting, never in the new marker alone. `clip` ++ `drop`
-- and `drop` ++ `bce` are the two new adjacencies:
#guard ("emalambaccdp8x64wxclipdropbce".splitOn "do").length == 1   -- `dr`, not `do`
#guard ("emalambaccdp8x64wxclipdropbce".splitOn "drop").length > 1
#guard ("emalambaccdp8x64wxclipdropbce".splitOn "rms").length == 1
#guard ("emalambaccdp8x64wxclipdropbce".splitOn "acc").length > 1
#guard "emalambaccdp8x64wxclipdropbce".startsWith "ema"
-- ⚠ and `drop` must not reach the `k` parse, which reads digits between `acc`/`accdp` and the `x`.
#guard ("emalambaccdp8x64wxclipdropbcewd001".splitOn "acc").length > 1
-- ▶ `accK`/`nRegions`/`sdOn` are the CONSUMING side and live in `VerifiedTrain.lean`, which imports
-- this file — pinned for these names in `tests/TestVariantPredicates.lean`.

-- ⚠ THE FOUR EMA SPELLINGS, pinned on the PRODUCING side like every one below them. The `ema`
-- prefix is the OUTERMOST marker any name in this file carries, and it goes in FRONT of a string
-- that already holds five — so it is run rather than reasoned about.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" false true == "emalambaccdp8x64wxclipbce"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" false true == "emalambacc8x64wxclipbce"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" false true == "emalambaccdp8x64wxclipbcewd001"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" false true == "emalambacc8x64wxclipbcewd001"
-- ⚠⚠ **AND THE MARKER MUST LEAD.** This is the counterfactual, and it is the one that is silent:
-- a trailing `ema` produces a name the driver reads as four regions, so the graph gets a blob one
-- region short and every parameter after θ is misaligned. Nothing throws.
#guard "emalambaccdp8x64wxclipbce".startsWith "ema"
#guard ("lambaccdp8x64wxclipbceema".startsWith "ema") == false
-- ⚠ and the prefix must disturb none of the four axes it is not: `ema` ++ `lamb` spells `emalamb`,
-- which contains no "rms", no "drop" and no "do".
#guard ("emalambaccdp8x64wxclipbce".splitOn "rms").length == 1
#guard ("emalambaccdp8x64wxclipbce".splitOn "drop").length == 1
#guard ("emalambaccdp8x64wxclipbce".splitOn "do").length == 1
#guard ("emalambaccdp8x64wxclipbce".splitOn "acc").length > 1
-- ▶ `nRegions`/`nScalars`/`emaRegion` are the CONSUMING side and live in `VerifiedTrain.lean`,
-- which imports this file — pinned for these names in `tests/TestVariantPredicates.lean`.

-- ⚠ THE FOUR NEW SPELLINGS, pinned on the PRODUCING side like every one above them. Two of these
-- (`lambaccdp8x64wxclipbce`, `lambacc8x64wxclipbce`) are already guarded further up — they are the
-- 160 family's names, and the A2 pair reuses them at a DIFFERENT SLUG, which is exactly why the
-- slug is part of the path and not part of the variant (N2 of the naming scheme).
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" == "lambacc8x64wxclipbcewd001"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" true == "lambaccdp8x64wxclipbcebf16"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "" true == "lambacc8x64wxclipbcebf16"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" true == "lambaccdp8x64wxclipbcewd001bf16"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" true == "lambacc8x64wxclipbcewd001bf16"
-- ⚠⚠ `bf16` TRAILS the decay marker, and this is the counterfactual: `…bf16wd001` is what the
-- wrong order produces and the shim would refuse it with nothing but "entry mismatch" to say why.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8)
         true true true "0.01" true != "lambaccdp8x64wxclipbcebf16wd001"
-- ⚠⚠ AND THE DRIVER'S PREDICATES MUST STILL READ THESE. `accOn` is a SUBSTRING test and `accK` is
-- parsed from AFTER the marker, so the new trailing markers must not disturb either — and none of
-- the four may accidentally spell `ema` or `do`, which select the EMA shadow and classifier
-- dropout respectively. `wd001` is the risk: it is the first marker here that ends in a digit run.
#guard ("lambaccdp8x64wxclipbcewd001bf16".splitOn "acc").length > 1
#guard ("lambaccdp8x64wxclipbcewd001bf16".startsWith "acc") == false
#guard ("lambaccdp8x64wxclipbcewd001bf16".startsWith "ema") == false
#guard ("lambaccdp8x64wxclipbcewd001bf16".splitOn "do").length == 1
#guard ("lambacc8x64wxclipbcewd001bf16".splitOn "do").length == 1
-- ▶ `accK` and `nRegions` are the CONSUMING side and live in `LeanMlir/VerifiedTrain.lean`, which
-- imports this file — so they cannot be guarded from here without a cycle. They are pinned for
-- these four names in `tests/TestVariantPredicates.lean`, which is where that file's own header
-- says the consuming half belongs.

#eval IO.FS.writeFile "verified_mlir/resnet50in160_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 64 1000 "1.0e-05" "resnet50in160" (q := 5))

-- ⭐ eval at 224 (`q = 7`) and batch 256, exactly as the 224 net scores. This is A3's
-- `train@160 / test@224 (crop 0.95)` split, and the only thing that makes it expressible is that
-- BN in `.eval` mode reads FROZEN per-channel statistics — which are resolution-independent.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 256 1000 "1.0e-05" "resnet50in160" (q := 7))

#eval IO.FS.writeFile "verified_mlir/resnet50in_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 256 1000 "1.0e-05" "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 256 1000 "1.0e-05" "resnet50in")

-- ═══════════════════════════════════════════════════════════════════════════
-- § `resnet50Verified` — the IMAGENETTE (10-class) renders.
--
-- `resnet50Verified` has existed as a layout skeleton since the R50 scoping work
-- ("no render, no proof chain, no artifact yet"); these are the artifacts. Same
-- renderer, same 224² backbone as `resnet50in` — the ONLY deltas are `nClasses`
-- 1000 → 10 and the slug, which is why this is three lines and not a new chain.
--
-- ⭐ B = 32 makes `r34AdamVariant` emit the bare name `"adam"` (the `if B == 32
-- then ""` branch), so the artifact, the `@resnet50_adam_train_step` entry point
-- and the driver's default `LEAN_MLIR_VARIANT=adam` are one string — matching
-- R34's Imagenette pairing exactly rather than inventing a second convention.
--
-- ⚠ The forwards are rendered at B = 32 too, because the driver scores eval
-- through them at the train batch. Any other batch needs re-rendered forwards or
-- `LEAN_MLIR_SKIP_EVAL=1`, and would otherwise be a shape error at first invoke.
#eval IO.FS.writeFile "verified_mlir/resnet50_adam_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 32 10 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50")

#eval IO.FS.writeFile "verified_mlir/resnet50_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 32 10 "1.0e-05" "resnet50")

#eval IO.FS.writeFile "verified_mlir/resnet50_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 32 10 "1.0e-05" "resnet50")
