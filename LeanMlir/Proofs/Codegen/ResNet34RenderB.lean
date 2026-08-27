import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Codegen.ResNet34Render
import LeanMlir.ViTRender

/-! # ResNet-34 AdamW train step rendered from the verified AST, at the BATCHED index

The §2b peer of `ResNet34Render.lean`. Same net, different two things, and both are the point:

* **BatchNorm is `bnBatchF`** — μ/var reduced over `[0,2,3]`, coupling the batch — not
  `bnPerChannelF`'s per-example `[2,3]`. That is the semantics the AdamW trainer has always run
  (`tests/TestResnet34Train.lean`), and §2b's decision was to keep it rather than move the trainer
  onto the per-example chain. `ResNet34Render.lean` renders the per-example net; this file renders
  the batch-BN one. **They are different functions** — that is exactly the divergence §2a found
  between the two `resnet34_fwd` writers, and the reason these are two files rather than a flag.
* **The whole graph sits at `N := B`**, so every batch-coupled `den` here is honest: `bnBatchF`,
  `bnBatchBack`, and the whole `*GradB` family reduce over the batch, and at `N = 1` they would
  each describe a one-example function while the emitted text reduces over all `B` (§2b).

The optimizer is the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple applied to the un-fused
`*GradB` gradients — the fusion `θ − lr·g`, not Adam, was what kept every `_adam_train_step` in
`tests/` (§2a). β₁/β₂/ε/wd are baked; `%lr`/`%bc1`/`%bc2` arrive as runtime `tensor<f32>` args.

**The cotangent is composed from kit ops, not fused.** The hand-written render inlines label
smoothing (α = 0.1, K = 10) into one `[B,10]` block; here it is
`softmaxRow → subB → scaleB → addVB → shiftB → divConstB`, every line `pretty` of a verified node.
Same function, different graph — so the render does NOT match the hand-written artifact op-for-op,
and the tie against it has to be numeric. `%loss` is report-only and stays outside the AST, exactly
as `cifar8_adam_train_step`'s does.

Render is value-independent (`skel` erases values), so placeholder zeros and `lr := 0`/`ε := 0` are
passed; the emitted literals carry the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- A trainable parameter: emitted name (no `%`), gradient SSA name, and shape. The AdamW tail is
    a fold over this list, so the θ/m/v output order cannot drift from the signature order. -/
structure PGrad where
  nm   : String          -- parameter name without `%` (`%{nm}`, `%{nm}m`, `%{nm}v` are the args)
  grad : String          -- SSA name of its un-fused gradient
  ds   : List Nat        -- parameter shape, for the emitted Adam ops
deriving Inhabited

/-- Saved forward SSA names a block's backward + gradient passes reference. -/
structure BFwdB where
  code : String
  xin : String       -- block input (the merged dx flows back to this)
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum
  c1 : String        -- conv1 output (= BN1 input)
  n1 : String        -- BN1 output (= relu1 pre-activation)
  r1 : String        -- relu1 output (= conv2 input)
  c2 : String        -- conv2 output (= BN2 input)
  cp : String        -- projection conv output (downsample only; "" for identity)
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's parameter
    gradients in func-arg order. -/
structure BBackB where
  code : String
  dx : String
  ps : List PGrad

-- ════════════════════════════════════════════════════════════════
-- § Block forward (batch BN)
-- ════════════════════════════════════════════════════════════════

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`, all at `N := B`. -/
private def idFwdB (B c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) (bf16 : Bool := false) : StateM Nat BFwdB := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk`/`zc`/`zin` are: the render produces
  -- TEXT, and `skel` erases every ℝ payload before a token is emitted. The rounding-bearing
  -- `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk zc else .conv (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk zc) (.operand xName zin))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zin))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk zc else .conv (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk zc) (.operand nR1 zin))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zin))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zin))
  let _ := zbn
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "" }

/-- Downsample block forward: strided body + strided projection skip. `cin→c`, `2hh→hh`. -/
private def downFwdB (B cin c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) (bf16 : Bool := false) : StateM Nat BFwdB := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  -- §2l step A: the projection shortcut is He et al.'s option-B **1×1**, not the 3×3 this repo
  -- rendered until 2026-07-30. It is a SEPARATE kernel from the block's `zk1` — sharing that
  -- binding is exactly how the deviation stayed invisible (§2k).
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk`/`zc`/`zin` are: the render produces
  -- TEXT, and `skel` erases every ℝ payload before a token is emitted. The rounding-bearing
  -- `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk1 zc else .convStrided (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk1 zc) (.operand xName zinS))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zout))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (if bf16 then .convBf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk2 zc else .conv (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk2 zc) (.operand nR1 zout))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zout))
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := hh) (w := ww) zrnd s!"%{p}Wp" (biasName convBias s!"%{p}bp" c) zkp zc else .convStrided (h := hh) (w := ww) s!"%{p}Wp" (biasName convBias s!"%{p}bp" c) zkp zc) (.operand xName zinS))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zc zc (.operand nCp zout))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + UN-FUSED parameter gradients
-- ════════════════════════════════════════════════════════════════

/-- Identity block backward + its 8 parameter gradients. -/
private def idBackGradB (B c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String)
    (convBias : Bool) (bf16 : Bool := false) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk`/`zc`/`zin` are: the render produces
  -- TEXT, and `skel` erases every ℝ payload before a token is emitted. The rounding-bearing
  -- `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zin (.operand dyName zin))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W2" zk zc (.operand nDn2 zin) else .convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk zc (.operand nDn2 zin))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zin (.operand nDc2 zin))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W1" zk zc (.operand nDn1 zin) else .convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk zc (.operand nDn1 zin))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zin) (.operand nDa zin))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2
  let (cW1, nW1) ← pretty B (if bf16 then .convWeightGradBBf16 zrnd xName zc zin zk (.operand nDn1 zin) else .convWeightGradB xName zc zin zk (.operand nDn1 zin))
  let (cb1, nb1) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn1 zin)) else pure ("", "")
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (if bf16 then .convWeightGradBBf16 zrnd f.r1 zc zin zk (.operand nDn2 zin) else .convWeightGradB f.r1 zc zin zk (.operand nDn2 zin))
  let (cb2, nb2) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn2 zin)) else pure ("", "")
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbn (.operand nDa zbn))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b1", nb1, [c]⟩] else []) ++
                [⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                 ⟨s!"{p}W2", nW2, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b2", nb2, [c]⟩] else []) ++
                [⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩] }

/-- Downsample block backward + its 12 parameter gradients. -/
private def downBackGradB (B cin c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String)
    (convBias : Bool) (bf16 : Bool := false) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0      -- §2l step A: the 1×1 option-B shortcut
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn  : Vec (B*(c*(hh*ww))) := fun _ => 0
  -- ▶ The rounding is a PLACEHOLDER here, exactly as `zk`/`zc`/`zin` are: the render produces
  -- TEXT, and `skel` erases every ℝ payload before a token is emitted. The rounding-bearing
  -- `den` lives in the tie theorems, not here.
  let zrnd : ℝ → ℝ := fun r => r
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zout (.operand dyName zout))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (if bf16 then .convBackBatchedBf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W2" zk2 zc (.operand nDn2 zout) else .convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk2 zc (.operand nDn2 zout))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zout (.operand nDc2 zout))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (if bf16 then .convStridedBackBatchedBf16 (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W1" zk1 zc (.operand nDn1 zout) else .convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk1 zc (.operand nDn1 zout))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zc zbn (.operand nDa zbn))
  let (cDcp, nDcp) ← pretty B (if bf16 then .convStridedBackBatchedBf16 (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}Wp" zkp zc (.operand nDnp zout) else .convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" zkp zc (.operand nDnp zout))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zinS) (.operand nDcp zinS))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2 Wp bp gp btp
  let (cW1, nW1) ← pretty B (if bf16 then .convStridedWeightGradBBf16 zrnd xName zc zinS zk1 (.operand nDn1 zout) else .convStridedWeightGradB xName zc zinS zk1 (.operand nDn1 zout))
  let (cb1, nb1) ← if convBias then pretty B (.convStridedBiasGradB (h := hh) (w := ww) zk1 zinS zc (.operand nDn1 zout)) else pure ("", "")
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (if bf16 then .convWeightGradBBf16 zrnd f.r1 zc zout zk2 (.operand nDn2 zout) else .convWeightGradB f.r1 zc zout zk2 (.operand nDn2 zout))
  let (cb2, nb2) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk2 zout zc (.operand nDn2 zout)) else pure ("", "")
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbn (.operand nDa zbn))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  let (cWp, nWp) ← pretty B (if bf16 then .convStridedWeightGradBBf16 zrnd xName zc zinS zkp (.operand nDnp zout) else .convStridedWeightGradB xName zc zinS zkp (.operand nDnp zout))
  let (cbp, nbp) ← if convBias then pretty B (.convStridedBiasGradB (h := hh) (w := ww) zkp zinS zc (.operand nDnp zout)) else pure ("", "")
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbn (.operand nDa zbn))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2 ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,cin,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b1", nb1, [c]⟩] else []) ++
                [⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                 ⟨s!"{p}W2", nW2, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b2", nb2, [c]⟩] else []) ++
                [⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩,
                 ⟨s!"{p}Wp", nWp, [c,cin,1,1]⟩] ++
                (if convBias then [⟨s!"{p}bp", nbp, [c]⟩] else []) ++
                [⟨s!"{p}gp", ngp, [c]⟩, ⟨s!"{p}btp", ntp, [c]⟩] }

end Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The optimizer tail — proven ops per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- Which optimizer tail this render emits. The forward, the backward, the 146 un-fused parameter
    gradients and the whole packed signature are **shared** — only the per-parameter tail differs,
    the `CifarOpt` shape from `CnnRender` (handoff §2i) brought to R34.

    * `.adamw` — the committed recipe. Byte-for-byte what this file emitted before the threading.
    * `.heavyBall` — **the `jax/MainResnetImagenet.lean` reference rule**: coupled L2 decay, then
      heavy-ball momentum. See `optOne` for why it needs no new `SHlo` op. -/
inductive R34Opt
  /-- `θ' = θ − lr·(m̂/(√v̂+ε)) − lr·wd·θ`; both moments live. -/
  | adamw
  /-- `g ← g + wd·θ`, `v' = μ·v + g`, `θ' = θ − lr·v'`; velocity in the `v` slot, `m` untouched. -/
  | heavyBall
  /-- ⭐⭐ **LAMB** (You et al. 2019) — RSB-A3's optimizer, `planning/rsb_a3_r50_verified.md` §2.3's
      one ESTIMATED line, now measured at **two new ops**. Adam moments give a direction
      `r = m̂/(√v̂+ε) + wd·θ`, then a PER-PARAMETER-TENSOR trust ratio `‖θ‖/‖r‖` rescales the step.
      `Proofs.Lamb` is the ℝ reference; see `optOne`. -/
  | lamb
  /-- ⭐ **AdamW over `k` accumulated micro-batches** — `planning/next_session_pipeline_then_r50.md`
      §4's blocker. A FOURTH parameter region `G` holds the running gradient sum, and the graph is
      one function for both phases with two runtime scalars deciding which it is. See `optOne`. -/
  | adamwAccum (k : Nat)
  /-- ⭐⭐ **LAMB over `k` accumulated micro-batches — RSB-A3's ACTUAL optimizer**, and the
      composition `planning/next_session_rsb_a3.md` §1 exists to make expressible.

      ▶ **The observation that makes this one constructor rather than a redesign:** the accumulator
      `Gt = akeep·G + g` sits UPSTREAM of the optimizer and does not care who consumes it. So the
      accumulate/apply machinery is shared verbatim with `.adamwAccum` (see `accumScalarConsts`,
      which both arms emit), and only the tail that consumes `Gt` differs.

      ⚠ An accumulate micro-batch needs `m' = m`, `v' = v`, `θ' = θ`. The first two come from
      `%b1 = %b2 = 1`, `%ob1 = %ob2 = 0` exactly as for AdamW. **`θ' = θ` comes from `lr = 0`**,
      because LAMB's parameter step is `sgdParamF θ lr (trust·r)` — at `lr = 0` that is `θ − 0·(…)`
      exactly, with no decay term left running, since LAMB's `wd` lives INSIDE `r` and the zero
      multiplies it away. (AdamW gets the same result for a different reason: its decay is
      DECOUPLED, so `lr = 0` kills it too.)

      ⚠ `lambDirF` also reads `%b1..%ob2`, so on an accumulate micro-batch it computes an `r` built
      from `β₁·m` rather than the real moment. **That is harmless and is said out loud here**: `r`
      feeds only `lambScaleF → sgdParamF`, and `lr = 0` discards it. Nothing stateful is written —
      the moments are passthroughs and θ is frozen, so the accumulate phase's ONLY effect is `Gt`. -/
  | lambAccum (k : Nat)
deriving DecidableEq, Repr

/-- The `%aup`-driven scalar block that turns ONE graph into both accumulation phases.

    ⭐⭐ **ONE WRITER, shared by `.adamwAccum` and `.lambAccum`.** These eight lines are the whole
    accumulate/apply mechanism, and duplicating them per optimizer is exactly the double-writer
    failure `planning/next_session_rsb_a3.md` §1.1 wanted the type restructured to avoid — the
    restructure's real purpose was to stop this block existing twice, and factoring it out buys that
    without the ~8 match sites and 13-artifact re-render the restructure costs.

        accumulate (%aup = 0):  β₁ = 1, (1−β₁) = 0  ⇒  m' = m,  v' = v   exactly
        apply      (%aup = 1):  β₁ = 0.9, (1−β₁)/k  ⇒  m' = 0.9·m + (1−β₁)·(Gt/k)

    ⭐ `1/k` is folded in HERE, and asymmetrically: `%ob1` carries `1/k` while `%ob2` carries `1/k²`,
    because `v` consumes the gradient SQUARED. `v' = β₂v + ((1−β₂)/k²)·Gt² = β₂v + (1−β₂)·(Gt/k)²` —
    the identity that makes accumulation equal a real large-batch step rather than the "mean of
    per-micro-batch second moments" a naive implementation produces.

    ⚠ `fmt12`, not `fmt6`: at k = 4, `(1−β₂)/k² = 6.25e-5`, and `fmt6` emits `0.000063` — 0.8%
    wrong, baked, in the optimizer.

    ⚠ β₁/β₂ are 0.9/0.999 for BOTH optimizers (LAMB's moments ARE Adam's), which is why this block
    needs no per-optimizer parameter. Only `%eps`/`%wd` differ, and those stay in `optConstsB`. -/
def accumScalarConsts (k : Nat) : String :=
  let kf := k.toFloat
  "    %aone = stablehlo.constant dense<1.0> : tensor<f32>\n" ++
  s!"    %aob1 = stablehlo.constant dense<{fmt12 (1.0 - 0.9)}> : tensor<f32>\n" ++
  "    %ab1d = stablehlo.multiply %aup, %aob1 : tensor<f32>\n" ++
  "    %b1 = stablehlo.subtract %aone, %ab1d : tensor<f32>\n" ++
  s!"    %aob1k = stablehlo.constant dense<{fmt12 ((1.0 - 0.9) / kf)}> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.multiply %aup, %aob1k : tensor<f32>\n" ++
  s!"    %aob2 = stablehlo.constant dense<{fmt12 (1.0 - 0.999)}> : tensor<f32>\n" ++
  "    %ab2d = stablehlo.multiply %aup, %aob2 : tensor<f32>\n" ++
  "    %b2 = stablehlo.subtract %aone, %ab2d : tensor<f32>\n" ++
  s!"    %aob2k = stablehlo.constant dense<{fmt12 ((1.0 - 0.999) / (kf * kf))}> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.multiply %aup, %aob2k : tensor<f32>\n"

/-- **Is this parameter in timm's `no_weight_decay` group?**, recovered from the ONE name
    `r34WdName` produces rather than passed alongside it.

    ⚠⚠ Derived and not a second argument, on purpose. The skip-list now controls TWO things —
    whether the decay term enters `r` (`%wdz`) and whether the trust ratio applies at all
    (`recipe_fidelity_diffs.md` D2) — and a caller threading a `Bool` beside the name is exactly the
    two-writers shape that lets them disagree: a parameter decayed but not adapted, or the reverse.
    One name, one predicate, both consumers downstream of it. -/
def wdNameExcludes (wdName : String) : Bool := wdName != "%wd"

/-- `(θ', m', v')` for one parameter, from its un-fused gradient.

    For `.adamw` the three ops are the proven `adamMNextF`/`adamVNextF`/`adamWParamF`
    (`adamW_triple_faithful` bundles their `den`s into `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are
    baked literals; `%lr`/`%bc1`/`%bc2` are runtime `tensor<f32>` args, so one render serves a whole
    LR schedule. For `.heavyBall` see the inline notes — same discipline, different three ops, and
    `m` becomes a passthrough so the packed signature does not move.

    At `replicas > 1` the gradient is first averaged across devices by
    `ViTRender.emitGradAllReduce`. **That collective is a TRUSTED CARVE-OUT** — it is emitted text,
    not `pretty` of an AST node, so it is outside every faithfulness theorem here. What the proofs
    still cover is unchanged and is the whole rest of the graph: the optimizer tail consumes the
    averaged gradient as an `.operand`, exactly as it consumed the raw one, so the `den` side does
    not shift. What is trusted is that `all_reduce(add)/N` computes the mean — handoff §5:
    *"the gradient averaging is a proven identity; the collective implementing it is trusted,
    exactly like the lowerer."* §2b's `%loss` bug is the standing reminder that a carve-out needs
    its own numeric check; here that is the cifar8 exact decomposition gate (no BN ⇒ the identity
    holds exactly), because at R34 scale BN makes N×b ≠ 1×(N·b) BY DESIGN and no exact tie exists.

    At `replicas ≤ 1` this emits **nothing** and threads the raw gradient, so the single-device
    render stays byte-identical — which is the cheap self-check that this insertion is inert. -/
-- ⚠ PUBLIC (was `private` until 2026-08-05): `ResNet50RenderB.lean` folds the SAME optimizer
-- tail over its own parameter list. Duplicating it there would put AdamW/heavy-ball semantics
-- in two places, which is the double-writer failure this repo keeps paying for. Visibility does
-- not change emitted bytes, so every committed R34 artifact is untouched.
def optOne (opt : R34Opt) (B : Nat) (replicas : Nat) (g : PGrad)
    -- ⚠ TRAILING and DEFAULTED to `"%wd"`, so every existing call site is unchanged and every
    -- committed R34/R50 artifact re-renders byte-identically. The caller decides per PARAMETER
    -- whether this is `"%wd"` or the zero constant `"%wdz"` — timm's `no_weight_decay` skip-list
    -- (`a3_paper_fidelity.md` §2.1), which ConvNeXt and ViT already implement this exact way.
    -- ⭐ It needs NO new op: `adamWParamF`/`lambDirF`/`momVNextF` all take the decay as an OPERAND
    -- NAME, so excluding a parameter is binding that name to a zero rather than changing a graph.
    (wdName : String := "%wd")
    -- ⚠ **`preAvg` — the caller has already all-reduced (and clipped) this gradient**, so the
    -- collective must not be emitted a second time. The global-norm clip needs EVERY gradient at
    -- once while this function is per parameter, and under DP the clip must come AFTER the
    -- `all_reduce` (the reference clips the combined gradient; clipping per replica clips 161
    -- PARTIAL gradients — a different function that still trains and still descends). So at
    -- `gradClip := true` the caller hoists both and sets this. `ConvNeXtRender.convnextAdamOne`
    -- carries the identical flag for the identical reason; `planning/grad_clip.md` §4.
    -- ⭐ It needs no interface change beyond the flag: `emitGradAllReduce` at `replicas ≤ 1` emits
    -- NOTHING and threads its input name straight through, so forcing 1 here is exactly "skip it".
    (preAvg : Bool := false)
    -- ⚠⚠ **`accIn` — the caller has already emitted the ACCUMULATOR too, and this is its RAW
    -- output name.** Only the accumulating optimizers read it, and only under the clip.
    --
    -- The reference clips the MEAN ACCUMULATED gradient, not the micro-batch one
    -- (`jax/Jax/Codegen.lean:2439` — `grads = _gsum / _K` and only THEN the clip line), so the fold
    -- has to run on `Gt`, which is computed here, per parameter. The caller therefore hoists the
    -- `momVNextF` as well, folds the norm across all 161 of them, clips, and passes the clipped
    -- total back in `g.grad` while naming the UNCLIPPED one here.
    --
    -- ⚠⚠ **The two must stay distinct, and that is the whole subtlety.** `%<p>a` rides out as the
    -- fourth region and is the carry the NEXT micro-batch accumulates onto; the reference's carry
    -- (`_gsum`) is raw, clipped only on the way into the optimizer. Returning the clipped total
    -- here would compound the clip across the k micro-batches of every cycle — a contraction that
    -- trains and descends and is not the recipe.
    (accIn : Option String := none)
    -- ▶▶ **`ema` — the MODEL-EMA shadow, a region of its own** (`planning/ema.md`, lifted onto the
    -- residual family 2026-08-27 for RSB-A2/A1). One extra op per parameter, emitted AFTER the
    -- optimizer's own tail because the shadow tracks the UPDATED weight: `e' = %emad·e + %oemad·θ'`.
    --
    -- ⭐ It needs NO new `SHlo` constructor. `Proofs.adamMNext b ob m g = b·m + ob·g` instantiated
    -- at `(b := %emad, ob := %oemad, m := e, g := θ')` denotes exactly the exponential moving
    -- average, which is the same reading `.adamwAccum`'s accumulator takes of `momVNextF`. So the
    -- faithfulness theorems carry over untouched and this costs none of the ten-site surgery an
    -- added op does. `ViTRender.vitAdamOne` has emitted the identical line since `ema.md`.
    --
    -- ⚠⚠ **IT READS `nT`, THE UPDATED PARAMETER — NOT `%<p>`.** The reference EMAs the weights
    -- AFTER the optimizer moves them (`ema_params = ema_update(ema_params, params, step)` follows
    -- the `train_step` call, `jax/Jax/Codegen.lean:3017`). Reading the incoming θ instead gives a
    -- shadow lagging by one step — a number that trains, descends and is quietly not the
    -- reference's.
    --
    -- ⚠ **The DECAY is a runtime scalar, not a constant**, because it is warmup-corrected:
    -- `d = min(decay, (1+t)/(10+t))` moves every step and the driver computes it. Baking 0.9999
    -- here is the EMA-warmup defect this repo has already paid for once.
    (ema : Bool := false) :
    StateM Nat (String × String × String × String × Option String × Option String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let replicas := if preAvg then 1 else replicas
  let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
  let gr : SHlo n := .operand gAvg z
  -- ⚠ Emitted by every arm below rather than once here, because it consumes each arm's OWN `nT`.
  -- Hoisting it would need the updated-parameter name before the arm that produces it has run.
  let emaTail : String → StateM Nat (String × Option String) := fun nT =>
    if ema then do
      let (c, nE) ← pretty B (.adamMNextF s!"%{g.nm}e" "%emad" "%oemad" g.ds 0 z (.operand nT z))
      pure (c, some nE)
    else pure ("", none)
  match opt with
  | .adamw =>
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
    let (cT, nT) ← pretty B (.adamWParamF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" wdName g.ds 0 0 0 0 0 0 0 z z z gr)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cM ++ cV ++ cT ++ cE, nT, nM, nV, none, nE)
  | .lamb =>
    -- ⭐⭐ LAMB, in four ops per parameter, TWO of which are new (`planning/rsb_a3_r50_verified.md`
    -- §2.3 estimated "2–3"; measured at 2, because `gradSumSqAccF` was already here for the clip
    -- and `sgdParamF` for heavy-ball). `Proofs.Lamb` carries the ℝ reference and the clauses.
    let z1 : Vec 1 := fun _ => 0
    -- ① the DIRECTION, `r = m̂/(√v̂+ε) + wd·θ`, from the incoming moments and this step's gradient.
    -- ⚠ ε OUTSIDE the root and the decay INSIDE `r` — both placements are load-bearing and both
    -- have a plausible wrong neighbour (RMSProp-TF's `√(v̂+ε)`, AdamW's decay after the ratio).
    let (cR, nR) ← pretty B (.lambDirF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%eps" wdName g.ds 0 0 0 0 0 0 z z z gr)
    -- ② `‖θ‖²`, THIS parameter's own. ⚠ Seeded at `%lzero` and never folded across parameters —
    -- that single-leaf fold is the entire difference from the global-norm clip, whose whole
    -- semantic content is that ONE scalar is shared (`Proofs.clipFactor_shared` against
    -- `Proofs.lambScale_not_shared`). The two features emit nearly the same lines.
    -- ⭐⭐ **D2: the no_weight_decay group is NOT layer-adapted, and the fix is to SKIP this op.**
    -- timm reads `if weight_decay != 0 or group['always_adapt']:` before computing the ratio, so an
    -- excluded parameter takes a plain Adam step at `trust = 1`. Feeding `%lzero` here — the same
    -- zero this op would otherwise seed from — makes `wn2 = 0`, and `Proofs.lambTrust_zero_weight`
    -- (`lambTrust 0 rn2 = 1`, already `@[simp]`) says that IS 1. ▶ No new op, no new constructor,
    -- no new theorem; the excluded params emit one op FEWER than before.
    -- ⚠ The existing zero-norm guard does not already cover this. It fires at `‖θ‖ = 0` exactly,
    -- i.e. step one, where every BN β and bias starts; from step two the parameter is
    -- small-but-nonzero and `‖θ‖/‖r‖` collapses to ~0.01–0.1 against timm's 1.0. That is why
    -- `lambTrust_zero_weight` could hold on both sides while both were wrong.
    -- ⚠ INERT unless the skip-list is on: with `wdExclude := false` every `wdName` is `"%wd"`, so
    -- this takes the `else` and every committed non-`wx` artifact re-renders byte-identically.
    let (cN, nN) ← if wdNameExcludes wdName then pure ("", "%lzero")
                   else pretty B (.gradSumSqAccF (n := n) g.ds (.operand "%lzero" z1)
                                   (.operand s!"%{g.nm}" z))
    -- ③ `trust · r`, with `‖r‖²` reduced inside the op from its own tensor child.
    let (cS, nS) ← pretty B (.lambScaleF (n := n) g.ds (.operand nN z1) (.operand nR z))
    -- ④ `θ' = θ − lr·(trust·r)` — `sgdParamF`, an op that already exists, applied to the scaled
    -- direction exactly as `.heavyBall` applies it to the velocity.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nS z))
    -- the moments themselves, unchanged from `.adamw` — LAMB's m and v ARE Adam's.
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cM ++ cV ++ cR ++ cN ++ cS ++ cT ++ cE, nT, nM, nV, none, nE)
  | .adamwAccum _ =>
    -- ⭐ **The whole feature, and it adds exactly ONE op per parameter.**
    --
    -- ① the accumulator, `Gt = akeep·G + g`. That is `momVNextF` read the way `.heavyBall` reads it
    -- for coupled L2 — `Proofs.momVNext μ v g = μ·v + g`, so instantiating `(μ := %akeep, v := G)`
    -- denotes exactly the accumulation. **No new `SHlo` constructor**, so none of the ten-site
    -- surgery an added op costs, and the faithfulness theorem carries over untouched.
    --
    -- ⚠ `%akeep` is 0 on the FIRST micro-batch of a cycle and 1 after, so the cycle RESETS by
    -- dropping the previous total rather than by a separate zeroing pass — which is why one scalar
    -- and one op suffice, and why there is no "clear the accumulator" step that could be skipped.
    -- ⚠ Under `accIn` the caller emitted this op itself (it needed `Gt` to fold the global norm),
    -- and `gr` is already the CLIPPED total — so the tail reads `gr` while the fourth region still
    -- reports the raw `Gt`. See `accIn`'s note: those two must not collapse into one name.
    let (cG, nG, gt) ← match accIn with
      | some a => pure ("", a, gr)
      | none   => do
          let (c, nm') ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z gr)
          pure (c, nm', (.operand nm' z : SHlo n))
    -- ② the moments and the parameter, **byte-identical to `.adamw`'s** except that they consume
    -- `Gt` rather than `g`. The `1/k` that turns a SUM into a MEAN is not applied here: it is folded
    -- into `%ob1 = (1−β₁)/k` and `%ob2 = (1−β₂)/k²` by `optConstsB`, because `v` is QUADRATIC in the
    -- gradient and a single shared scale factor cannot serve both moments. Folding it is what keeps
    -- this from needing a scalar-multiply op the vocabulary does not have.
    --
    -- On an ACCUMULATE micro-batch `optConstsB` sets `%b1 = %b2 = 1` and `%ob1 = %ob2 = 0`, so both
    -- moments are exact passthroughs; `%lr = 0` freezes θ, and AdamW's decay is DECOUPLED (`−lr·wd·θ`)
    -- so lr = 0 freezes it COMPLETELY rather than leaving a decay term running k times per step.
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gt)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gt)
    let (cT, nT) ← pretty B (.adamWParamF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" wdName g.ds 0 0 0 0 0 0 0 z z z gt)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cG ++ cM ++ cV ++ cT ++ cE, nT, nM, nV, some nG, nE)
  | .lambAccum _ =>
    -- ⭐⭐ **RSB-A3's optimizer.** Structurally: `.adamwAccum`'s ① accumulator, then `.lamb`'s tail
    -- reading `Gt` where it read `g`. Nothing else changes, and nothing new is introduced — no new
    -- `SHlo` constructor, so no ten-site surgery and every faithfulness theorem carries over.
    let z1 : Vec 1 := fun _ => 0
    -- ① the accumulator, `Gt = akeep·G + g` — the SAME `momVNextF` instantiation `.adamwAccum`
    -- uses, for the same reason (`Proofs.momVNext μ v g = μ·v + g` at `(μ := %akeep, v := G)`).
    -- ⚠ It is upstream of the optimizer and does not know which one follows: that independence is
    -- the whole reason this composition is a constructor and not a redesign.
    -- ⚠ Same `accIn` carve-out as `.adamwAccum`'s, and this is the arm RSB-A3 actually renders:
    -- timm's `Lamb` is the optimizer whose `max_grad_norm = 1.0` default makes the clip mandatory
    -- (D1), so `.lambAccum` + clip is the composition that has to be right.
    let (cG, nG, gt) ← match accIn with
      | some a => pure ("", a, gr)
      | none   => do
          let (c, nm') ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z gr)
          pure (c, nm', (.operand nm' z : SHlo n))
    -- ② LAMB's four ops, **byte-identical to `.lamb`'s except that they consume `Gt` rather than
    -- `g`** — the same substitution `.adamwAccum` makes to AdamW's three.
    let (cR, nR) ← pretty B (.lambDirF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%eps" wdName g.ds 0 0 0 0 0 0 z z z gt)
    -- ⚠ `‖θ‖²` reads θ ALONE — no gradient, so accumulation cannot reach it and this line is
    -- character-for-character `.lamb`'s, INCLUDING its D2 skip: an excluded parameter feeds
    -- `%lzero` so `lambTrust 0 _ = 1`. See the `.lamb` branch above for why.
    let (cN, nN) ← if wdNameExcludes wdName then pure ("", "%lzero")
                   else pretty B (.gradSumSqAccF (n := n) g.ds (.operand "%lzero" z1)
                                   (.operand s!"%{g.nm}" z))
    let (cS, nS) ← pretty B (.lambScaleF (n := n) g.ds (.operand nN z1) (.operand nR z))
    -- ④ `θ' = θ − lr·(trust·r)`. ⭐ THIS is where the accumulate phase is frozen: `%lr = 0` makes it
    -- `θ − 0·(…) = θ` exactly. LAMB's decay is inside `r`, which the zero multiplies away, so there
    -- is no decay term left running k times per optimizer step.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nS z))
    -- the moments, from `Gt`, with `%b1`/`%ob1` computed from `%aup` — passthroughs while accumulating.
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gt)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gt)
    let (cE, nE) ← emaTail nT
    pure (arS ++ cG ++ cM ++ cV ++ cR ++ cN ++ cS ++ cT ++ cE, nT, nM, nV, some nG, nE)
  | .heavyBall =>
    -- ── Three applications of ops that ALREADY EXIST. No new `SHlo` constructor, so none of the
    -- ten-site surgery (and none of the `StableHLOParse` roundtrip risk) an added op costs.
    --
    -- ① COUPLED L2 decay, `g ← g + wd·θ`. This reuses **`momVNextF`**, which is not a pun:
    -- `Proofs.momVNext μ v g = μ·v + g`, so instantiating `(μ := wd, v := θ)` denotes exactly
    -- `wd·θ + g`. Same function, so the faithfulness theorem carries over unchanged; only the
    -- *reading* of the two slots differs. NOTE this is COUPLED (into the gradient, so it flows
    -- through the velocity), not AdamW's DECOUPLED `−lr·wd·θ` — that difference is the whole
    -- reason `.adamw` cannot stand in for the reference recipe.
    let (cD, nD) ← pretty B (.momVNextF s!"%{g.nm}" wdName g.ds 0 z gr)
    let gwd : SHlo n := .operand nD z
    -- ② velocity, `v' = μ·v + g`.
    let (cV, nV) ← pretty B (.momVNextF s!"%{g.nm}v" "%mu" g.ds 0 z gwd)
    -- ③ HEAVY-BALL parameter step, `θ' = θ − lr·v'` — `sgdParamF` applied to the velocity rather
    -- than to the gradient. ⚠ This is deliberately NOT `momParamF`: that op is **Nesterov**
    -- (`θ − lr·(g + μ·v')`, see `Proofs.momParam_heavyBall_diff`), and the JAX reference this
    -- render exists to match steps by `v'` alone. Using the momentum-named op here would have been
    -- the obvious move and would have silently produced a different optimizer.
    let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nV z))
    -- `m` rides through untouched, so the packed `[θ|m|v]` signature is shared with `.adamw` and
    -- the driver is byte-identical across variants (the `CnnRender.optTail` `.sgd` convention).
    let (cE, nE) ← emaTail nT
    pure (arS ++ cD ++ cV ++ cT ++ cE, nT, s!"%{g.nm}m", nV, none, nE)

/-- **Does this parameter get weight decay?** timm's `no_weight_decay` rule, and it is the PLAIN
    RANK TEST with no name carve-out — every 1-D parameter is excluded: BN γ, BN β and every bias.

    ⚠ Identical to `cnxWdDecays` by construction rather than by coincidence: the rule is timm's,
    not the net's, and ConvNeXt's own docstring records that its ViT-style `nm != "pos"` carve-out
    does not apply to a net with no positional parameter. ResNet has none either.

    ⚠⚠ **This is `a3_paper_fidelity.md` §2.1, open since the A3 run.** The live A3 artifact has
    ZERO `%wdz` occurrences against ConvNeXt's 123 — so the 77.43% run decayed BN γ/β and every
    bias at wd = 0.02 where its reference (`resnet50ImagenetConfigRSBFaithful`, which sets
    `wdExcludeNormBias := true`) did not. Decay on pre-BN conv weights is renormalised away by BN
    and acts only as an effective-LR control; decay on γ/β is not, because γ directly scales the
    layer's output. The effect concentrates at low LR — i.e. in the cosine endgame. -/
def r34WdDecays (_nm : String) (ds : List Nat) : Bool := ds.length ≥ 2

/-- The decay operand for one parameter: the real `%wd`, or the zero constant when excluded. -/
def r34WdName (wdExclude : Bool) (nm : String) (ds : List Nat) : String :=
  if wdExclude && !r34WdDecays nm ds then "%wdz" else "%wd"

/-- The `%wdz` declaration an excluding render needs. ⚠ Emitted only when the flag is on, so at
    `wdExclude := false` not one byte moves and every committed artifact is untouched. -/
def wdzConst (wdExclude : Bool) : String :=
  if wdExclude then
    "    // ── timm no_weight_decay (wdExcludeNormBias): 1-D params take %wdz, not %wd ──\n" ++
    "    %wdz = stablehlo.constant dense<0.0> : tensor<f32>\n"
  else ""

/-- **How many micro-batches the optimizer accumulates over** — `k` for the two accumulating
    constructors and `1` for every other, so a caller can ask the question without a second `match`
    that could disagree with `accOn`'s.

    ⚠ It exists for the CLIP (`clipNormStr`/`clipEpsStr` below), which is the first feature whose
    emitted constants depend on `k` from OUTSIDE `accumScalarConsts`. -/
def optAccumK : R34Opt → Nat
  | .adamwAccum k => k
  | .lambAccum k  => k
  | _             => 1

/-- **The clip threshold as the render bakes it, `k·C`** — and the `k` is not a typo.

    ⚠⚠ **THE REFERENCE CLIPS THE MEAN ACCUMULATED GRADIENT, NOT THE MICRO-BATCH ONE.**
    `jax/Jax/Codegen.lean:2439` is unambiguous about the order:

    ```python
    grads = jax.tree.map(lambda _a: _a / _K, _gsum)   # the MEAN over k micro-batches
    loss  = jnp.mean(_ls)
    gn    = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree.leaves(grads)))
    grads = jax.tree.map(lambda g: g * jnp.minimum(1.0, C / (gn + 1e-6)), grads)
    ```

    This render never materialises that mean: `optOne`'s accumulator carries the SUM `Gt`, and the
    `1/k` is folded into `%ob1 = (1−β₁)/k` and `%ob2 = (1−β₂)/k²` downstream (`accumScalarConsts`,
    and it is split that way because `v` is QUADRATIC in the gradient). So the fold here runs on
    `Gt`, whose norm is `k·‖mean‖`, and the threshold must move with it:

    `min(1, kC / (‖Gt‖ + k·ε))  =  min(1, C / (‖Gt‖/k + ε))`

    — the reference's factor on the mean, exactly, with no new op and no division emitted. ▶ And the
    factor is then applied to `Gt` rather than to the mean, which is the same thing for the same
    reason: scaling commutes with the `1/k` the moments fold in afterwards.

    ⭐ `fmt12`, not `fmt6`, for `accumScalarConsts`' stated reason — these are baked literals in the
    optimizer, where nothing downstream would question a truncated one. At `k = 1` this is the
    identity and emits the plain threshold. -/
def clipNormStr (clipNorm : Float) (k : Nat) : String := fmt12 (clipNorm * k.toFloat)

/-- The clip's `ε`, scaled by the same `k` and for the same reason — see `clipNormStr`.

    ⚠ It is NOT cosmetic and it does not cancel: the reference's `+ 1e-6` is what keeps the factor
    from being `0/0` at a zero gradient (`Proofs.clipDenom_pos`), and leaving it unscaled while the
    numerator scales would shift the factor by `k` in exactly the regime the guard exists for. -/
def clipEpsStr (k : Nat) : String := fmt12 (0.000001 * k.toFloat)

/-- The rank-0 zero that seeds the global-norm fold. ⚠ Its own name rather than `%lzero`: that one
    exists only under the two LAMB constructors (`optConstsB`), and the clip is an independent axis
    that has to work over `.adamw` too. Emitted only when the flag is on, so at `gradClip := false`
    not one byte moves and every committed artifact is untouched — the same discipline `wdzConst`
    keeps one function up. -/
def clipZeroConst (gradClip : Bool) : String :=
  if gradClip then
    "    // ── timm Lamb.max_grad_norm (D1): seed of the GLOBAL squared-norm fold ──\n" ++
    "    %czero = stablehlo.constant dense<0.0> : tensor<f32>\n"
  else ""

/-- **The weight decay each optimizer bakes when the caller does not override it.**

    ⚠ The two families genuinely differ, and by 200×: AdamW's `1e-4` against LAMB's `0.02`, the
    latter off timm's a3 arg string (`lamb-cosine-lr0.008-wd0.02-…`). `optConstsB`'s `.lamb` arm
    records what reusing AdamW's number here would produce — a LAMB that is structurally right and
    200× off on the decay.

    ▶ It exists so that `wdStr` can mean "the optimizer's own value" by DEFAULT rather than by the
    caller restating a number that is already decided by the constructor — the two-writers shape
    `bce`/`vSuffix` was just removed for (`a3_paper_fidelity.md` §3.3). -/
def optWdDefault : R34Opt → String
  | .adamw | .heavyBall | .adamwAccum _ => "0.0001"
  | .lamb  | .lambAccum _              => "0.02"

/-- **The decay actually baked**: the caller's override, or `optWdDefault`. Empty means default.

    ⚠⚠ **`%wd` IS A BAKED `stablehlo.constant`, NOT A RUNTIME OPERAND — this parameterises the
    literal, it does not make the decay schedulable.** Unlike `%lr`, which stays a `tensor<f32>`
    argument so one graph serves a whole cosine, changing the decay is a RE-RENDER. That is the same
    shape `ConvNeXtRender.convnextAdamConsts` already has (`wdStr := "0.0001"`, with the ImageNet
    render passing `0.05`), copied rather than re-invented.

    ▶ Why it exists: RSB-**A1** uses wd = 0.01 where A3 uses 0.02
    (`planning/verified_optimizer_parity.md` §3), so A1 costs a re-render rather than a new op. -/
def optWdStr (opt : R34Opt) (wdStr : String := "") : String :=
  if wdStr.isEmpty then optWdDefault opt else wdStr

/-- **The variant marker for a NON-DEFAULT decay**, and it is not optional bookkeeping.

    ⚠⚠ **Two renders that differ only in a baked constant MUST NOT share a path.** `%wd` lives in
    the artifact, so an A1 render (0.01) and an A3 render (0.02) at the same optimizer, batch and
    replica count would otherwise both be `lambaccdp8x64wxclipbce` — the last-writer-wins race
    §2a cost this repo a committed artifact once already. `scripts/regen_verified_mlir.sh check`
    would catch it as a two-writer collision, but a collision that cannot be SPELLED is better than
    one that is merely detected (§3.3's lesson, one feature over).

    ▶ Spelling: `wd` ++ the digits with the point removed, so `0.01` → `wd001` and `0.005` →
    `wd0005`. Mechanical, and unambiguous because the leading `0` is kept. ⚠ Empty at the default,
    so every committed artifact keeps its name and its bytes. -/
def wdVariantMark (opt : R34Opt) (wdStr : String := "") : String :=
  if wdStr.isEmpty || wdStr == optWdDefault opt then "" else "wd" ++ wdStr.replace "." ""

/-- **The WHOLE optimizer stage for a net: the hoisted global-norm clip, then `optOne` per
    parameter.** Returns `(code, θ', m', v', G', E')`, with `G'` empty unless the optimizer
    accumulates and `E'` empty unless `ema`. The two are INDEPENDENT — see `VerifiedVariant.nRegions`
    for why they used to be one slot and what that cost RSB-A2/A1.

    ⚠⚠ **THIS IS A FUNCTION SO THAT THE ONE-STEP GATE CAN DRIVE THE SHIPPED PATH.** It was inline in
    `ResNet50RenderB.resnet50TrainStepFaithfulB` until 2026-08-14, which meant the only way to
    exercise the clip numerically was to render a whole 161-parameter train step and run a forward.
    `planning/verified_optimizer_parity.md` §5's gate — one step of each optimizer on the same
    `(θ, g, state)` — needs the optimizer stage ALONE, and a second copy of it written for the gate
    would gate a transcription rather than the emission (§5's own point, one level down: *a gate on
    a copy is not a gate on the thing copied*). `tests/TestOptStepFixtures.lean` calls exactly this.
    ⭐ Pure refactor: every committed R50 artifact re-renders byte-identically.

    ⚠⚠ **THE ORDER IS THE SEMANTICS, and there are two orderings to get right, not one.**

      ① the clip goes AFTER the `all_reduce`. Each replica holds a PARTIAL gradient; clipping those
         and then averaging is a clip of nothing in particular. `optOne` all-reduces per parameter,
         so under the clip the collective is hoisted here and `optOne` is told (`preAvg`) not to
         repeat it.
      ② the clip goes AFTER the ACCUMULATION. The reference is explicit
         (`jax/Jax/Codegen.lean:2439`): `grads = _gsum / _K` and only THEN the clip line, so the
         norm is of the MEAN over the k micro-batches. Clipping the micro-batch gradient instead
         would clip k times per optimizer step against a threshold meant for their mean — again
         something that trains and descends. So the accumulator is hoisted here too, and the
         threshold moves to `k·C` to read the fold on `Gt` as a fold on `Gt/k` (`clipNormStr`).

    ▶ Neither ordering is visible to a gate that only checks "the gradients got smaller", which is
    why `Proofs.clipFactor_shared` is the statement to drive and why it must be driven in the
    CLIPPING regime — the identity-below-threshold gate is structurally blind to placement.

    ⚠ At `gradClip := false` NOT ONE `pretty` CALL happens in the clip block, so the fresh-name
    counter does not move and every committed artifact re-renders byte-identically. -/
def optAllParams (opt : R34Opt) (B replicas : Nat) (ps : List PGrad)
    (wdExclude : Bool := false) (gradClip : Bool := false) (clipNorm : Float := 1.0)
    -- ▶▶ **`ema` — the model-EMA shadow region**, threaded straight to `optOne` (2026-08-27).
    -- TRAILING and defaulted, so every committed R50 artifact re-renders byte-identically and the
    -- one-step gate's existing call is unchanged.
    -- ⚠ It is INDEPENDENT of the accumulator: `G` and `E` are two regions, not one slot two
    -- features share, which is the whole point of the change (`VerifiedVariant.nRegions`).
    (ema : Bool := false) :
    StateM Nat (String × List String × List String × List String × List String × List String) := do
  let accOn := match opt with | .adamwAccum _ => true | .lambAccum _ => true | _ => false
  let z1 : Vec 1 := fun _ => 0
  let accK := optAccumK opt
  let mut clipCode := ""
  let mut clipped : List (String × String) := []
  let mut accRaw : List (String × String) := []
  if gradClip then
    -- ① average across replicas first. At `replicas ≤ 1` this emits nothing and threads the name
    -- through, so the single-device clip render carries no collective at all.
    let mut avg : List (String × String) := []
    for g in ps do
      let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
      clipCode := clipCode ++ arS
      avg := avg ++ [(g.nm, gAvg)]
    -- ② accumulate, when the optimizer accumulates. `Gt = akeep·G + g`, the SAME `momVNextF`
    -- instantiation `optOne` would have emitted — moved, not duplicated, and handed back to it by
    -- name so the fourth region still reports the RAW total.
    -- ⚠ On an ACCUMULATE micro-batch the clip is computed on a PARTIAL `Gt` and then discarded:
    -- `%lr = 0` freezes θ and `%b1 = %b2 = 1` / `%ob1 = %ob2 = 0` make both moments exact
    -- passthroughs, so only the APPLY micro-batch's factor — the one taken on the full sum — can
    -- reach a weight. That is what makes ② expressible without a second buffer.
    let mut src : List (String × String) := avg
    if accOn then
      let mut acc : List (String × String) := []
      for g in ps do
        let n := g.ds.foldl (· * ·) 1
        let z : Vec n := fun _ => 0
        let gAvg := (avg.lookup g.nm).getD g.grad
        let (cG, nG) ← pretty B (.momVNextF s!"%{g.nm}a" "%akeep" g.ds 0 z (.operand gAvg z))
        clipCode := clipCode ++ cG
        acc := acc ++ [(g.nm, nG)]
      accRaw := acc
      src := acc
    -- ③ ONE scalar, folded across every parameter before any of them is scaled. The fold must run
    -- to completion first — that is the global-vs-local distinction, made structural by
    -- `Proofs.clipScale` taking the factor as a PARAMETER it cannot compute from its own tensor.
    let mut total : SHlo 1 := .operand "%czero" z1
    for g in ps do
      let n := g.ds.foldl (· * ·) 1
      let gS := (src.lookup g.nm).getD g.grad
      total := .gradSumSqAccF (n := n) g.ds total (.operand gS (fun _ => 0))
    let (cN, normSSA) ← pretty B total
    clipCode := clipCode ++ cN
    -- ④ scale each parameter's gradient by that one shared factor.
    for g in ps do
      let n := g.ds.foldl (· * ·) 1
      let z : Vec n := fun _ => 0
      let gS := (src.lookup g.nm).getD g.grad
      let (cS, sSSA) ← pretty B (.clipScaleF (n := n) (clipNormStr clipNorm accK) (clipEpsStr accK)
                          0 0 g.ds (.operand normSSA z1) (.operand gS z))
      clipCode := clipCode ++ cS
      clipped := clipped ++ [(g.nm, sSSA)]
  -- ═══ the optimizer: one proven triple per parameter ═══
  let mut code := clipCode
  let mut thetaN : List String := []
  let mut mNames : List String := []
  let mut vNames : List String := []
  let mut aNames : List String := []
  let mut eNames : List String := []
  for g in ps do
    -- ⚠ The decay operand comes from the SAME `PGrad` that names the site, so the parameter whose
    -- shape decides exclusion is the parameter being updated — §2e's slot rule. Reading the shape
    -- off a parallel list is how this class of bug ships.
    -- ⚠ Under the clip the gradient `optOne` consumes is the CLIPPED one, while `accIn` names the
    -- unclipped accumulator it must still return: see `optOne`'s note for why those two names
    -- cannot collapse into one.
    let gIn : PGrad :=
      if gradClip then { g with grad := (clipped.lookup g.nm).getD g.grad } else g
    let (c, nT, nM, nV, nA, nE) ← optOne opt B replicas gIn (r34WdName wdExclude g.nm g.ds)
                                (preAvg := gradClip) (accIn := accRaw.lookup g.nm) (ema := ema)
    code := code ++ c
    thetaN := thetaN ++ [nT]
    mNames := mNames ++ [nM]
    vNames := vNames ++ [nV]
    -- ⭐ The accumulator's output name, present only under the accumulating constructors. It becomes
    -- the FOURTH region of the packed blob — the same shape the EMA renders use, so the driver's
    -- `nRegions = 4` path is reused rather than a second one being written.
    match nA with | some a => aNames := aNames ++ [a] | none => pure ()
    -- ⭐ The shadow's output name, present only under `ema`. It becomes the FIFTH region — AFTER
    -- the accumulator, never instead of it. Region order `[θ|m|v|G|E]` is what keeps every blob
    -- written before the fifth region existed readable at the index it was written at.
    match nE with | some e => eNames := eNames ++ [e] | none => pure ()
  pure (code, thetaN, mNames, vNames, aNames, eNames)

/-- The optimizer's baked constants. `.adamw` is byte-for-byte the committed block; `.heavyBall`
    emits only what it reads, so there are no dead constants in the momentum artifact.

    `%wd` is baked rather than a runtime arg because weight decay is not scheduled — unlike `%lr`,
    which stays a `tensor<f32>` argument so one graph serves the whole cosine schedule. -/
def optConstsB (opt : R34Opt) (wdStr : String := "") : String :=
  -- ⚠ ONE binding, used by every arm below, so the two families cannot drift apart in how they
  -- honour the override — `optWdStr` owns "the caller's value or this optimizer's default".
  let wd := optWdStr opt wdStr
  match opt with
  | .adamw =>
    "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
    "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
    "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
    "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  | .heavyBall =>
    "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n"
  | .lamb =>
    -- ⚠ **`%eps` is 1e-6, NOT AdamW's 1e-8**, and `%wd` is 0.02, NOT 1e-4. Both come off timm's a3
    -- arg string (`lamb-cosine-lr0.008-wd0.02-…`), which `jax/MainResnet50Imagenet.lean` decodes in
    -- its own comment. Reusing AdamW's numbers here would render a LAMB that is structurally right
    -- and 200x off on the decay.
    -- ⚠ `%lzero` seeds each parameter's OWN norm fold, one leaf deep. The clip's `%zero` seeds a
    -- fold across ALL parameters. Same op, and the seed placement is the whole difference.
    "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
    "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
    "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
    "    %eps = stablehlo.constant dense<1.0e-6> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n"
  | .adamwAccum k =>
    -- ⚠ **A TRUSTED CARVE-OUT, and deliberately the smallest one that does the job**: eight lines of
    -- SCALAR arithmetic emitted ONCE, next to the constants that were already emitted text. The 161
    -- per-parameter tails stay `pretty(verified AST node)` and are byte-identical to `.adamw`'s.
    --
    -- `%aup ∈ {0, 1}` is the APPLY flag, supplied per micro-batch by the driver. It selects between
    -- the two phases by arithmetic rather than by two artifacts — one graph, one compile, one
    -- resident parameter set, and no way for an "accumulate" and an "apply" render to drift:
    --
    --     accumulate (%aup = 0):  β₁ = 1, (1−β₁) = 0  ⇒  m' = m,  v' = v   exactly
    --     apply      (%aup = 1):  β₁ = 0.9, (1−β₁)/k  ⇒  m' = 0.9·m + (1−β₁)·(Gt/k)
    --
    -- ⭐ `1/k` is folded in HERE, and asymmetrically: `%ob1` carries `1/k` while `%ob2` carries
    -- `1/k²`, because `v` consumes the gradient SQUARED. `v' = β₂v + ((1−β₂)/k²)·Gt² =
    -- β₂v + (1−β₂)·(Gt/k)²` — the identity that makes accumulation equal a real large-batch step
    -- rather than the "mean of per-micro-batch second moments" a naive implementation produces.
    --
    -- ⚠ `fmt12`, not `fmt6` — see `accumScalarConsts`, which now owns those eight lines so that
    -- `.lambAccum` emits the SAME mechanism rather than a second copy of it.
    "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    accumScalarConsts k
  | .lambAccum k =>
    -- ⭐⭐ **THE COMPOSITION, AND IT IS EXACTLY "LAMB's CONSTANTS + THE SHARED ACCUMULATOR".**
    -- ⚠ `%eps` 1e-6 and `%wd` 0.02 are LAMB's, off timm's a3 arg string — NOT AdamW's 1e-8/1e-4.
    -- Reusing AdamW's numbers here would render a LAMB that is structurally right and 200× off on
    -- the decay, which is the exact trap `.lamb`'s own comment records.
    -- ⚠ `%lzero` seeds each parameter's OWN norm fold, one leaf deep — carried over from `.lamb`
    -- unchanged, because accumulation does not touch the trust ratio.
    -- ▶ `%b1`/`%ob1`/`%b2`/`%ob2` are COMPUTED from `%aup`, not baked, which is the only difference
    -- from `.lamb`'s constant block and is precisely what `accumScalarConsts` supplies.
    "    %eps = stablehlo.constant dense<1.0e-6> : tensor<f32>\n" ++
    s!"    %wd = stablehlo.constant dense<{wd}> : tensor<f32>\n" ++
    "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
    accumScalarConsts k

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/resnet34_<variant>_train_step.mlir`, the entry point is
    `@resnet34_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it.

    All three must agree. The shim checks the entry name and refuses a mismatch outright ("entry
    mismatch") rather than running the wrong graph — which is exactly what it did the first time the
    DP render kept the single-device name (§2b-quater). Deriving the name here, from the same two
    numbers the render is built from, is what stops it drifting from the `#eval` paths below; the
    `#guard`s at the bottom pin those literal paths against this function.

    `B = 32` is deliberately unsuffixed, so the two existing artifacts keep their names and bytes. -/
def r34AdamVariant (B replicas : Nat) (opt : R34Opt := .adamw)
    -- ▶ `wx` = timm `no_weight_decay`. TRAILING and defaulted, so every existing spelling is
    -- unchanged. ⚠ It must reach this function: R50 DERIVES its entry name from the variant, so a
    -- flag that reached the renderer but not here produces an artifact whose declared entry
    -- disagrees with its own path — the shim then refuses the call outright. ConvNeXt shipped
    -- exactly that defect twice (`wx`, then `clip`); the `#guard`s below pin every spelling.
    -- ⚠ It needs NO driver predicate: excluding a parameter changes no arity, type or region.
    (wdExclude : Bool := false)
    -- ▶ `clip` = timm `Lamb.max_grad_norm` (D1), the GLOBAL-norm gradient clip. TRAILING and
    -- defaulted, exactly as `wx` is, so every existing spelling is unchanged.
    -- ⚠ It must reach THIS function and not merely the renderer — the same rule `wx` states above,
    -- and `cnxAdamVariant`'s docstring records ConvNeXt shipping that defect twice (once for `wx`,
    -- once for `clip`). R50 derives its entry name from this string, so a flag that stopped at the
    -- emission would produce an artifact whose declared entry disagrees with its own path and the
    -- shim would refuse the call outright.
    -- ⚠ Like `wx` it needs NO driver predicate: the clip adds ops, not arity, types or regions.
    (gradClip : Bool := false)
    -- ▶▶ `bce` = BCE-with-logits instead of smoothed CE — RSB-A2/A3's loss, and the reason the
    -- recipe's lr is what it is. ⚠⚠ **IT IS A PARAMETER HERE BECAUSE IT USED NOT TO BE**
    -- (`a3_paper_fidelity.md` §3.3, closed 2026-08-14): `resnet50TrainStepFaithfulB` carried a
    -- `bce : Bool` that swapped the loss AND a separate `vSuffix : String` that the caller had to
    -- spell `"bce"` by hand, so the FLAG and the NAME were two writers for one fact and could
    -- disagree with nothing noticing — on the artifact the 77.43% run depends on. Measured before
    -- removing it: 12 call sites passed `bce := true`, 12 passed `vSuffix := "bce"`, and no call
    -- site ever passed a `vSuffix` that was not `"bce"`. So the string was a restatement of the
    -- Bool, and deriving it here makes the disagreement unspellable rather than merely unobserved.
    -- ⭐ Pure refactor: `{variant}{vSuffix}` and `{variant-with-bce}` are the same characters, so
    -- every committed artifact keeps its name and its bytes.
    -- ⚠ It TRAILS `wx` and `clip`, which is the order the hand-passed suffix already produced —
    -- `lambaccdp8x64wxclipbce`. Preserved deliberately; the `#guard`s below pin it.
    (bce : Bool := false)
    -- ▶▶ A NON-DEFAULT weight decay, and it must reach the name for the reason `wdVariantMark`
    -- gives: `%wd` is BAKED, so two renders differing only in it would otherwise collide on one
    -- artifact path. Empty = the optimizer's own default = no marker = every committed name
    -- unchanged. ⚠ It goes LAST because it is the newest axis and §2m's rule is unconditional.
    (wdStr : String := "")
    -- ▶ `bf16` LAST, after even the decay marker, for that marker's own reason: it is the newest
    -- axis and appending is the only placement that leaves every existing spelling untouched.
    -- ⚠ It MUST reach this function and not merely the renderer — the `wx`/`clip` rule three
    -- parameters up, which ConvNeXt shipped wrong twice. And it must not collide with the driver's
    -- variant predicates: `momdp64bf16` contains no "acc", no "ema", and no "do", so `accOn`/
    -- `emaOn`/`cdOn` all stay false and the region/scalar counts are unchanged (checked).
    (bf16 : Bool := false)
    -- ▶▶ **`ema` — the model-EMA shadow, and it is the ONLY marker that LEADS** (2026-08-27).
    -- ⚠⚠ **PREFIX, NOT SUFFIX, AND THAT IS FORCED BY THE DRIVER**: `VerifiedVariant.emaOn` is
    -- `startsWith "ema"` while `accOn` is a substring test, so `lambaccdp8x64wxclipbceema` would
    -- read as accumulation-only — four regions packed into a five-region graph, i.e. every
    -- parameter misaligned, with no error anywhere. `tests/TestVariantPredicates.lean` pins both
    -- directions.
    -- ⚠ It is LAST in this signature and FIRST in the string, which is the one place in this
    -- function where those two orders disagree. Parameter position is "newest axis appends" (§2m);
    -- string position is the driver's predicate. Defaulted, so every committed spelling is
    -- unchanged — `ema` prepends nothing at `false`.
    (ema : Bool := false) : String :=
  (if ema then "ema" else "") ++
  (match opt with
   | .adamw     => if replicas ≤ 1 then "adam" else "adamdp"
   | .heavyBall => if replicas ≤ 1 then "mom"  else "momdp"
   -- ⭐ `k` is IN THE NAME — `acc4x64`, `accdp4x64` — because the driver has to know it (to decide
   -- which micro-batch applies) and the graph has it baked (in `%ob1`/`%ob2`). Two places, and a
   -- disagreement between them is silent: the run would apply on a cadence the `1/k` does not
   -- match, i.e. a wrong effective learning rate with no error anywhere. Carrying `k` in the
   -- artifact NAME makes the driver read it off the same string that selects the file.
   | .lamb      => if replicas ≤ 1 then "lamb" else "lambdp"
   | .adamwAccum k => (if replicas ≤ 1 then "acc" else "accdp") ++ toString k ++ "x"
   -- ⚠⚠ `lamb` ++ `acc` — the marker is NO LONGER LEADING, and that is what broke the driver's
   -- `startsWith "acc"` predicate (defect #4 in `tests/TestVariantPredicates.lean`). The name is
   -- spelled this way rather than as `accdp8x64lamb` because the OPTIMIZER is the primary axis and
   -- every other variant leads with it; the fix belongs in the predicate, not in the spelling.
   -- ⚠ `dp` goes INSIDE, right after `acc`, matching `.adamwAccum`'s placement exactly — so the
   -- `k` parse is one rule for both, not two.
   | .lambAccum k => (if replicas ≤ 1 then "lambacc" else "lambaccdp") ++ toString k ++ "x") ++
  (if B == 32 then "" else toString B) ++
  -- ▶ `wx` TRAILS THE BATCH, so it composes with every optimizer spelling and with the `clip` and
  -- `bce` markers appended after it — `lambaccdp8x64bcewx` would be wrong; the order below gives
  -- `lambaccdp8x64wxclipbce`. ⚠ The order is a choice and
  -- the `#guard`s below are what make it a fixed one, because `cnxAdamVariant` learned the hard way
  -- that a marker's POSITION is as load-bearing as its presence.
  (if wdExclude then "wx" else "") ++
  -- ▶ `clip` TRAILS `wx`, which is `cnxAdamVariant`'s order (`wx` ++ `clip`) followed deliberately
  -- rather than re-chosen — the two nets spell the same two flags, and one rule for both is what
  -- keeps a reader from having to know which net a slug came from. With R50's `bce` appended after,
  -- the RSB-A3 composition reads `lambaccdp8x64wxclipbce`. ⚠ The order is a CHOICE; the `#guard`s
  -- below are what make it a fixed one.
  (if gradClip then "clip" else "") ++
  -- ▶ `bce` LAST, which is where the hand-passed `vSuffix` put it. See this parameter's note.
  (if bce then "bce" else "") ++
  -- ▶ …and the decay marker after even that, because it is the newest axis and appending is the
  -- only placement that leaves all four existing spellings untouched. Empty at the default.
  wdVariantMark opt wdStr ++
  (if bf16 then "bf16" else "")

set_option maxRecDepth 4000000 in
/-- **ResNet-34 `[3,4,6,3]` AdamW train step, batch-BN, rendered from the verified AST at `N := B`.**
    515 inputs (`%x`, 146 θ, 146 m, 146 v, `%lr`/`%bc1`/`%bc2`, 72 running-stat slots, `%onehot`)
    and 513 outputs (146 θ', 146 m', 146 v', `%loss`/`%bc1`/`%bc2`, 72 batch stats) — the interface
    `tests/TestResnet34Train.lean`'s hand-written render already presents, so the driver is
    unchanged. Parameter ORDER comes from `r34SigList`, the same single source the per-example
    render and both forwards use, so the arity/order contract cannot drift between them. -/
def resnet34AdamTrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (opt : R34Opt := .adamw) (slug : String := "resnet34")
    (convBias : Bool := false)
    -- ▶ TRAILING and defaulted, so every existing render is byte-identical (gate 1).
    (bf16 : Bool := false) : String :=
  let optLabel : String := match opt with
    | .adamw     => "AdamW"
    | .heavyBall => "heavy-ball momentum + coupled L2"
    -- ⚠ R34 renders no accumulation artifact — `resnet34TrainStepFaithfulB` has no fourth region in
    -- its signature, so passing `.adamwAccum` here would emit an optimizer that reads `%<p>a` inputs
    -- the function does not declare. The renderer REFUSES rather than emitting invalid MLIR that
    -- `iree-compile` would report as an undefined-value error a hundred lines from the cause.
    | .lamb      => "LAMB"
    | .adamwAccum k => panic! s!"resnet34TrainStepFaithfulB: .adamwAccum {k} needs a fourth \
parameter region and R34's signature has three — render it from ResNet50RenderB, or add the region \
here first"
    -- ⚠ Same refusal, same reason: the fourth region is a property of the SIGNATURE, not of which
    -- optimizer consumes the accumulator, so `.lambAccum` is no more renderable here than
    -- `.adamwAccum`. ▶ This arm exists because the match is exhaustive — adding the constructor
    -- without it is a build error, which is how the type caught this site rather than a run doing so.
    | .lambAccum k => panic! s!"resnet34TrainStepFaithfulB: .lambAccum {k} needs a fourth \
parameter region and R34's signature has three — render it from ResNet50RenderB, or add the region \
here first"
  let go : StateM Nat String := do
    -- ═══ stem: 7×7/s2 conv → batch BN → relu → 2×2 maxpool ═══
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*112*112)) := fun _ => 0
    let z112b : Vec (B*(64*(112*112))) := fun _ => 0
    let z56   : Vec (B*(64*56*56)) := fun _ => 0
    -- Placeholder rounding, as `zSk`/`z64` are placeholders — see `idFwdB`.
    let zrnd  : ℝ → ℝ := fun r => r
    let (cStc, nStc) ← pretty B (.batchOp (N := B) (if bf16 then .convStridedBf16 (h := 112) (w := 112) zrnd "%sW" (biasName convBias "%sbi" 64) zSk z64 else .convStrided (h := 112) (w := 112) "%sW" (biasName convBias "%sbi" 64) zSk z64) (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*112*112)) (.operand nStn z112))
    -- ⭐ He et al.'s 3×3/s2 stem pool. ⚠ This read `.maxPool` (2×2, non-overlapping) until
    -- 2026-08-04 — a different function at the identical 112→56 output shape, so nothing ever
    -- failed. `planning/rsb_a3_r50_verified.md` §4b.
    let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool3s2 (c := 64) (h := 56) (w := 56)) (.operand nStr z112))
    -- ═══ 16 blocks ═══
    let f1  ← idFwdB   B 64 56 epsStr "s1b0" nStp convBias bf16
    let f2  ← idFwdB   B 64 56 epsStr "s1b1" f1.o convBias bf16
    let f3  ← idFwdB   B 64 56 epsStr "s1b2" f2.o convBias bf16
    let f4  ← downFwdB B 64 128 28 epsStr "d2" f3.o convBias bf16
    let f5  ← idFwdB   B 128 28 epsStr "s2b0" f4.o convBias bf16
    let f6  ← idFwdB   B 128 28 epsStr "s2b1" f5.o convBias bf16
    let f7  ← idFwdB   B 128 28 epsStr "s2b2" f6.o convBias bf16
    let f8  ← downFwdB B 128 256 14 epsStr "d3" f7.o convBias bf16
    let f9  ← idFwdB   B 256 14 epsStr "s3b0" f8.o convBias bf16
    let f10 ← idFwdB   B 256 14 epsStr "s3b1" f9.o convBias bf16
    let f11 ← idFwdB   B 256 14 epsStr "s3b2" f10.o convBias bf16
    let f12 ← idFwdB   B 256 14 epsStr "s3b3" f11.o convBias bf16
    let f13 ← idFwdB   B 256 14 epsStr "s3b4" f12.o convBias bf16
    let f14 ← downFwdB B 256 512 7 epsStr "d4" f13.o convBias bf16
    let f15 ← idFwdB   B 512 7 epsStr "s4b0" f14.o convBias bf16
    let f16 ← idFwdB   B 512 7 epsStr "s4b1" f15.o convBias bf16
    -- ═══ head: GAP(7×7) → dense(512→nClasses) ═══
    let zL    : Vec (B*(512*7*7)) := fun _ => 0
    let z512  : Vec (B*512) := fun _ => 0
    let zWd   : Mat 512 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp  : Vec (B*nClasses) := fun _ => 0
    let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 512) (h := 7) (w := 7)) (.operand f16.o zL))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z512))
    -- ═══ label-smoothed softmax-CE cotangent, COMPOSED from kit ops (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. Every line is a verified node;
    --     the hand-written render fuses this into one [B,K] block, so the two graphs differ. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← pretty B (.scaleB "0.100000" 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← pretty B (.shiftB s!"-{alphaOverK nClasses}" 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB s!"{B}.0" 0 (.operand nD2 zNCb))
    -- ═══ head backward + dense grads ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) (a := 512) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWd,  nWd)  ← pretty B (.denseWeightGradB (c := nClasses) nGap z512 (.operand nDy zNCp))
    let (cbd,  nbd)  ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 512) (h := 7) (w := 7) (.operand nDgi z512))
    -- ═══ 16 block backwards ═══
    let b16 ← idBackGradB   B 512 7 epsStr "s4b1" f16 nDgp convBias bf16
    let b15 ← idBackGradB   B 512 7 epsStr "s4b0" f15 b16.dx convBias bf16
    let b14 ← downBackGradB B 256 512 7 epsStr "d4" f14 b15.dx convBias bf16
    let b13 ← idBackGradB   B 256 14 epsStr "s3b4" f13 b14.dx convBias bf16
    let b12 ← idBackGradB   B 256 14 epsStr "s3b3" f12 b13.dx convBias bf16
    let b11 ← idBackGradB   B 256 14 epsStr "s3b2" f11 b12.dx convBias bf16
    let b10 ← idBackGradB   B 256 14 epsStr "s3b1" f10 b11.dx convBias bf16
    let b9  ← idBackGradB   B 256 14 epsStr "s3b0" f9  b10.dx convBias bf16
    let b8  ← downBackGradB B 128 256 14 epsStr "d3" f8 b9.dx convBias bf16
    let b7  ← idBackGradB   B 128 28 epsStr "s2b2" f7 b8.dx convBias bf16
    let b6  ← idBackGradB   B 128 28 epsStr "s2b1" f6 b7.dx convBias bf16
    let b5  ← idBackGradB   B 128 28 epsStr "s2b0" f5 b6.dx convBias bf16
    let b4  ← downBackGradB B 64 128 28 epsStr "d2" f4 b5.dx convBias bf16
    let b3  ← idBackGradB   B 64 56 epsStr "s1b2" f3 b4.dx convBias bf16
    let b2  ← idBackGradB   B 64 56 epsStr "s1b1" f2 b3.dx convBias bf16
    let b1  ← idBackGradB   B 64 56 epsStr "s1b0" f1 b2.dx convBias bf16
    -- ═══ stem backward: maxpool-back → relu mask → BN back, then the 4 stem grads ═══
    let (cDmp, nDmp) ← pretty B (.maxPool3s2BackB (N := B) (c := 64) (h := 56) (w := 56) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 64) (h := 112) (w := 112) "%sg" nStc epsStr 0 z64 z112b (.operand nDsr z112b))
    let (csW, nsW) ← pretty B (if bf16 then .convStridedWeightGradBBf16 zrnd "%x" z64 zx zSk (.operand nDsn z112) else .convStridedWeightGradB "%x" z64 zx zSk (.operand nDsn z112))
    let (csb, nsb) ← if convBias then
        pretty B (.convStridedBiasGradB (h := 112) (w := 112) zSk zx z64 (.operand nDsn z112))
      else pure ("", "")
    let (csg, nsg) ← pretty B (.bnGammaGradB nStc epsStr 0 z112b (.operand nDsr z112b))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 64) (h := 112) (w := 112) (.operand nDsr z112b))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × String × String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      pure (cM ++ cV, nM, nV)
    let idStats (oc hh : Nat) (f : BFwdB) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1
      let (c2, m2, v2) ← bnStat oc hh f.c2
      pure (c1 ++ c2, [m1, v1, m2, v2])
    let downStats (oc hh : Nat) (f : BFwdB) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1
      let (c2, m2, v2) ← bnStat oc hh f.c2
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (c1 ++ c2 ++ cp, [m1, v1, m2, v2, mp, vp])
    let (cSt0, st0) ← bnStat 64 112 nStc
    let (cSt1, st1) ← idStats 64 56 f1
    let (cSt2, st2) ← idStats 64 56 f2
    let (cSt3, st3) ← idStats 64 56 f3
    let (cSt4, st4) ← downStats 128 28 f4
    let (cSt5, st5) ← idStats 128 28 f5
    let (cSt6, st6) ← idStats 128 28 f6
    let (cSt7, st7) ← idStats 128 28 f7
    let (cSt8, st8) ← downStats 256 14 f8
    let (cSt9, st9) ← idStats 256 14 f9
    let (cSt10, st10) ← idStats 256 14 f10
    let (cSt11, st11) ← idStats 256 14 f11
    let (cSt12, st12) ← idStats 256 14 f12
    let (cSt13, st13) ← idStats 256 14 f13
    let (cSt14, st14) ← downStats 512 7 f14
    let (cSt15, st15) ← idStats 512 7 f15
    let (cSt16, st16) ← idStats 512 7 f16
    -- ═══ the 146 parameter gradients in func-arg order ═══
    let stemPs : List PGrad :=
      [⟨"sW", nsW, [64,3,7,7]⟩] ++ (if convBias then [⟨"sbi", nsb, [64]⟩] else []) ++
      [⟨"sg", nsg, [64]⟩, ⟨"sbt", nst, [64]⟩]
    let headPs : List PGrad := [⟨"Wd", nWd, [512, nClasses]⟩, ⟨"bd", nbd, [nClasses]⟩]
    let allPs : List PGrad := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++
      b9.ps ++ b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ headPs
    -- ═══ AdamW: one proven triple per parameter ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    for g in allPs do
      -- ⚠ R34 renders no accumulation variant and no EMA one, so the fifth and sixth components
      -- (the accumulator's and the shadow's output names) are always `none` here. Dropping them
      -- rather than threading them is what keeps every committed R34 artifact byte-identical.
      let (c, nT, nM, nV, _, _) ← optOne opt B replicas g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
    -- ═══ assemble ═══
    let statCode := cSt0 ++ cSt1 ++ cSt2 ++ cSt3 ++ cSt4 ++ cSt5 ++ cSt6 ++ cSt7 ++ cSt8 ++
      cSt9 ++ cSt10 ++ cSt11 ++ cSt12 ++ cSt13 ++ cSt14 ++ cSt15 ++ cSt16
    let statNames := st0.1 :: st0.2 :: (st1 ++ st2 ++ st3 ++ st4 ++ st5 ++ st6 ++ st7 ++ st8 ++
      st9 ++ st10 ++ st11 ++ st12 ++ st13 ++ st14 ++ st15 ++ st16)
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path. It is NOT
    -- `pretty` of an AST node and says so in the emitted text — the same carve-out
    -- `cifar8_adam_train_step`'s `%loss` takes (§5 of the handoff).
    -- The SMOOTHED cross-entropy, matching the cotangent's soft target:
    --   loss = −(1/B)·Σ_b [ (1−α)·Σ_k onehot·log sm  +  (α/K)·Σ_k log sm ].
    -- Getting this wrong is invisible to every proof in the repo — `%loss` is report-only and on
    -- no gradient path — but the driver logs it and the epoch curve is how a run is judged against
    -- the reference. A first cut here computed PLAIN CE (dropping the (1−α) factor and the α/K
    -- term); the numeric tie caught it as a 0.28% loss disagreement against an otherwise
    -- bit-identical forward. Hand-written emit is not verified emit (§5).
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
    let body := cStc ++ cStn ++ cStr ++ cStp ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++ f16.code ++
      cGap ++ cLog ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWd ++ cbd ++ cDgp ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := (r34StatSigList.map (·.2))
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure <|
      -- With `optLabel = "AdamW"` these are the committed byte sequences character for character;
      -- interpolating a constant changes the source, not the output (gate 1 checks exactly that).
      (if replicas ≤ 1 then
        s!"    // ── ResNet-34 batch-BN {optLabel} train step: every line is pretty(verified AST node) ──\n"
       else
        s!"    // ── ResNet-34 batch-BN {optLabel} train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted\n" ++
        "    // text outside the faithfulness theorems. Each replica evaluates the same tied graph\n" ++
        "    // at the batch it was rendered for; the collective averages that function's gradients\n" ++
        "    // over disjoint equal batches. NOTE this does NOT equal a single-device step at the\n" ++
        "    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design (§10.3b).\n") ++
      zeroBiasPrelude convBias [64, 128, 256, 512] ++ body ++ optConstsB opt ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := r34SigList nClasses convBias
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  let statSig := String.intercalate ", " (r34StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ (r34StatSigList.map (·.2)))
  let inner : String := go.run' 0
  -- The entry name must track the driver's `{slug}_{variant}_train_step` convention, or the shim
  -- refuses the call ("entry mismatch") — which is exactly what it did the first time this was
  -- rendered. `r34AdamVariant` is the single source for the name, the artifact path, and
  -- `LEAN_MLIR_VARIANT`.
  -- ⚠⚠ `bf16` MUST be passed here, not merely to the block renderers. This function's own
  -- docstring three hundred lines up says why, for `wx` and `clip`: the entry NAME is derived
  -- from the variant, so a flag that reaches the emission but not the name produces an artifact
  -- whose declared entry disagrees with its own path, and the driver refuses the call outright
  -- ("entry mismatch: session holds @…_momdp64_train_step, caller asked …_momdp64bf16_…").
  -- ConvNeXt shipped that defect twice; bf16 reproduced it a third time on its first run, which
  -- is what this comment exists to stop happening a fourth.
  let fname := s!"{slug}_{r34AdamVariant B replicas opt (bf16 := bf16)}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/resnet34_adam_train_step.mlir` — the BATCHED (`N := B`) AdamW train
-- step as `pretty(provenGraph)`. B=32, nClasses=10, ε=1e-5. **This is the artifact
-- `resnet34-verified-adam{,-xla}` trains on**, and this `#eval` is its ONLY writer.
--
-- It rendered to a separate `…_b.mlir` while the hand-written emitter in
-- `tests/TestResnet34Train.lean` still owned this path — two writers for one artifact is the
-- last-writer-wins race §2a found. The swap happened once both gates were in:
--
--   * the numeric tie (`resnet34-adam-tie`) — forward bit-exact, backward norm-rel ≤ 2e-6;
--   * the step bench (`resnet34-adam-bench`) — no cost, despite 1.68× the emitted ops, because
--     XLA's CSE collapses the recomputes (handoff §2b-bis).
--
-- The hand-written emitter now renders only the DATA-PARALLEL variant, to its own
-- `…_dp.mlir` path. To re-run the tie against the retired render, recover it with
-- `git show <rev>:verified_mlir/resnet34_adam_train_step.mlir` and pass it as the first argument.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 32 10 "1.0e-05")

-- The DATA-PARALLEL render (handoff §2b-quater), selected at run time by `LEAN_MLIR_VARIANT=adamdp`.
-- Same graph, plus one `all_reduce(add)/N` per parameter gradient before its AdamW triple. This
-- replaces the hand-written `tests/TestResnet34Train.lean` DP emitter, so the certified renderer is
-- now the ONLY writer of both R34 AdamW artifacts.
--
-- `2` is the replica count these are rendered at, and it must match `PJRT_REPLICAS` at run time —
-- the graph bakes `replica_groups`. Re-render here to change it.
#eval IO.FS.writeFile "verified_mlir/resnet34_adamdp_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 32 10 "1.0e-05" 2)

-- The **bs256** render (handoff §2d.1), selected at run time by `LEAN_MLIR_VARIANT=adam256` with
-- `cfg.batchSize := 256`. Batch is worth ~1.8× img/s on this net and bs256 fits on a 7900 XTX;
-- it is also the batch ImageNet wants. `B` is a true parameter of the renderer, so this is the
-- whole change — the graph structure is identical and only the tensor dimensions move.
--
-- It renders to its OWN path rather than re-pointing `resnet34_adam_train_step.mlir`, so the
-- artifact the trainer runs today is untouched and the §2b tie/bench baselines stay valid. Note
-- the eval forwards are still bs32: train at 256 with `LEAN_MLIR_SKIP_EVAL=1`, or re-render them.
--
-- Gated by `resnet34-batch-check` (§2d.1): feeding 8 identical copies of one bs32 batch makes the
-- batch-BN statistics and the mean-CE cotangent identical to the bs32 render's, so the two must
-- produce the SAME step — an exact known-answer check on the re-render, not a tolerance argument.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam256_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 256 10 "1.0e-05")

-- **bs128 × 2 replicas** — the data-parallel render at a real batch (global 256), the EfficientNet
-- `adamdp128` shape (§2e-quater) brought to R34. `B` and `replicas` are both true parameters, so
-- this composes the two `#eval`s above with no new renderer code.
--
-- Why this batch: §2d.1 measured bs256 at **1.78× img/s** over bs32 single-device, most of it
-- amortising the ~272 MB `[θ|m|v]` host↔device round trip over 8× the images — and that transfer is
-- exactly what the DP path pays per replica per step (§2c). So 2×128 is where the batch win and the
-- replica win stack rather than fight.
--
-- ⚠ **The eval forwards are still bs32**, so this variant needs `LEAN_MLIR_SKIP_EVAL=1` — it yields
-- descent and throughput, NOT a validation accuracy. Same caveat `adam256` carries.
#eval IO.FS.writeFile "verified_mlir/resnet34_adamdp128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 128 10 "1.0e-05" 2)

-- **bs64, SINGLE device** — the controlled peer of `adamdp` (bs32 x 2 = global 64). Same global
-- batch, same 147 steps/epoch, same schedule; the ONLY difference is where the BatchNorm statistics
-- come from — 64 examples here, 32-per-replica there.
--
-- That is the §10.3b caveat as an experiment rather than an argument. The handoff asserts
-- everywhere that `2 x 32 != 1 x 64` under batch BN and uses it to explain why R34's collective
-- cannot be gated by splitting a batch (hence the cifar8 proxy). Nothing had ever measured how much
-- it is worth in accuracy. Run both and subtract; with step count and global batch held fixed, the
-- residual IS the BN-splitting effect.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam64_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 64 10 "1.0e-05")

-- **bs128, SINGLE device** — the fourth point on the batch/step-count curve (global 128, 73
-- steps/epoch), and the single-device peer of `adamdp128` (bs128 x 2 = global 256). Together with
-- `adam`, `adam64` and `adamdp128` this brackets the step count 295 / 147 / 73 / 36 at a fixed
-- 80-epoch budget and unscaled LR, which is what isolates "fewer optimizer steps" from every other
-- moving part.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 128 10 "1.0e-05")

-- **HEAVY-BALL MOMENTUM + coupled L2** — the optimizer `jax/MainResnetImagenet.lean` actually uses,
-- so the verified/XLA path and the Lean→JAX reference can be run as a matched pair rather than as
-- two different experiments. Selected by `LEAN_MLIR_VARIANT=mom`.
--
-- The reference rule, from `Jax/Codegen.lean`'s `.sgd` branch at `hasMomentum`:
--     grads    = g + WD * p          -- COUPLED L2, wd = 1e-4, every param (no wdExclude)
--     velocity = MOMENTUM * v + g    -- μ = 0.9
--     params   = p - lr * velocity   -- heavy-ball
--
-- ⚠ **This is NOT `momParamF`, and that trap is the reason to read `optOne` before editing here.**
-- The `SgdMomentumStep` family is **Nesterov** (`θ − lr·(g + μ·v')`); the reference steps by `v'`
-- alone. `Proofs.momParam_heavyBall_diff` states the exact difference. Reaching for the
-- momentum-named op would have compiled, rendered, trained, and produced a *different optimizer*
-- than the thing this artifact exists to be 1:1 with — silently.
--
-- Rendered at **bs32 / 10 classes**, i.e. the direct peer of `adam`, ON PURPOSE: that is the shape
-- the existing gates and the Imagenette trainer can exercise today. The ImageNet shape is
-- `B := 256, nClasses := 1000` and is one more `#eval` whenever the streaming loader lands — `B`
-- and `nClasses` are both true renderer parameters, so no renderer change is involved.
--
-- ⚠ **Not yet numerically gated.** Gate 1 held (all six `adam*` artifacts re-render byte-identical
-- through the `opt` threading) and the interface matches positionally, but the exact known-answer
-- check is still owed: from `m = v = 0`, the `adam` render's `m' = (1−β₁)·g` recovers the gradient
-- exactly (`g = 10·m'`), so `v'` here must equal `g + wd·θ` and `θ'` must equal `θ − lr·v'` on
-- shared inputs. That is a cross-render known answer, not a tolerance argument — the `shard-check`
-- construction, reused. Until it is run, say "rendered", not "certified".
#eval IO.FS.writeFile "verified_mlir/resnet34_mom_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 32 10 "1.0e-05" 1 .heavyBall)

-- ══ ImageNet-1k: the three artifacts the reference-pairing run needs (handoff §2k) ══
--
-- `B := 256, nClasses := 1000`, heavy-ball — i.e. the `jax/MainResnetImagenet.lean` recipe, on the
-- certified renderer. **No renderer change was needed for any of this**: `B`, `nClasses`, `opt` and
-- `slug` are all parameters, which is the whole reason this is three `#eval`s and not a project.
--
-- ⚠ **The slug is `resnet34in`, NOT `resnet34`, and that is load-bearing.** The forward artifacts
-- carry no variant in their path (`<slug>_fwd.mlir`), so rendering a 1000-class forward under the
-- `resnet34` slug would OVERWRITE the 10-class Imagenette one that five committed runs and the
-- prefix audit depend on — silently, and with a graph of a different arity. A distinct slug is what
-- keeps the two nets' artifacts disjoint; `slug` defaults to "resnet34" so every existing render is
-- byte-identical (checked).
--
-- Batch 256 on the forwards too, matching the shim's val batch: 195 batches × 256 = 49,920 images
-- after tfds `drop_remainder`, which is the count the JAX reference reports scoring.
#eval IO.FS.writeFile "verified_mlir/resnet34in_mom256_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 256 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.heavyBall "resnet34in")

-- **The 4-GPU data-parallel peer**, at `B := 64` PER REPLICA so the GLOBAL batch is 64×4 = 256 —
-- the same global batch, the same 5004 steps/epoch and the same recipe as the single-device
-- `mom256` render above, and the same batch the JAX reference trains at (4×64, see
-- `jax/runs/r34_imagenet_bf16_90ep/RESULTS.md`). That is deliberate: rendering 256 per replica
-- would make the global batch 1024 and silently change the recipe, so the run would no longer be
-- comparable to either peer without an LR rescale. Matching the reference is what makes the
-- resulting wall-clock a like-for-like number rather than a new experiment.
--
-- Nothing in the renderer changed for this: `optOne` already takes `replicas` and already calls
-- `emitGradAllReduce`, exactly as §2h-bis found for mnv2. This is one `#eval`.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp64_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.heavyBall "resnet34in")

-- ⭐⭐ **The bf16 peer of the render above** — `momdp64bf16`, byte-for-byte the same graph except
-- that every conv (stem, both block convs, the 1×1 projection, both dgrads, every wgrad) is its
-- bf16 twin: bf16 operands, a **bf16-TYPED** convolution result, then a convert back to f32.
-- Everything else — BN, the residual adds, the loss, the heavy-ball tail, the master weights —
-- stays f32, which is what `jax/MainResnetImagenet.lean` does and why its bf16 arm converges to
-- the same place as its f32 arm.
--
-- ⚠ The bf16-typed RESULT is the load-bearing part and is not cosmetic. A conv with bf16 operands
-- and an f32 result has its converts deleted by XLA under excess precision — cuDNN then gets f32
-- parameters and the graph runs entirely in fp32 while still *reading* as mixed precision. This
-- is measured, not feared; `BatchableOp.convBf16` carries the note. Verify with the operand dtypes
-- in the OPTIMIZED HLO, never by grepping the op line, which shows only the result type.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp64bf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.heavyBall "resnet34in" false true)

-- **The 2-GPU data-parallel peer**, at `B := 128` PER REPLICA so the global batch is 128×2 = 256 —
-- the same global batch, the same 5004 steps/epoch and the same recipe as `mom256` and `momdp64`
-- above. That is the whole point: the phase-4 table in §5.7 compares wall-clock across boxes, so a
-- 2-card row is only readable if the recipe underneath it is the one the 4-card row ran. Rendering
-- 64 per replica would have been the smaller change and the wrong number — global batch 128, 10,008
-- steps/epoch, and a run that needs an LR rescale before it means anything.
--
-- `B` and `replicas` are both true parameters, so this composes the two `#eval`s above and adds no
-- renderer code — the same "one `#eval`" the 4-GPU peer was. bs128/card fits a 24 GB 7900 XTX with
-- room: the single-device `mom256` render above is bs256 and §2d.1 measured it fitting on one.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 128 1000 "1.0e-05" 2
    Proofs.StableHLO.R34Opt.heavyBall "resnet34in")

#eval IO.FS.writeFile "verified_mlir/resnet34in_fwd.mlir"
  (Proofs.StableHLO.resnet34FwdFaithfulV 256 1000 "1.0e-05" "resnet34in")

#eval IO.FS.writeFile "verified_mlir/resnet34in_fwd_eval.mlir"
  (Proofs.StableHLO.resnet34FwdEvalFaithfulV 256 1000 "1.0e-05" "resnet34in")

-- Pin the seven literal artifact paths above against the name the renderer actually emits. If a
-- variant is renamed, this fails at `lake build` instead of at run time as an "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.r34AdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.r34AdamVariant 256 1 == "adam256"
#guard Proofs.StableHLO.r34AdamVariant 128 2 == "adamdp128"
#guard Proofs.StableHLO.r34AdamVariant 64 1 == "adam64"
#guard Proofs.StableHLO.r34AdamVariant 128 1 == "adam128"
-- ⭐ The bf16 marker, pinned the way every other marker here is. ⚠ These `#guard`s pin the SPELLING
-- but not the wiring, and the wiring is what actually broke: `resnet34AdamTrainStepFaithfulB`
-- derives its entry name from this function, and on bf16's first run it called it WITHOUT the flag
-- — so the artifact was written to `…momdp64bf16_train_step.mlir` while declaring
-- `@resnet34in_momdp64_train_step` inside, and the driver refused at load with an entry mismatch.
-- The third time this repo has shipped that exact defect (ConvNeXt's `wx`, then its `clip`).
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.heavyBall
         false false false "" true == "momdp64bf16"
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.R34Opt.heavyBall == "momdp64"
-- ▶ And the marker must not collide with the DRIVER's variant predicates, which read the same
-- string to decide the blob layout. `cdOn` is the dangerous one: it is a substring test for "do",
-- and a slug that tripped it would silently add a dropout region to the checkpoint.
-- These are `cdOn`/`accOn`/`emaOn` (`VerifiedTrain.lean`) evaluated on the bf16 slug: each is a
-- SUBSTRING test, and a false positive changes `nRegions`/`nScalars` — i.e. the checkpoint layout —
-- with no error anywhere.
#guard ("momdp64bf16".splitOn "do").length == 1
#guard ("momdp64bf16".splitOn "acc").length == 1
#guard !"momdp64bf16".startsWith "ema"
-- The optimizer axis. `.adamw` must keep every legacy name unchanged — that is what makes the
-- threading a no-op for the six artifacts above — and `.heavyBall` gets its own.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw == "adam"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall == "mom"
#guard Proofs.StableHLO.r34AdamVariant 32 2 .heavyBall == "momdp"
#guard Proofs.StableHLO.r34AdamVariant 256 1 .heavyBall == "mom256"
-- The 4-GPU ImageNet render: per-replica 64, so the slug carries 64 and NOT the 256 global batch.
-- Pinned because the driver derives `LEAN_MLIR_VARIANT` from `(B, replicas)` while the `#eval`
-- above hardcodes the path, and an "entry mismatch" at run time is the failure they drift into.
#guard Proofs.StableHLO.r34AdamVariant 64 4 .heavyBall == "momdp64"
-- The 2-GPU ImageNet render. Same rule, and worth pinning separately: `momdp64` and `momdp128`
-- differ only in the per-replica batch, so a slug that dropped `B` would collide two artifacts
-- rendered at different replica counts onto one path — the last-writer-wins race §2a found.
#guard Proofs.StableHLO.r34AdamVariant 128 2 .heavyBall == "momdp128"
-- ⚠ **THE TWO TRAILING FLAG AXES ARE INERT ON EVERY NAME ABOVE**, and that is what the defaults
-- buy: R34 renders no `wx` and no `clip` artifact, so passing them explicitly as `false` must
-- reproduce the legacy spellings character for character. `ResNet50RenderB.lean`'s guards pin the
-- ON spellings, since R50 is the net that renders them.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw false false == "adam"
#guard Proofs.StableHLO.r34AdamVariant 64 4 .heavyBall false false == "momdp64"
-- ⚠ And the ORDER, pinned here beside the function that decides it rather than only at the call
-- site: `wx` then `clip`, both after the batch. Getting this backwards renders an artifact whose
-- declared entry disagrees with its path, which the shim reports only as "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 64 1 .adamw true true == "adam64wxclip"

-- ⭐⭐ **D1's TWO BAKED CONSTANTS, pinned against the theorem that licenses them.**
-- `Proofs.clipFactor_accum` says `min(1, kc/(√(k²s) + kε)) = min(1, c/(√s + ε))`, i.e. that folding
-- the norm on the accumulated SUM and scaling BOTH constants by `k` reproduces the reference's clip
-- of the MEAN exactly. These are that identity's `k = 8` instance as the render actually emits it,
-- and they are the line a reader checks when `dense<8.000000000000>` looks like a wrong threshold.
#guard Proofs.StableHLO.clipNormStr 1.0 8 == "8.000000000000"
#guard Proofs.StableHLO.clipEpsStr 8 == "0.000008000000"
-- ⚠ `k = 1` — no accumulation — must leave both at the reference's own values, which is what makes
-- the scaling inert on a non-accumulating clip render.
#guard Proofs.StableHLO.clipNormStr 1.0 1 == "1.000000000000"
#guard Proofs.StableHLO.clipEpsStr 1 == "0.000001000000"
-- ⚠ and `optAccumK` is the SINGLE source of the `k` those two read, so pin that it agrees with the
-- constructor rather than being a second parse of the variant string.
#guard Proofs.StableHLO.optAccumK (Proofs.StableHLO.R34Opt.lambAccum 8) == 8
#guard Proofs.StableHLO.optAccumK (Proofs.StableHLO.R34Opt.adamwAccum 4) == 4
#guard Proofs.StableHLO.optAccumK Proofs.StableHLO.R34Opt.lamb == 1
#guard Proofs.StableHLO.optAccumK Proofs.StableHLO.R34Opt.adamw == 1
