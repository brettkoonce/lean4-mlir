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
    (convBias : Bool) : StateM Nat BFwdB := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk zc) (.operand xName zin))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zin))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk zc) (.operand nR1 zin))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zin))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zin))
  let _ := zbn
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "" }

/-- Downsample block forward: strided body + strided projection skip. `cin→c`, `2hh→hh`. -/
private def downFwdB (B cin c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) : StateM Nat BFwdB := do
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
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk1 zc) (.operand xName zinS))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zout))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk2 zc) (.operand nR1 zout))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zout))
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}Wp" (biasName convBias s!"%{p}bp" c) zkp zc) (.operand xName zinS))
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
    (convBias : Bool) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zin (.operand dyName zin))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk zc (.operand nDn2 zin))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zin (.operand nDc2 zin))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk zc (.operand nDn1 zin))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zin) (.operand nDa zin))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2
  let (cW1, nW1) ← pretty B (.convWeightGradB xName zc zin zk (.operand nDn1 zin))
  let (cb1, nb1) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn1 zin)) else pure ("", "")
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradB f.r1 zc zin zk (.operand nDn2 zin))
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
    (convBias : Bool) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0      -- §2l step A: the 1×1 option-B shortcut
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn  : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zout (.operand dyName zout))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk2 zc (.operand nDn2 zout))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zout (.operand nDc2 zout))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (.convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk1 zc (.operand nDn1 zout))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zc zbn (.operand nDa zbn))
  let (cDcp, nDcp) ← pretty B (.convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" zkp zc (.operand nDnp zout))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zinS) (.operand nDcp zinS))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2 Wp bp gp btp
  let (cW1, nW1) ← pretty B (.convStridedWeightGradB xName zc zinS zk1 (.operand nDn1 zout))
  let (cb1, nb1) ← if convBias then pretty B (.convStridedBiasGradB (h := hh) (w := ww) zk1 zinS zc (.operand nDn1 zout)) else pure ("", "")
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradB f.r1 zc zout zk2 (.operand nDn2 zout))
  let (cb2, nb2) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk2 zout zc (.operand nDn2 zout)) else pure ("", "")
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbn (.operand nDa zbn))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  let (cWp, nWp) ← pretty B (.convStridedWeightGradB xName zc zinS zkp (.operand nDnp zout))
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

/-- Fixed-6-decimal float literal, so a computed smoothing constant emits in the SAME textual form
    the hand-written literals used and `nClasses = 10` re-renders byte-identical. -/
private def fmt6 (x : Float) : String :=
  let neg := x < 0.0
  let n := ((if neg then -x else x) * 1000000.0 + 0.5).toUInt64.toNat
  let ip := n / 1000000
  let fp := n % 1000000
  let fs := (toString fp).leftpad 6 '0'
  (if neg then "-" else "") ++ toString ip ++ "." ++ fs

/-- **The label-smoothing mass per class, α/K.** α = 0.1 throughout; K is `nClasses`.

    ⚠ **This was hardcoded `0.010000` — correct at K = 10 and WRONG at every other K**, and it sat
    in the COTANGENT, not just in the report-only `%loss`. At `nClasses = 1000` it made the smoothing
    term 100× too large: it removes 10.0 of probability mass instead of 0.1, i.e. a different
    objective, silently. Caught 2026-07-30 by the first ImageNet smoke run reporting loss ≈ 87 where
    1000-class CE at init must be ≈ ln(1000) = 6.9 — the number was implausible, and that is the only
    reason it surfaced. Nothing in the repo's proofs covers it: `α` is a *literal in emitted text*,
    which is exactly the carve-out class §5 says needs its own numeric check, and §2b's `%loss` bug
    is the standing precedent for it going wrong unnoticed. -/
private def alphaOverK (nClasses : Nat) : String := fmt6 (0.1 / nClasses.toFloat)

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
deriving DecidableEq, Repr

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
private def optOne (opt : R34Opt) (B : Nat) (replicas : Nat) (g : PGrad) :
    StateM Nat (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) := ViTRender.emitGradAllReduce g.grad g.ds g.nm replicas
  let gr : SHlo n := .operand gAvg z
  match opt with
  | .adamw =>
    let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
    let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
    let (cT, nT) ← pretty B (.adamWParamF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                      "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" g.ds 0 0 0 0 0 0 0 z z z gr)
    pure (arS ++ cM ++ cV ++ cT, nT, nM, nV)
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
    let (cD, nD) ← pretty B (.momVNextF s!"%{g.nm}" "%wd" g.ds 0 z gr)
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
    pure (arS ++ cD ++ cV ++ cT, nT, s!"%{g.nm}m", nV)

/-- The optimizer's baked constants. `.adamw` is byte-for-byte the committed block; `.heavyBall`
    emits only what it reads, so there are no dead constants in the momentum artifact.

    `%wd` is baked rather than a runtime arg because weight decay is not scheduled — unlike `%lr`,
    which stays a `tensor<f32>` argument so one graph serves the whole cosine schedule. -/
private def optConstsB (opt : R34Opt) : String :=
  match opt with
  | .adamw =>
    "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
    "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
    "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
    "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
    "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"
  | .heavyBall =>
    "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
    "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

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
def r34AdamVariant (B replicas : Nat) (opt : R34Opt := .adamw) : String :=
  (match opt with
   | .adamw     => if replicas ≤ 1 then "adam" else "adamdp"
   | .heavyBall => if replicas ≤ 1 then "mom"  else "momdp") ++
  (if B == 32 then "" else toString B)

set_option maxRecDepth 4000000 in
/-- **ResNet-34 `[3,4,6,3]` AdamW train step, batch-BN, rendered from the verified AST at `N := B`.**
    515 inputs (`%x`, 146 θ, 146 m, 146 v, `%lr`/`%bc1`/`%bc2`, 72 running-stat slots, `%onehot`)
    and 513 outputs (146 θ', 146 m', 146 v', `%loss`/`%bc1`/`%bc2`, 72 batch stats) — the interface
    `tests/TestResnet34Train.lean`'s hand-written render already presents, so the driver is
    unchanged. Parameter ORDER comes from `r34SigList`, the same single source the per-example
    render and both forwards use, so the arity/order contract cannot drift between them. -/
def resnet34AdamTrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (opt : R34Opt := .adamw) (slug : String := "resnet34")
    (convBias : Bool := false) : String :=
  let optLabel : String := match opt with
    | .adamw     => "AdamW"
    | .heavyBall => "heavy-ball momentum + coupled L2"
  let go : StateM Nat String := do
    -- ═══ stem: 7×7/s2 conv → batch BN → relu → 2×2 maxpool ═══
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*112*112)) := fun _ => 0
    let z112b : Vec (B*(64*(112*112))) := fun _ => 0
    let z56   : Vec (B*(64*56*56)) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStrided (h := 112) (w := 112) "%sW" (biasName convBias "%sbi" 64) zSk z64) (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*112*112)) (.operand nStn z112))
    let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool (c := 64) (h := 56) (w := 56)) (.operand nStr z112))
    -- ═══ 16 blocks ═══
    let f1  ← idFwdB   B 64 56 epsStr "s1b0" nStp convBias
    let f2  ← idFwdB   B 64 56 epsStr "s1b1" f1.o convBias
    let f3  ← idFwdB   B 64 56 epsStr "s1b2" f2.o convBias
    let f4  ← downFwdB B 64 128 28 epsStr "d2" f3.o convBias
    let f5  ← idFwdB   B 128 28 epsStr "s2b0" f4.o convBias
    let f6  ← idFwdB   B 128 28 epsStr "s2b1" f5.o convBias
    let f7  ← idFwdB   B 128 28 epsStr "s2b2" f6.o convBias
    let f8  ← downFwdB B 128 256 14 epsStr "d3" f7.o convBias
    let f9  ← idFwdB   B 256 14 epsStr "s3b0" f8.o convBias
    let f10 ← idFwdB   B 256 14 epsStr "s3b1" f9.o convBias
    let f11 ← idFwdB   B 256 14 epsStr "s3b2" f10.o convBias
    let f12 ← idFwdB   B 256 14 epsStr "s3b3" f11.o convBias
    let f13 ← idFwdB   B 256 14 epsStr "s3b4" f12.o convBias
    let f14 ← downFwdB B 256 512 7 epsStr "d4" f13.o convBias
    let f15 ← idFwdB   B 512 7 epsStr "s4b0" f14.o convBias
    let f16 ← idFwdB   B 512 7 epsStr "s4b1" f15.o convBias
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
    let b16 ← idBackGradB   B 512 7 epsStr "s4b1" f16 nDgp convBias
    let b15 ← idBackGradB   B 512 7 epsStr "s4b0" f15 b16.dx convBias
    let b14 ← downBackGradB B 256 512 7 epsStr "d4" f14 b15.dx convBias
    let b13 ← idBackGradB   B 256 14 epsStr "s3b4" f13 b14.dx convBias
    let b12 ← idBackGradB   B 256 14 epsStr "s3b3" f12 b13.dx convBias
    let b11 ← idBackGradB   B 256 14 epsStr "s3b2" f11 b12.dx convBias
    let b10 ← idBackGradB   B 256 14 epsStr "s3b1" f10 b11.dx convBias
    let b9  ← idBackGradB   B 256 14 epsStr "s3b0" f9  b10.dx convBias
    let b8  ← downBackGradB B 128 256 14 epsStr "d3" f8 b9.dx convBias
    let b7  ← idBackGradB   B 128 28 epsStr "s2b2" f7 b8.dx convBias
    let b6  ← idBackGradB   B 128 28 epsStr "s2b1" f6 b7.dx convBias
    let b5  ← idBackGradB   B 128 28 epsStr "s2b0" f5 b6.dx convBias
    let b4  ← downBackGradB B 64 128 28 epsStr "d2" f4 b5.dx convBias
    let b3  ← idBackGradB   B 64 56 epsStr "s1b2" f3 b4.dx convBias
    let b2  ← idBackGradB   B 64 56 epsStr "s1b1" f2 b3.dx convBias
    let b1  ← idBackGradB   B 64 56 epsStr "s1b0" f1 b2.dx convBias
    -- ═══ stem backward: maxpool-back → relu mask → BN back, then the 4 stem grads ═══
    let (cDmp, nDmp) ← pretty B (.maxPoolBackB (N := B) (c := 64) (h := 56) (w := 56) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 64) (h := 112) (w := 112) "%sg" nStc epsStr 0 z64 z112b (.operand nDsr z112b))
    let (csW, nsW) ← pretty B (.convStridedWeightGradB "%x" z64 zx zSk (.operand nDsn z112))
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
      let (c, nT, nM, nV) ← optOne opt B replicas g
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
      zeroBiasPrelude convBias ++ body ++ optConstsB opt ++ adamCode ++ lossCode ++
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
  let fname := s!"{slug}_{r34AdamVariant B replicas opt}_train_step"
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
-- The optimizer axis. `.adamw` must keep every legacy name unchanged — that is what makes the
-- threading a no-op for the six artifacts above — and `.heavyBall` gets its own.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw == "adam"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall == "mom"
#guard Proofs.StableHLO.r34AdamVariant 32 2 .heavyBall == "momdp"
#guard Proofs.StableHLO.r34AdamVariant 256 1 .heavyBall == "mom256"
