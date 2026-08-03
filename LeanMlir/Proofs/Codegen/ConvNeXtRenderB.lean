import LeanMlir.Proofs.Codegen.ConvNeXtRender

/-! # ConvNeXt-T at the BATCHED index `N := B` — the forward (handoff §0.2 ▶2)

The ConvNeXt peer of `ResNet34RenderB` / `MobileNetV2RenderB`, and the reason it exists is
`planning/stochastic_depth.md`: **the drop mask is per-EXAMPLE**, and in the per-example-indexed
render (`ConvNeXtRender.lean`) a node denotes ONE example — `pretty B` lifts it across the batch, so
the node cannot see `j`. `dropPathB` needs its operand at index `B·n`, which is what this file's
chain produces.

⚠⚠ **The trap this closes is that the wrong thing TYPECHECKS.** `pretty B` already emits
`tensor<B×n>`, so a `broadcast_in_dim %mask, dims = [0]` + multiply against a per-example node
compiles, trains and descends — with no faithful `den` behind it. Every node below is instead a
`batchOp`/`*B` form whose `den` is `batchMap N (…)` or `batchMapAux N (…)`, i.e. honest about which
index is the batch.

**What this file is NOT (yet).** The forward only. The backward, the optimizer tail and the `#eval`
writers stay in `ConvNeXtRender.lean` until this chain is tied — §2b's order, which produced
`resnet34_adam_train_step_b.mlir` and tied it *before* anything swapped. Nothing here writes an
artifact, so `verified_mlir/` is untouched by construction.

**The gate** (`lake build convnext-fwd-b-tie`): this chain and the committed
`verified_mlir/convnext_fwd.mlir` must emit **byte-identical** text. That is a much stronger claim
than the numeric ties elsewhere in this thread, and it is available *because* every batched form was
built to emit its per-example peer's text byte-for-byte (`tests/TestBatchedEmitTie.lean`, 31 forms).
So the whole-net statement is the per-form statement composed — and if it ever fails, the tie file
localises which form did it in one run.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

/-! ## The batched shape helpers

⚠ `reassoc`/`unassoc` in the per-example renderer cast `SHlo (c*h*h) ↔ SHlo (c*(h*h))`. At the
batched index the same cast has to happen UNDER `N * ·`, which is `bnBatchLA`'s existing move
(`congrArg (N * ·) (Nat.mul_assoc …)`). It is a reindex, not a function change — the emitted text
is unaffected, since `skel` never sees the index. -/
private def reassocB {N c h : Nat} (e : SHlo (N*(c*h*h))) : SHlo (N*(c*(h*h))) :=
  (congrArg (N * ·) (Nat.mul_assoc c h h)) ▸ e

private def zVB {n : Nat} : Vec n := fun _ => 0
private def zKB {o i kh kw : Nat} : Kernel4 o i kh kw := fun _ _ _ _ => 0
private def zDB {c kh kw : Nat} : DepthwiseKernel c kh kw := fun _ _ _ => 0
private def zMB {a b : Nat} : Mat a b := fun _ _ => 0

/-- Batch, eps and the stage table — read from the per-example renderer's own constants where they
    are public, restated where they are `private`. ⚠ Restated, not re-derived: if these drift the
    byte tie fails loudly, which is the point of tying against the committed artifact rather than
    against a second copy of the shapes. -/
private def bB : Nat := 32
private def bEPS : String := "1.0e-6"
private def bDepths : Array Nat := #[3, 3, 9, 3]
private def bDims   : Array Nat := #[96, 192, 384, 768]
private def bSpats  : Array Nat := #[56, 28, 14, 7]

-- ⚠ The stochastic-depth site table (`cnxDropTotal`, `cnxBlockIdx`, `cnxDropSig`) lives in
-- `ConvNeXtRender.lean` — the train-step renderer there owns the signature and the variant name, and
-- this file imports it rather than the reverse. It is derived from THAT file's `cDepths`; this guard
-- is what says the two stage tables have not drifted, which the ramp indices below depend on.
#guard bDepths.foldl (· + ·) 0 == cnxDropTotal

/-! ## The sites, one per per-example site in `ConvNeXtRender.lean` -/

/-- One **channel-LN forward** site, batched: transpose to `[h·w, c]`, normalise each spatial row
    over its channels at the scalar identities `%one`/`%zero`, apply the real `[c]` affine,
    transpose back. Five `batchOp`s where the per-example peer has five bare nodes.

    ⚠ `m := h*h` is the SPATIAL row count PER EXAMPLE and `N := bB` is the batch. Collapsing those
    two into one index is exactly the defect this file exists to remove, and it is why the
    descriptors carry `(m, n)` internally instead of reading the SHlo index. -/
private def lnFwdSiteB (gN btN xin : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, t)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand xin (zVB : Vec (bB*(c*h*h))))))
    let (k2, n)  ← pretty bB (.batchOp (N := bB)
                                  (.lnRow (m := h*h) (n := c) "%one" "%zero" bEPS 0 1 0)
                                  (.operand t (zVB : Vec (bB*(h*h*c)))))
    let (k3, sc) ← pretty bB (.batchOp (N := bB) (.rowScale (m := h*h) (n := c) gN (zVB : Vec c))
                                  (.operand n (zVB : Vec (bB*(h*h*c)))))
    let (k4, bi) ← pretty bB (.batchOp (N := bB) (.rowBias (m := h*h) (n := c) btN (zVB : Vec c))
                                  (.operand sc (zVB : Vec (bB*(h*h*c)))))
    let (k5, o)  ← pretty bB (.batchOp (N := bB) (.transpose (m := h*h) (n := c))
                                  (.operand bi (zVB : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- One **ConvNeXt block** forward, batched: depthwise 7×7 → channel-LN → 1×1 expand → GELU →
    1×1 project → LayerScale → [drop] → `+ skip`. The residual add is `addVB`, the binary batched
    form.

    `drop = some i` puts the stochastic-depth site at ramp index `i` **on the residual branch**, i.e.
    between `layerScaleCh` and the `addVB`. ⚠⚠ **That placement is the whole correctness question and
    the obvious gate is blind to it** — at an all-ones mask `1 ⊙ (branch + x) = branch + x` exactly,
    so a site on the block OUTPUT is the same function bit-for-bit and every endpoint gate passes on
    it (`stochastic_depth.md` §7b, measured on EfficientNet). `scripts/misplace_drop_sites.py` builds
    exactly that render — same SSA names, order, types and line count — and it is the control that
    licenses believing any green run here.

    ⚠ At `drop = none` **no `pretty` call happens**, so the fresh-name counter does not move and the
    drop-free chain re-renders byte-identically. That is what keeps `convnext-fwd-b-tie` and the
    committed artifacts free of this feature. -/
private def fwdBlockB (pfx xin : String) (c e h : Nat) (drop : Option Nat := none) :
    StateM Nat (String × FNames) := do
  let (k1, d) ← pretty bB (.batchOp (N := bB)
      (.depthwise (h := h) (w := h) s!"%{pfx}dW" s!"%{pfx}db" (zDB : DepthwiseKernel c 7 7) zVB)
      (.operand xin zVB))
  let (k2, n) ← lnFwdSiteB s!"%{pfx}ng" s!"%{pfx}nbt" d c h
  let (k3, e') ← pretty bB (.batchOp (N := bB)
      (.conv (h := h) (w := h) s!"%{pfx}eW" s!"%{pfx}eb" (zKB : Kernel4 e c 1 1) zVB)
      (.operand n zVB))
  let (k4, g) ← pretty bB (.batchOp (N := bB) (.gelu (n := e*h*h))
      (.operand e' (zVB : Vec (bB*(e*h*h)))))
  let (k5, p) ← pretty bB (.batchOp (N := bB)
      (.conv (h := h) (w := h) s!"%{pfx}pW" s!"%{pfx}pb" (zKB : Kernel4 c e 1 1) zVB)
      (.operand g zVB))
  let (k6, ls) ← pretty bB (.batchOp (N := bB)
      (.layerScaleCh (h := h) (w := h) s!"%{pfx}lg" (zVB : Vec c)) (.operand p zVB))
  let (kD, br) ← match drop with
    | some i => pretty bB (.dropPathB (N := bB) (n := c*h*h) (dpName i) (fun _ => 0 : Vec bB)
                             (.operand ls (zVB : Vec (bB*(c*h*h)))))
    | none   => pure ("", ls)
  let (k7, bout) ← pretty bB (.addVB (.operand br (zVB : Vec (bB*(c*h*h)))) (.operand xin zVB))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ kD ++ k7, ⟨xin, d, n, e', g, p, bout⟩)

/-- One **downsample** forward, batched: channel-LN then 2×2/s2 conv. -/
private def fwdDownB (pfx xin : String) (ci co h2 : Nat) :
    StateM Nat (String × String × String) := do
  let (k1, n) ← lnFwdSiteB s!"%{pfx}ng" s!"%{pfx}nbt" xin ci (2*h2)
  let (k2, o) ← pretty bB (.batchOp (N := bB)
      (.convStrided (h := h2) (w := h2) s!"%{pfx}W" s!"%{pfx}b" (zKB : Kernel4 co ci 2 2) zVB)
      (.operand n zVB))
  pure (k1 ++ k2, n, o)

set_option maxRecDepth 8000 in
/-- **The full ConvNeXt-T `[3,3,9,3]` forward at the batched index.** Node for node the same chain
    `convNextFwdChain` emits — 4×4/s4 patchify stem (3→96, 224→56) → stem channel-LN → 4 stages at
    56/28/14/7 with 2×2/s2 downsamples between → GAP(7×7) → dense(768→nClasses).

    ⚠ Every node is a `batchOp`/`*B` form, so `den` is a `batchMap`/`batchMapAux` at `N := bB` and
    the batch is an index of the AST rather than a number only `pretty` knows. That is the entire
    content of the move; the emitted text is unchanged, which the tie checks.

    `sd := true` adds the 18 stochastic-depth sites, one per block, at ramp index `cnxBlockIdx si j`
    — and the sites are emitted in the FORWARD as well as the train step deliberately
    (`stochastic_depth.md` §3): at eval the driver supplies an all-ones mask, so they are the exact
    identity, and the `forward ⊂ train-step` prefix audit keeps a partner for the SD render instead
    of quietly not covering it. -/
def convNextFwdChainB (nClasses : Nat := 10) (sd : Bool := false) : StateM Nat CFwd := do
  let (cS, stemC) ← pretty bB (.batchOp (N := bB)
      (.convStride4 (h := 56) (w := 56) "%psW" "%psb" (zKB : Kernel4 96 3 4 4) zVB)
      (.operand "%x" (zVB : Vec (bB*(3*(2*(2*56))*(2*(2*56)))))))
  let (cSln, stem) ← lnFwdSiteB "%psng" "%psnbt" stemC 96 56
  let mut fwd := cS ++ cSln
  let mut cur := stem
  let mut blksAll : Array (Array FNames) := #[]
  let mut downLn : Array String := #[]
  let mut downIn : Array String := #[]
  for si in [0:4] do
    let c := bDims[si]!; let e := 4 * c; let h := bSpats[si]!
    let mut blks : Array FNames := #[]
    for j in [0:bDepths[si]!] do
      -- ⚠ `cnxBlockIdx si j`, NOT `j`: the ramp counts blocks over the whole net (denominator 17).
      -- The backward calls the same function walking the stages in reverse, which is why it is a
      -- function of `(si, j)` rather than a counter either loop carries.
      let (code, bn) ← fwdBlockB s!"s{si}b{j}" cur c e h (if sd then some (cnxBlockIdx si j) else none)
      fwd := fwd ++ code; cur := bn.bout; blks := blks.push bn
    blksAll := blksAll.push blks
    if si < 3 then
      downIn := downIn.push cur
      let (code, n, o) ← fwdDownB s!"d{si}" cur c bDims[si+1]! bSpats[si+1]!
      fwd := fwd ++ code; downLn := downLn.push n; cur := o
  let (cG, gap) ← pretty bB (.batchOp (N := bB) (.gap (c := 768) (h := 7) (w := 7))
      (.operand cur zVB))
  let (cLog, logits) ← pretty bB (.batchOp (N := bB)
      (.dense "%Wd" "%bd" (zMB : Mat 768 nClasses) zVB) (.operand gap zVB))
  pure { code := fwd ++ cG ++ cLog, blksAll := blksAll, downLn := downLn, downIn := downIn,
         gap := gap, stemC := stemC, hn := gap, logits := logits }

set_option maxRecDepth 8000 in
/-- **`@convnext_fwd_b`** — the batched-index peer of `convNextFwdFaithfulV`, same 180-parameter
    signature and same `%x`. Not written to `verified_mlir/`: it exists to be TIED against the
    committed per-example artifact, and an artifact nothing loads is a silent-hyperparameter hazard
    waiting to happen (§2a-quater). The writer lands with the swap, not before. -/
def convNextFwdRenderB (funcName : String := "convnext_fwd_b") (nClasses : Nat := 10)
    (banner : String :=
      "    // ── ConvNeXt-T forward at the BATCHED index N := B: every line is pretty(batchOp …) ──\n")
    -- ⚠ TRAILING, per §2m: a parameter inserted mid-list captures an existing positional argument
    -- at every call site, which is how the mnv2 `convBias` threading went wrong.
    (sd : Bool := false)
    : String := Id.run do
  let F : CFwd := (convNextFwdChainB nClasses sd).run' 0
  let body := F.code; let logits := F.logits
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [bB, 3*224*224]) :: (cnxAllParams nClasses).map (fun (nm, d) => s!"%{nm}: {ty d}"))
    -- ⚠ The mask inputs go LAST, after every parameter, matching the train step's placement and the
    -- driver's blob layout. Anywhere else and they capture an existing positional slot — the mnv2
    -- `convBias` failure (§2m), silent until the driver mis-walks the blob.
    ++ cnxDropSig bB sd
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [bB,nClasses]} " ++ "{\n" ++
    banner ++
    chLnPrelude ++ body ++
    s!"    return {logits} : {ty [bB,nClasses]}\n" ++ "  }\n}\n"

/-! ## The backward

Every site below is the per-example site with its node swapped for the batched form. ⚠ Two of them
are where increments 2 and 3's constructors earn their keep, and both are cases the emit tie alone
cannot judge:

* `lnRowBackB` takes the whole-batch saved LN input and hands example `k` `batchSlice k x`
  (`batchMapAux`). A descriptor here would hand example 0's activation to all `N` — same types,
  same bytes, different function (`den_lnRowBackB_per_example`).
* `veclnGammaGradB` / `rowDenseBiasGradB` contract the batch **and** the spatial rows. The
  per-example peers contract only the rows, and the two spellings agree at `N = 1`
  (`den_rowDenseBiasGradB_at_one`) — so a render that dropped the batch sum would pass a
  one-example check. -/

/-- One **channel-LN input-VJP** site, batched. -/
private def lnBackSiteB (gN xName cot : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, xT)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                   (reassocB (.operand xName (zVB : Vec (bB*(c*h*h))))))
    let (k2, dT)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                   (reassocB (.operand cot (zVB : Vec (bB*(c*h*h))))))
    let (k3, da)  ← pretty bB (.batchOp (N := bB) (.rowScale (m := h*h) (n := c) gN (zVB : Vec c))
                                   (.operand dT (zVB : Vec (bB*(h*h*c)))))
    let (k4, dxT) ← pretty bB (.lnRowBackB (N := bB) (m := h*h) (n := c) "%one" xT bEPS 0 1 zVB
                                   (.operand da (zVB : Vec (bB*(h*h*c)))))
    let (k5, o)   ← pretty bB (.batchOp (N := bB) (.transpose (m := h*h) (n := c))
                                   (.operand dxT (zVB : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- The **γ tail** for one LN site — the two-level `veclnGammaGradB`. -/
private def lnGammaTailB (gN xName cot : String) (c h : Nat) :
    StateM Nat (String × String) := do
    let (k1, xT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand xName (zVB : Vec (bB*(c*h*h))))))
    let (k2, dT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand cot (zVB : Vec (bB*(c*h*h))))))
    let (k3, o) ← pretty bB (.veclnGammaGradB (N := bB) (R := h*h) (D := c) xT bEPS 0
                                 (zVB : Vec (bB*(h*h*c))) (.operand dT (zVB : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3, o)

/-- The **β tail** — the two-level `rowDenseBiasGradB`, contracting batch and spatial rows. -/
private def lnBetaTailB (cot : String) (c h : Nat) : StateM Nat (String × String) := do
    let (k1, dT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand cot (zVB : Vec (bB*(c*h*h))))))
    let (k2, o) ← pretty bB (.rowDenseBiasGradB (N := bB) (R := h*h) (c := c)
                                 (.operand dT (zVB : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2, o)

/-- One **ConvNeXt block** backward: the cotangent chain only, param tails separate (the
    per-example renderer factors it the same way, which is what makes ConvNeXt the cheapest of the
    five to thread).

    ▶ **THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK** (`Proofs.dropPath_vjp_is_self`): a
    diagonal linear map is its own transpose, so there is no `*Grad` peer to keep in step and no
    second emitter to drift.

    ⚠ **IT APPLIES TO THE BRANCH ONLY, and that mirrors the forward's placement exactly.** The
    dropped cotangent `cotD` feeds the whole branch — LayerScale, project, GELU, expand, LN,
    depthwise — including every parameter gradient computed off it. The skip's fan-in at the bottom
    keeps the RAW `dy`. Dropping there too would attenuate the identity path, which is
    `s ⊙ (branch + x)` arriving by the other door.

    ⚠⚠ **AND `dyd` IS RETURNED BECAUSE ONE PARAMETER GRADIENT READS IT DIRECTLY.** LayerScale's γ
    gradient is `Σ (cot ⊙ p)` at the cotangent of the LayerScale OUTPUT — which is `s ⊙ dy` once a
    drop site sits between LayerScale and the add, not `dy`. Every other block gradient descends
    from `cot_p` and inherits the scale for free; `%…lg` is the one that would silently be computed
    against an undropped cotangent. It type-checks, trains and descends: 18 of 180 gradients wrong
    by a per-example factor, on the parameter stochastic depth is *about*. At `drop = none` this is
    `dy` itself, so nothing moves. -/
private def bwdBlockB (pfx dy : String) (b : FNames) (c e h : Nat) (drop : Option Nat := none) :
    StateM Nat (String × String × String × String × String × String × String) := do
  let (kD, dyd) ← match drop with
    | some i => pretty bB (.dropPathB (N := bB) (n := c*h*h) (dpName i) (fun _ => 0 : Vec bB)
                             (.operand dy (zVB : Vec (bB*(c*h*h)))))
    | none   => pure ("", dy)
  let (k1, cot_p) ← pretty bB (.batchOp (N := bB)
      (.layerScaleCh (h := h) (w := h) s!"%{pfx}lg" (zVB : Vec c)) (.operand dyd zVB))
  let (k2, cot_g) ← pretty bB (.convBackBatched (N := bB) (h := h) (w := h) s!"%{pfx}pW"
      (zKB : Kernel4 c e 1 1) zVB (.operand cot_p zVB))
  let (k3, cot_e) ← pretty bB (.geluBackB b.e (zVB : Vec (bB*(e*h*h))) (.operand cot_g zVB))
  let (k4, cot_n) ← pretty bB (.convBackBatched (N := bB) (h := h) (w := h) s!"%{pfx}eW"
      (zKB : Kernel4 e c 1 1) zVB (.operand cot_e zVB))
  let (k5, cot_d) ← lnBackSiteB s!"%{pfx}ng" b.d cot_n c h
  let (k6, cot_main) ← pretty bB (.depthwiseBackBatched (N := bB) (h := h) (w := h) s!"%{pfx}dW"
      (zDB : DepthwiseKernel c 7 7) zVB (.operand cot_d zVB))
  let (k7, cot_xin) ← pretty bB (.addVB (.operand cot_main (zVB : Vec (bB*(c*h*h))))
      (.operand dy zVB))
  pure (kD ++ k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d, dyd)

/-- One **downsample** backward. -/
private def bwdDownB (pfx dy xin : String) (ci co h2 : Nat) :
    StateM Nat (String × String × String) := do
  let (k1, cot_n) ← pretty bB (.convStridedBackBatched (N := bB) (h := h2) (w := h2) s!"%{pfx}W"
      (zKB : Kernel4 co ci 2 2) zVB (.operand dy (zVB : Vec (bB*(co*h2*h2)))))
  let (k2, cot_x) ← lnBackSiteB s!"%{pfx}ng" xin cot_n ci (2*h2)
  pure (k1 ++ k2, cot_n, cot_x)

/-- The **parameter gradients of one block** — every one a `*GradB`, i.e. `Σ_n` over the batch of
    the per-example gradient on `batchSlice n`. AdamW only: the SGD tail stays in the per-example
    renderer until the swap, since `%lr` is a runtime operand on the AdamW path and a baked literal
    on the SGD one (§2a-quater's silent-hyperparameter hazard). -/
private def blockParamGradB (pfx : String) (b : FNames)
    (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat) :
    StateM Nat (String × List (String × String)) := do
  let (cLg, nLg) ← pretty bB (.layerScaleChGammaGradB (N := bB) (c := c) (h := h) (w := h) b.p
      (zVB : Vec (bB*(c*h*h))) (.operand dy zVB))
  let (cPw, nPw) ← pretty bB (.convWeightGradB (N := bB) (ic := e) (oc := c) (h := h) (w := h)
      (kH := 1) (kW := 1) b.g (zVB : Vec c) (zVB : Vec (bB*(e*h*h))) (zKB : Kernel4 c e 1 1)
      (.operand cot_p zVB))
  let (cPb, nPb) ← pretty bB (.convBiasGradB (N := bB) (ic := e) (oc := c) (h := h) (w := h)
      (kH := 1) (kW := 1) (zKB : Kernel4 c e 1 1) (zVB : Vec (bB*(e*h*h))) (zVB : Vec c)
      (.operand cot_p zVB))
  let (cEw, nEw) ← pretty bB (.convWeightGradB (N := bB) (ic := c) (oc := e) (h := h) (w := h)
      (kH := 1) (kW := 1) b.n (zVB : Vec e) (zVB : Vec (bB*(c*h*h))) (zKB : Kernel4 e c 1 1)
      (.operand cot_e zVB))
  let (cEb, nEb) ← pretty bB (.convBiasGradB (N := bB) (ic := c) (oc := e) (h := h) (w := h)
      (kH := 1) (kW := 1) (zKB : Kernel4 e c 1 1) (zVB : Vec (bB*(c*h*h))) (zVB : Vec e)
      (.operand cot_e zVB))
  let (cNg, nNg) ← lnGammaTailB s!"%{pfx}ng" b.d cot_n c h
  let (cNb, nNb) ← lnBetaTailB cot_n c h
  let (cDw, nDw) ← pretty bB (.depthwiseWeightGradB (N := bB) (c := c) (h := h) (w := h)
      (kH := 7) (kW := 7) b.xin (zVB : Vec c) (zVB : Vec (bB*(c*h*h)))
      (zDB : DepthwiseKernel c 7 7) (.operand cot_d zVB))
  let (cDb, nDb) ← pretty bB (.depthwiseBiasGradB (N := bB) (c := c) (h := h) (w := h)
      (kH := 7) (kW := 7) (zDB : DepthwiseKernel c 7 7) (zVB : Vec (bB*(c*h*h))) (zVB : Vec c)
      (.operand cot_d zVB))
  pure (cLg ++ cPw ++ cPb ++ cEw ++ cEb ++ cNg ++ cNb ++ cDw ++ cDb,
    [(s!"{pfx}dW", nDw), (s!"{pfx}db", nDb), (s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb),
     (s!"{pfx}eW", nEw), (s!"{pfx}eb", nEb), (s!"{pfx}pW", nPw), (s!"{pfx}pb", nPb),
     (s!"{pfx}lg", nLg)])

/-- The **parameter gradients of one downsample**. -/
private def downParamGradB (pfx downLn downIn cot_n dy : String) (ci co h2 : Nat) :
    StateM Nat (String × List (String × String)) := do
  let (cB, nB) ← pretty bB (.convStridedBiasGradB (N := bB) (ic := ci) (oc := co) (h := h2)
      (w := h2) (kH := 2) (kW := 2) (zKB : Kernel4 co ci 2 2)
      (zVB : Vec (bB*(ci*(2*h2)*(2*h2)))) (zVB : Vec co) (.operand dy zVB))
  let (cNg, nNg) ← lnGammaTailB s!"%{pfx}ng" downIn cot_n ci (2*h2)
  let (cNb, nNb) ← lnBetaTailB cot_n ci (2*h2)
  let (wcode, nW) ← pretty bB (.convStridedWeightGradB (N := bB) (ic := ci) (oc := co) (h := h2)
      (w := h2) (kH := 2) (kW := 2) downLn (zVB : Vec co)
      (zVB : Vec (bB*(ci*(2*h2)*(2*h2)))) (zKB : Kernel4 co ci 2 2)
      (.operand dy (zVB : Vec (bB*(co*h2*h2)))))
  pure (cB ++ cNg ++ cNb ++ wcode,
    [(s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb), (s!"{pfx}W", nW), (s!"{pfx}b", nB)])

set_option maxRecDepth 8000 in
/-- **The whole-net batched traversal** — forward + cotangent + every parameter gradient, the
    batched peer of `convNextBackAll true (some …)`. Returns `(code, gradMap, softmaxSSA)` with the
    same shape, so the AdamW tail in `ConvNeXtRender.lean` can consume either.

    ⚠ **AdamW only, deliberately.** The per-example traversal serves both renders off one `adam`
    flag; this one does not, because the SGD path bakes `lr` as a literal where AdamW takes it as a
    runtime `%lr` operand, and an SGD render at the batched index is not something any config asks
    for yet. Adding the flag later is cheap; adding an artifact nobody loads is §2a-quater's
    silent-hyperparameter hazard.

    ⚠ The `%dgi`/`%dgb`/`%dgn`/`%dgd`/`%dgapf` GAP-backward block is **hand-written text on both
    sides**, carried over verbatim. It is one of §5's declared non-AST carve-outs, so the batched
    move neither improves nor degrades it — but note it is parameterised by `bB` and therefore
    already batch-correct, which is why it needs no peer. -/
def convNextBackAllB (smooth : Option (String × String × String) := none) (nClasses : Nat := 10)
    (sd : Bool := false) :
    StateM Nat (String × List (String × String) × String) := do
    -- ═══ forward — the SAME chain the byte-tied `convNextFwdChainB` emits ═══
    let F : CFwd ← convNextFwdChainB nClasses sd
    let (cSm, nSm) ← pretty bB (.batchOp (N := bB) (.softmaxDiv (n := nClasses))
        (.batchOp (N := bB) (.expe (n := nClasses))
          (.operand F.logits (zVB : Vec (bB*nClasses)))))
    let (cSub, dyr) ← pretty bB (.subB (.operand nSm (zVB : Vec (bB*nClasses)))
        (.operand "%onehot" zVB))
    let fwd := F.code ++ cSm ++ cSub
    -- ═══ the cotangent ═══
    let (cDyC, dyName) ← match smooth with
      | none => pure (s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [bB, nClasses]}\n", "%dy")
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty bB (.scaleB (N := bB) (n := nClasses) aStr 0
              (.operand "%onehot" (zVB : Vec (bB*nClasses))))
          let (c2, n2) ← pretty bB (.addVB (.operand dyr (zVB : Vec (bB*nClasses)))
              (.operand n1 (zVB : Vec (bB*nClasses))))
          -- ⚠ At the batched index these are `N := bB`, where the per-example render writes
          -- `N := 1` — the SAME emitted text (the tag's `n` is what the emitter reads), and the
          -- annotation trap that note warns about disappears, because `bB * nClasses` never has to
          -- reduce definitionally to anything.
          let (c3, n3) ← pretty bB (.shiftB (N := bB) (n := nClasses) negAK 0
              (.operand n2 (zVB : Vec (bB*nClasses))))
          let (c4, n4) ← pretty bB (.divConstB (N := bB) (n := nClasses) bStr 0
              (.operand n3 (zVB : Vec (bB*nClasses))))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    -- ═══ head ═══
    let (cDd, cot_gap) ← pretty bB (.batchOp (N := bB) (.dotOut "%Wd" (zMB : Mat 768 nClasses))
        (.operand dyName zVB))
    let (cWd, nWd) ← pretty bB (.weightGradB (N := bB) (m := 768) (n := nClasses) F.hn
        (zVB : Vec (bB*768)) (.operand dyName (zVB : Vec (bB*nClasses))))
    let (cBd, nBd) ← pretty bB (.biasGradB (N := bB) (n := nClasses)
        (.operand dyName (zVB : Vec (bB*nClasses))))
    let mut updMap : List (String × String) := [("Wd", nWd), ("bd", nBd)]
    let mut bwd := cDyC ++ cDd ++ cWd ++ cBd ++
      s!"    %dgi = stablehlo.reshape {cot_gap} : ({ty [bB,768]}) -> {ty [bB,768,1,1]}\n" ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [bB,768,1,1]}) -> {ty [bB,768,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [bB,768,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [bB,768,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [bB,768,7,7]}) -> {ty [bB, 768*7*7]}\n"
    let mut dy := "%dgapf"
    for si' in [0:4] do
      let si := 3 - si'
      let c := bDims[si]!; let e := 4 * c; let h := bSpats[si]!
      for j' in [0:bDepths[si]!] do
        let j := bDepths[si]! - 1 - j'
        let b := (F.blksAll[si]!)[j]!
        -- ⚠ `cnxBlockIdx si j` again, from the SAME function the forward called. This loop runs the
        -- stages and the blocks BACKWARDS, so a counter carried by either loop would have to be run
        -- in reverse to name the same sites — the mismatch would pair every backward site with the
        -- wrong forward mask, which typechecks and trains.
        let (code, cot_xin, cot_p, cot_e, cot_n, cot_d, dyd) ←
          bwdBlockB s!"s{si}b{j}" dy b c e h (if sd then some (cnxBlockIdx si j) else none)
        -- ⚠ `dyd`, not `dy` — LayerScale's γ gradient reads the cotangent at the LayerScale OUTPUT,
        -- which the drop site scales. See `bwdBlockB`.
        let (pcode, pairs) ← blockParamGradB s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dyd c e h
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_xin
      if si > 0 then
        let ci := bDims[si-1]!; let h2 := bSpats[si]!
        let (code, cot_n, cot_x) ← bwdDownB s!"d{si-1}" dy (F.downIn[si-1]!) ci c h2
        let (pcode, pairs) ← downParamGradB s!"d{si-1}" (F.downLn[si-1]!) (F.downIn[si-1]!)
            cot_n dy ci c h2
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_x
    -- ═══ stem: back through the stem LN, then the patchify conv's own gradients ═══
    let (cg, ng) ← lnGammaTailB "%psng" F.stemC dy 96 56
    let (cb, nb) ← lnBetaTailB dy 96 56
    let (cx, dx) ← lnBackSiteB "%psng" F.stemC dy 96 56
    bwd := bwd ++ cg ++ cb ++ cx
    updMap := updMap ++ [("psng", ng), ("psnbt", nb)]
    dy := dx
    let (cPsb, nPsb) ← pretty bB (.convBiasGradB (N := bB) (ic := 3) (oc := 96) (h := 56) (w := 56)
        (kH := 4) (kW := 4) (zKB : Kernel4 96 3 4 4) (zVB : Vec (bB*(3*56*56))) (zVB : Vec 96)
        (.operand dy zVB))
    let (cPsW, nPsW) ← pretty bB (.convStride4WeightGradB (N := bB) (ic := 3) (oc := 96) (h := 56)
        (w := 56) (kH := 4) (kW := 4) "%x" (zVB : Vec 96)
        (zVB : Vec (bB*(3*(2*(2*56))*(2*(2*56))))) (zKB : Kernel4 96 3 4 4)
        (.operand dy (zVB : Vec (bB*(96*56*56)))))
    bwd := bwd ++ cPsW ++ cPsb
    updMap := updMap ++ [("psW", nPsW), ("psb", nPsb)]
    pure (fwd ++ bwd, updMap, nSm)

set_option maxRecDepth 8000 in
/-- **The ConvNeXt-T AdamW train step at the batched index.** ⚠ It is the SAME renderer the
    per-example path uses — `convNextAdamTrainStepFaithful` with `traversal` pointed at
    `convNextBackAllB` — not a copy.

    That is possible because the AdamW tail is entirely **parameter-space**: `adamMNextF`,
    `adamVNextF`, `adamWParamF`, `gradSumSqAccF` and `clipScaleF` are indexed by the parameter's
    own size and never see the batch. So "the AdamW tail at the batched index" turned out to be no
    work at all — the batch had already been factored out of it, by the ops' own shapes. The only
    thing that moves is which traversal produced the gradients. -/
def convNextAdamTrainStepFaithfulB (alphaStr negAlphaKStr bStr : String)
    (replicas : Nat := 1) (nClasses : Nat := 10) (slug : String := "convnext")
    (ema : Bool := false) (wdExclude : Bool := false) (wdStr : String := "0.0001")
    (clip : Bool := false) (clipStr : String := "1.0") (sd : Bool := false) : String :=
  let negAK := if negAlphaKStr.isEmpty then "-" ++ alphaOverK nClasses 0.1 else negAlphaKStr
  convNextAdamTrainStepFaithful alphaStr negAlphaKStr bStr replicas nClasses slug ema
    wdExclude wdStr clip clipStr
    -- ⚠⚠ `sd` IS SPELLED ONCE AND REACHES BOTH HALVES FROM HERE — the traversal (which places the
    -- 18 sites) and the wrapper (which declares the 18 inputs, the 18 pass-through outputs and the
    -- `drop` variant name). Letting a caller set them independently is the shape of defect this
    -- file's own thread keeps finding; here there is nothing to keep in step.
    (traversal := some (convNextBackAllB (some (alphaStr, negAK, bStr)) nClasses sd))
    (sd := sd)

/-- The per-example render's own banner, so the tie can demand **byte-identity** rather than
    "identical apart from a comment". ⚠ Worth the parameter: a tie that compares modulo one line is
    a tie with a hole in it, and the hole is exactly where a renderer's own description of what it
    did would live. Measured first, then removed — the two differed in this line and nothing else. -/
def cnxFwdPerExampleBanner : String :=
  "    // ── ConvNeXt-T forward: every line is pretty(verified AST node) ──\n"

/-- The SD forward's banner. Its own, and not `cnxFwdPerExampleBanner`, because these bytes ARE a
    different render and a banner claiming otherwise is the `VerifiedNets` docstring defect (§0.9
    finding 3) in the artifact itself. -/
def cnxDropFwdBanner : String :=
  "    // ── ConvNeXt-T forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──\n" ++
  "    // 18 drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the\n" ++
  "    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:\n" ++
  "    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the\n" ++
  "    // SD train step and the forward-subset-train-step audit keeps a partner.\n"

end Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § ▶ THE STOCHASTIC-DEPTH ARTIFACTS (`planning/stochastic_depth.md`, handoff §0.10)
-- ════════════════════════════════════════════════════════════════
--
-- ⚠⚠ THESE ARE THE ONLY ARTIFACTS THIS FILE WRITES, AND THEY ARE ALL NEW. `ConvNeXtRenderB`'s
-- drop-free chain is still TIED BUT NOT SWAPPED: it differs from the committed
-- `convnext_adam_train_step.mlir` on 78 lines (the conv-VJP's `transpose`/`reverse` in the other
-- order — commuting ops on disjoint axes), and §5 requires a NUMERIC tie to license moving bytes in
-- a committed artifact. `convnext-adam-tie` is IREE-linked and does not link on this box. So the SD
-- renders are built on the batched chain — where the per-example mask is expressible at all — and
-- every existing artifact keeps its bytes. The 78-line spelling rides along in these NEW files,
-- where it is not a change to anything.
--
-- ⚠ The SD render therefore is NOT byte-comparable line-for-line with `convnext_adam`. Its keep = 1
-- gate is NUMERIC (`adamdrop` at every keep 1.0 must train what `adam` trains), under
-- `scripts/det_shim.sh` — cross-graph numeric comparison on CUDA has no resolution without it,
-- three times recorded.

-- ── Imagenette, K=10, bs32 — the GATE VEHICLE ──────────────────────────────────────────────────
-- ⚠ Not a matched pair: `convNextTinyConfig` sets no `dropPath`, so this render's accuracy is
-- comparable to nothing. It exists because every gate that has to run (keep = 1, the misplacement
-- control, the ones-mask forward) is seconds here and minutes at ImageNet scale.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 1 10 "convnext"
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- Its prefix partner. ⚠ The SD variant gets its OWN forward rather than reusing `convnext_fwd`:
-- letting the SD trainer eval through the drop-free forward is what the reference literally does,
-- but it would leave the SD train step with no prefix partner at all — i.e. SPEND one of the two
-- load-bearing structural gates in the repo (it caught `resnet34_fwd` and `mobilenetv2_fwd` scoring
-- nets they had not trained) rather than pay 18 dead multiplies at eval.
#eval IO.FS.writeFile "verified_mlir/convnext_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnext_drop_fwd" 10
    Proofs.StableHLO.cnxDropFwdBanner (sd := true))

-- The **2-replica** peer, and it exists for one reason: `lake build drop-shard-check` is the gate
-- that says the per-example masks are SHARDED rather than replicated, and its known answer is exact
-- only at TWO replicas — f32 addition is COMMUTATIVE, so `(a+b)/2` is invariant under swapping the
-- halves, where above two the collective is a tree whose order a permutation changes and
-- associativity does not hold. ⚠ So this is not "the DP render at a smaller replica count"; it is
-- the only replica count at which that gate is a bit-exactness claim.
--
-- ⚠ The defect §5b predicted and §0.6 found was in the SHIM (the shard flag, and the mask buffer
-- sized at the per-device batch), not in any render — so it is net-independent and ConvNeXt inherits
-- the fix. Running the gate here is what says so rather than assumes it: ConvNeXt is the first
-- LayerNorm net to drive it, and on an LN net the replica-0-local witness is `%loss` ALONE (there
-- are no batch statistics), which is a materially weaker anti-vacuity half than EfficientNet's.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdpdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 2 10 "convnext"
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- ── Full 1000-class ImageNet, slug `cnxin` ─────────────────────────────────────────────────────
-- ⚠ BOTH SCALES, deliberately. §0.4 finding 5: *a feature is not done when its Imagenette artifact
-- renders* — RMSProp had been carried to both scales and EMA/dropPath had not, so EfficientNet's
-- ImageNet trainer did not carry the features its reference number depends on. Both scales are one
-- `#eval` apart, which is exactly why it is easy to stop at one.
--
-- ⚠⚠ AND BOTH THE SINGLE-DEVICE **AND THE DP** PEER, for the reason §0.5 records: an ImageNet run
-- loads the DP render, and the DP renders had silently fallen three features behind their
-- single-device peers. `wx` ++ `clip` ++ `drop` is `convNeXtTinyImagenetConfig` entire
-- (`weightDecay := 0.05`, `wdExcludeNormBias := true`, `gradClipNorm := 1.0`, `dropPath := 0.1`) —
-- the first ConvNeXt artifact that carries every optimizer-and-regulariser knob its reference sets.
#eval IO.FS.writeFile "verified_mlir/cnxin_adamwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 1 1000 "cnxin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true))
#eval IO.FS.writeFile "verified_mlir/cnxin_adamdpwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 4 1000 "cnxin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true))
#eval IO.FS.writeFile "verified_mlir/cnxin_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "cnxin_drop_fwd" 1000
    Proofs.StableHLO.cnxDropFwdBanner (sd := true))

-- The entry name, the artifact path and `LEAN_MLIR_VARIANT` must agree or the shim refuses the call
-- ("entry mismatch"). ⚠ These matter MORE for `drop` than for `wx` or `clip`, because `drop` also
-- changes the ARITY: a variant name that lost the marker would put an 18-input-wider graph behind
-- the plain `adam` path's artifact name, checkpoint and vmfb.
#guard Proofs.StableHLO.cnxAdamVariant 1 false false false true == "adamdrop"
#guard Proofs.StableHLO.cnxAdamVariant 4 false false false true == "adamdpdrop"
#guard Proofs.StableHLO.cnxAdamVariant 1 false true true true == "adamwxclipdrop"
#guard Proofs.StableHLO.cnxAdamVariant 4 false true true true == "adamdpwxclipdrop"
-- ⚠ The marker must not LEAD: the driver keys its 4-region `[θ|m|v|ema]` blob off
-- `variant.startsWith "ema"`, so `emadrop` has to still start with "ema".
#guard (Proofs.StableHLO.cnxAdamVariant 1 true false false true).startsWith "ema"
-- ⚠ And the driver's own predicate is a SUBSTRING test for `"drop"`; `"sd"` would fire on every
-- name containing `rmsdp`. ConvNeXt has no RMSProp variant, but the marker is shared with the nets
-- that do, so the property is checked here too rather than assumed to be EfficientNet's problem.
#guard ((Proofs.StableHLO.cnxAdamVariant 1).splitOn "drop").length == 1
#guard ((Proofs.StableHLO.cnxAdamVariant 4 false true true true).splitOn "drop").length == 2
