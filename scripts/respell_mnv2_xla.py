#!/usr/bin/env python3
"""Step 3b of planning/xla_same_respell_and_blueprint_audit.md — re-spell MobileNetV2's Proofs
tier at the XLA-SAME phase. ONE-OFF; delete after it has been applied and committed.

What it does (run from the repo root, on a clean tree, AFTER the B0 half of step 3 is committed):
  1. a 55-rule whole-word name map over the sixteen MobileNetV2 files (the reduced 6-block cone,
     the 17-block paper files, the batched Adam backward graphs, the retired PC test):
     flatConvStride2* -> flatConvStride2Xla*, depthwiseStride2Flat* -> depthwiseStride2FlatXla*,
     the backward / bridge / Maps / token / SGD-op names likewise;
  2. adds the four XLA-SAME twins of the SHARED certs to MobileNetV2Close.lean — the symmetric
     originals stay because ResNet-34 (stem) and EfficientNet-B0 (strided depthwise) reuse them;
  3. adds Mnv2PoC.convStridedXla{W,B}_den (the per-example XLA stem dens; ResNet34PoC's stay);
  4. adds depthwiseStridedXlaBackBatched_faithful beside its symmetric peer in EfficientNetBackB0;
  5. adds the seven #print axioms lines to tests/AuditAxioms.lean.
It refuses to run if the tree already carries any of the names it introduces.
Then: the fast build in the planning doc (3b), then lake build Proofs Certs and the section-6
gates; the numerals 2.154e3 / 1.444e96 and 4.750e153 / 1.076e152 must reproduce.
"""
import pathlib, sys
_guard = pathlib.Path("LeanMlir/Proofs/Architectures/MobileNetV2Close.lean").read_text()
if "mnv2_render_stem_convW_xla_certified" in _guard:
    sys.exit("already applied (MobileNetV2Close.lean carries the XLA twins); refusing to run twice")
import sys, pathlib, re
ROOT = pathlib.Path(".")
FILES = [
 "LeanMlir/Proofs/Codegen/MobileNetV2RenderPC.lean",
 "LeanMlir/Proofs/Codegen/MobileNetV2RenderPCEval.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2ChainClose.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2FaithfulPoC.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2FaithfulPoCPaper.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2TiePoCPaper.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2FullPaper.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2FullVJP.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2BackB0.lean",
 "LeanMlir/Proofs/Architectures/MobileNetV2BackCertifiedTie.lean",
 "LeanMlir/Proofs/Foundation/MobileNetV2WholeBackCertifiedTie.lean",
 "LeanMlir/Proofs/Float/MobileNetV2WholeFloatBridge.lean",
 "LeanMlir/Proofs/Float/MobileNetV2BackFloatBridge.lean",
 "LeanMlir/Proofs/Float/MobileNetV2FloatBudget.lean",
 "LeanMlir/Proofs/Float/MobileNetV2BackFloatBudget.lean",
 "tests/TestMobilenetV2TrainPC.lean",
]
RULES = [
 (r"\bflatConvStride2Back_eq_vjp_backward\b", "flatConvStride2XlaBack_eq_vjp_backward"),
 (r"\bfloatBridgesTo_flatConvStride2Back\b", "floatBridgesTo_flatConvStride2XlaBack"),
 (r"\bfloatBridges_flatConvStride2Back\b", "floatBridges_flatConvStride2XlaBack"),
 (r"\bMaps\.flatConvStride2Back\b", "Maps.flatConvStride2XlaBack"),
 (r"\bflatConvStride2Back\b", "flatConvStride2XlaBack"),
 (r"\bflatConvStride2_weight_grad_has_vjp_correct\b", "flatConvStride2Xla_weight_grad_has_vjp_correct"),
 (r"\bflatConvStride2_weight_grad_has_vjp\b", "flatConvStride2Xla_weight_grad_has_vjp"),
 (r"\bflatConvStride2_bias_grad_has_vjp\b", "flatConvStride2Xla_bias_grad_has_vjp"),
 (r"\bflatConvStride2_has_vjp\b", "flatConvStride2Xla_has_vjp"),
 (r"\bflatConvStride2_differentiable\b", "flatConvStride2Xla_differentiable"),
 (r"\bfloatBridgesTo_flatConvStride2\b", "floatBridgesTo_flatConvStride2Xla"),
 (r"\bfloatBridges_flatConvStride2\b", "floatBridges_flatConvStride2Xla"),
 (r"\bfloatClose_flatConvStride2\b", "floatClose_flatConvStride2Xla"),
 (r"\bMaps\.flatConvStride2\b", "Maps.flatConvStride2Xla"),
 (r"\bflatConvStride2F\b", "flatConvStride2XlaF"),
 (r"\bflatConvStride2\b", "flatConvStride2Xla"),
 (r"\bmnv2_render_stem_convW_certified\b", "mnv2_render_stem_convW_xla_certified"),
 (r"\bmnv2_render_stem_convb_certified\b", "mnv2_render_stem_convb_xla_certified"),
 (r"\bmnv2_render_depthwiseW_strided_certified\b", "mnv2_render_depthwiseW_strided_xla_certified"),
 (r"\bmnv2_render_depthwiseb_strided_certified\b", "mnv2_render_depthwiseb_strided_xla_certified"),
 (r"\bResNet34PoC\.convStridedW_den\b", "Mnv2PoC.convStridedXlaW_den"),
 (r"\bResNet34PoC\.convStridedB_den\b", "Mnv2PoC.convStridedXlaB_den"),
 (r"ResNet34PoC\.convStrided\{W,B\}_den", "Mnv2PoC.convStridedXla{W,B}_den"),
 (r"\bdepthwiseStride2FlatBack_eq_vjp_backward\b", "depthwiseStride2FlatXlaBack_eq_vjp_backward"),
 (r"\bfloatBridgesTo_depthwiseStride2Back\b", "floatBridgesTo_depthwiseStride2XlaBack"),
 (r"\bfloatBridges_depthwiseStride2Back\b", "floatBridges_depthwiseStride2XlaBack"),
 (r"\bMaps\.depthwiseStride2Back\b", "Maps.depthwiseStride2XlaBack"),
 (r"\bdepthwiseStride2FlatBack\b", "depthwiseStride2FlatXlaBack"),
 (r"\bdepthwiseStride2_weight_grad_has_vjp\b", "depthwiseStride2Xla_weight_grad_has_vjp"),
 (r"\bdepthwiseStride2_bias_grad_has_vjp\b", "depthwiseStride2Xla_bias_grad_has_vjp"),
 (r"\bdepthwiseStride2Flat_has_vjp\b", "depthwiseStride2FlatXla_has_vjp"),
 (r"\bdepthwiseStride2Flat_differentiable\b", "depthwiseStride2FlatXla_differentiable"),
 (r"\bfloatBridgesTo_depthwiseStride2Flat\b", "floatBridgesTo_depthwiseStride2FlatXla"),
 (r"\bfloatBridges_depthwiseStride2Flat\b", "floatBridges_depthwiseStride2FlatXla"),
 (r"\bfloatClose_depthwiseStride2Flat\b", "floatClose_depthwiseStride2FlatXla"),
 (r"\bMaps\.depthwiseStride2Flat\b", "Maps.depthwiseStride2FlatXla"),
 (r"\bdepthwiseStride2FlatF\b", "depthwiseStride2FlatXlaF"),
 (r"\bdepthwiseStride2Flat\b", "depthwiseStride2FlatXla"),
 (r"\bflatConvStridedF_faithful\b", "flatConvStridedXlaF_faithful"),
 (r"\.flatConvStridedF\b", ".flatConvStridedXlaF"),
 (r"\bdepthwiseStridedF_faithful\b", "depthwiseStridedXlaF_faithful"),
 (r"\.depthwiseStridedF\b", ".depthwiseStridedXlaF"),
 (r"\bdepthwiseStridedBackBatched_faithful\b", "depthwiseStridedXlaBackBatched_faithful"),
 (r"\.depthwiseStridedBackBatched\b", ".depthwiseStridedXlaBackBatched"),
 (r"\bdepthwiseStridedBack_faithful\b", "depthwiseStridedXlaBack_faithful"),
 (r"\.depthwiseStridedBack\b", ".depthwiseStridedXlaBack"),
 (r"\bdepthwiseStridedWeightSgdDen\b", "depthwiseStridedXlaWeightSgdDen"),
 (r"\bdepthwiseStridedBiasSgdDen\b", "depthwiseStridedXlaBiasSgdDen"),
 (r"\bdepthwiseStridedWeightSgd\b", "depthwiseStridedXlaWeightSgd"),
 (r"\bdepthwiseStridedBiasSgd\b", "depthwiseStridedXlaBiasSgd"),
 (r"\bconvStridedWeightSgd\b", "convStridedXlaWeightSgd"),
 (r"\bconvStridedBiasSgd\b", "convStridedXlaBiasSgd"),
 (r"`depthwiseStrided\{Weight,Bias\}Sgd`", "`depthwiseStridedXla{Weight,Bias}Sgd`"),
 (r"`convStrided\{Weight,Bias\}Sgd`", "`convStridedXla{Weight,Bias}Sgd`"),
]
total = 0
for f in FILES:
    p = ROOT / f; s = p.read_text(); orig = s
    for pat, rep in RULES:
        s, n = re.subn(pat, rep, s); total += n
    if s != orig:
        p.write_text(s); print(f"{f}: changed")
print("substitutions:", total)

# ── MobileNetV2Close.lean: the four XLA-SAME twins beside the shared symmetric certs ──
p = ROOT / "LeanMlir/Proofs/Architectures/MobileNetV2Close.lean"; s = p.read_text()
old = """  rw [flatConvStride2_weight_grad_has_vjp_correct]
"""
assert s.count(old) == 1
s = s.replace(old, old + """
/-- **Stem conv weight output, certified — XLA-`SAME` phase.** `mnv2_render_stem_convW_certified`
    at `flatConvStride2Xla`, the stem MobileNetV2 ships (every artifact since 2026-09-05; the
    Adam ones since 2026-08-08). ⚠ The symmetric lemma above stays: ResNet-34's PoC reuses it,
    and ResNet's stem is PyTorch-origin symmetric. -/
theorem mnv2_render_stem_convW_xla_certified {ic oc h w kH kW : Nat}
    (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w)))
    (v : Vec (oc * ic * kH * kW)) (dy : Vec (oc * h * w)) (lr : ℝ)
    (i : Fin (oc * ic * kH * kW)) :
    v i - lr * (flatConvStride2Xla_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun v' : Vec (oc * ic * kH * kW) => flatConvStride2Xla (Kernel4.unflatten v') b x)
            v i j * dy j := by
  rw [flatConvStride2Xla_weight_grad_has_vjp_correct]
""")
old = """  rw [(flatConvStride2_bias_grad_has_vjp W x).correct]
"""
assert s.count(old) == 1
s = s.replace(old, old + """
/-- **Stem conv bias output, certified — XLA-`SAME` phase.** -/
theorem mnv2_render_stem_convb_xla_certified {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Vec (ic * (2 * h) * (2 * w)))
    (b : Vec oc) (dy : Vec (oc * h * w)) (lr : ℝ) (o : Fin oc) :
    b o - lr * (flatConvStride2Xla_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (oc * h * w),
          pdiv (fun b' : Vec oc => flatConvStride2Xla W b' x) b o j * dy j := by
  rw [(flatConvStride2Xla_bias_grad_has_vjp W x).correct]
""")
old = """  rw [(depthwiseStride2_weight_grad_has_vjp b x).correct]
"""
assert s.count(old) == 1
s = s.replace(old, old + """
/-- **Strided depthwise weight output, certified — XLA-`SAME` phase.** MobileNetV2's four strided
    depthwises. ⚠ The symmetric lemma above stays: EfficientNet-B0's strided depthwises are
    symmetric in render and reference alike (`EfficientNetClose.lean` reuses it). -/
theorem mnv2_render_depthwiseW_strided_xla_certified {c h w kH kW : Nat}
    (b : Vec c) (x : Vec (c * (2 * h) * (2 * w)))
    (v : Vec (c * kH * kW)) (dy : Vec (c * h * w)) (lr : ℝ) (i : Fin (c * kH * kW)) :
    v i - lr * (depthwiseStride2Xla_weight_grad_has_vjp b x).backward v dy i
      = v i - lr * ∑ j : Fin (c * h * w),
          pdiv (fun v' : Vec (c * kH * kW) =>
            depthwiseStride2FlatXla (Tensor3.unflatten v' : DepthwiseKernel c kH kW) b x) v i j * dy j := by
  rw [(depthwiseStride2Xla_weight_grad_has_vjp b x).correct]
""")
old = """  rw [(depthwiseStride2_bias_grad_has_vjp W x).correct]
"""
assert s.count(old) == 1
s = s.replace(old, old + """
/-- **Strided depthwise bias output, certified — XLA-`SAME` phase.** -/
theorem mnv2_render_depthwiseb_strided_xla_certified {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (x : Vec (c * (2 * h) * (2 * w)))
    (b : Vec c) (dy : Vec (c * h * w)) (lr : ℝ) (o : Fin c) :
    b o - lr * (depthwiseStride2Xla_bias_grad_has_vjp W x).backward b dy o
      = b o - lr * ∑ j : Fin (c * h * w),
          pdiv (fun b' : Vec c => depthwiseStride2FlatXla W b' x) b o j * dy j := by
  rw [(depthwiseStride2Xla_bias_grad_has_vjp W x).correct]
""")
p.write_text(s); print("MobileNetV2Close.lean: four XLA twins added")

# ── MobileNetV2FaithfulPoC.lean: the per-example XLA stem dens (the ResNet ones stay symmetric) ──
p = ROOT / "LeanMlir/Proofs/Architectures/MobileNetV2FaithfulPoC.lean"; s = p.read_text()
old = "end Proofs.Mnv2PoC\n"
assert s.count(old) == 1
s = s.replace(old, """
/-- **Any emitted XLA-`SAME` strided-stem conv weight op = certified.** The per-example
    `convStridedXlaWeightSgd` (MobileNetV2's SGD train step, since 2026-09-05) denotes
    `W − lr·(certified ∂(flatConvStride2Xla)/∂W · c)`, for any cotangent `c`. The XLA peer of
    `ResNet34PoC.convStridedW_den`, which stays symmetric for ResNet's own stem. -/
theorem convStridedXlaW_den {ic oc h w kH kW : Nat}
    (xN wN lrStr cotN : String) (b : Vec oc) (x : Vec (ic*(2*h)*(2*w)))
    (W : Kernel4 oc ic kH kW) (c : Vec (oc*h*w)) (lr : ℝ) (idx : Fin (oc*ic*kH*kW)) :
    den (SHlo.convStridedXlaWeightSgd xN wN lrStr b x W lr (.operand cotN c)) idx
      = Kernel4.flatten W idx - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun v' : Vec (oc*ic*kH*kW) => flatConvStride2Xla (Kernel4.unflatten v') b x)
               (Kernel4.flatten W) idx j * c j :=
  mnv2_render_stem_convW_xla_certified b x (Kernel4.flatten W) c lr idx

/-- **Any emitted XLA-`SAME` strided-stem conv bias op = certified.** Same `reduce` text as
    `convBiasSgd`; the `den` is the odd-phase bias VJP. -/
theorem convStridedXlaB_den {ic oc h w kH kW : Nat}
    (bN lrStr cotN : String) (W : Kernel4 oc ic kH kW) (x : Vec (ic*(2*h)*(2*w)))
    (b : Vec oc) (c : Vec (oc*h*w)) (lr : ℝ) (o : Fin oc) :
    den (SHlo.convStridedXlaBiasSgd bN lrStr W x b lr (.operand cotN c)) o
      = b o - lr * ∑ j : Fin (oc*h*w),
          pdiv (fun b' : Vec oc => flatConvStride2Xla W b' x) b o j * c j :=
  mnv2_render_stem_convb_xla_certified W x b c lr o

end Proofs.Mnv2PoC
""")
p.write_text(s); print("MobileNetV2FaithfulPoC.lean: XLA stem dens added")

# ── EfficientNetBackB0.lean: the batched XLA depthwise input-VJP faithfulness (MobileNetV2's Adam render) ──
p = ROOT / "LeanMlir/Proofs/Architectures/EfficientNetBackB0.lean"; s = p.read_text()
old = """/-- **Batched depthwise input-VJP faithfulness.** The depthwise analogue of
    `convBackBatched_faithful`: `depthwiseBackBatched` denotes the proven VJP of"""
assert s.count(old) == 1
s = s.replace(old, """/-- **Batched XLA-`SAME` STRIDE-2 depthwise input-VJP faithfulness.** The odd-phase peer of
    `depthwiseStridedBackBatched_faithful`: `depthwiseStridedXlaBackBatched` (pad `[p+1, p-1]`,
    the token MobileNetV2's Adam render emits at its four strided depthwises) denotes the proven
    VJP of `batchMap N (depthwiseStride2FlatXla W b)`. Same proof: a scatter onto the odd
    positions is as linear as one onto the even ones. -/
theorem depthwiseStridedXlaBackBatched_faithful {N c h w kH kW : Nat} (wN : String)
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (v : Vec (N * (c * (2 * h) * (2 * w)))) (e : SHlo (N * (c * h * w))) :
    den (SHlo.depthwiseStridedXlaBackBatched (N := N) wN W b e)
      = (batchMap_has_vjp (depthwiseStride2FlatXla W b) (depthwiseStride2FlatXla_has_vjp W b)
          (depthwiseStride2FlatXla_differentiable W b)).backward v (den e) := by
  funext idx
  simp only [den, batchMap, batchMap_has_vjp, hasVJPMat_to_hasVJP, rowwise_has_vjp_mat]
  rfl

""" + old)
p.write_text(s); print("EfficientNetBackB0.lean: depthwiseStridedXlaBackBatched_faithful added")

# ── tests/AuditAxioms.lean: the seven new names ──
p = ROOT / "tests/AuditAxioms.lean"; s = p.read_text()
edits = [
("#print axioms Mnv2PoC.depthwiseStridedB_den\n",
 "#print axioms Mnv2PoC.depthwiseStridedB_den\n-- The XLA-SAME per-example stem dens (2026-09-05): MobileNetV2's SGD stem is convStridedXla{Weight,Bias}Sgd;\n-- ResNet34PoC.convStrided{W,B}_den stay symmetric for ResNet's own stem.\n#print axioms Mnv2PoC.convStridedXlaW_den\n#print axioms Mnv2PoC.convStridedXlaB_den\n"),
("#print axioms mnv2_render_depthwiseb_strided_certified\n",
 "#print axioms mnv2_render_depthwiseb_strided_certified\n-- Their XLA-SAME twins (2026-09-05). The symmetric four stay: ResNet-34 reuses the stem pair and\n-- EfficientNet-B0 the strided-depthwise pair, both symmetric in render and reference.\n#print axioms mnv2_render_stem_convW_xla_certified\n#print axioms mnv2_render_stem_convb_xla_certified\n#print axioms mnv2_render_depthwiseW_strided_xla_certified\n#print axioms mnv2_render_depthwiseb_strided_xla_certified\n"),
("#print axioms StableHLO.depthwiseStridedBackBatched_faithful\n",
 "#print axioms StableHLO.depthwiseStridedBackBatched_faithful\n-- Its XLA-SAME peer, the token MobileNetV2's Adam render emits at its four strided depthwises.\n#print axioms StableHLO.depthwiseStridedXlaBackBatched_faithful\n"),
]
for old, new in edits:
    assert s.count(old) == 1, old
    s = s.replace(old, new)
p.write_text(s); print("tests/AuditAxioms.lean: seven names added")

# ── prose to review by hand: lines that name another net next to a now-XLA name ──
import subprocess
print("\n--- REVIEW: prose in the re-spelled files that names ResNet/ConvNeXt/EfficientNet next to an Xla name ---")
for f in FILES:
    for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        if "Xla" in line and any(k in line for k in ("r34", "ResNet", "ConvNeXt", "EfficientNet", "B0")):
            print(f"{f}:{i}: {line.strip()[:120]}")
print("done — now build (planning doc, step 3b) and check the numerals")
