import LeanMlir.Proofs.Float.Resnet34WholeBackFloatBridgeB

/-! # ℝ→Float32 bridge: the WHOLE ResNet-50 backward at TRUE BATCH-NORM

`Resnet34WholeBackFloatBridgeB.lean` names `r34InputGradB`, the batched input-gradient chain of
ResNet-34's [3,4,6,3] BASIC-block ladder. This file is the same for ResNet-50: the same ladder in
BOTTLENECK form, `resnet50ForwardB_full`'s exact reverse, at a variable batch `N` **and a variable
resolution `q`**. It exists so the certified tie (`Foundation/Resnet50WholeBackCertifiedTieB.lean`,
tier T6 of `planning/proofs_tier_to_paper_nets.md` §3.5(e)) is a statement about a NAMED chain.

⭐⭐ **Nothing in this file is new, and that is the point.** ResNet-50's stem and head ARE
ResNet-34's (`r34StemB` / `r34HeadB`, §3.5b), so the stem's symmetric strided conv-back, its
BatchNorm and relu-mask slots, **the batched 3×3/s2 pool backward `maxPool3s2FlatBackB` and the
whole `FloatClose.batchMapAux` lift it needed** are §4.2d's, imported and applied at R50's widths.
The one thing this file adds is the sixteen bottleneck slots and their arithmetic.

⭐ **`q` is a binder here as it is in T1–T3**, so one chain covers `resnet50in_fwd` (`q = 7`, 224 px)
and `resnet50in160_fwd` (`q = 5`, 160 px — the net the quoted 76.66% trains). ⚠ Every dimension is
written as an explicit `2 * (…)` nest and not as `8 * q`: those are equal Nats and NOT
definitionally equal terms at a variable `q`, and each stage demands its operand at exactly the
spelling it names. §3.5b hit this on the forward side; it is the same trap here.

⚠ Padding is SYMMETRIC at both stride-2 sites of every downsample bottleneck and at the stem,
as the PyTorch-origin convention requires — `flatConvStride2Back`, not the XLA-`SAME`
`flatConvStride2XlaBack` MobileNetV2's stem takes.

⛔ **No number is stated about this chain.** §3.5(d)'s T4/T5 are float budgets and
`planning/float_budget_numbers.md` closed that thread; a batched backward number moves with `N`
besides (`bnGradInputReMag`'s gain carries `Xh² = N·h·w`). The chain is here for the tie.
-/

namespace Proofs

open FloatModel

/-- **The batched whole-net input-gradient backward of ResNet-50** — the exact reverse of
    `resnet50ForwardB_full = head ∘ [3,4,6,3] bottlenecks ∘ stem`: dense-back → GAP-back → the
    sixteen bottleneck backwards → the 3×3/s2 pool back → the stem's relu mask, BatchNorm back
    and 7×7/s2 conv back. The bottleneck backwards and the stem's BatchNorm back are supplied;
    the conv, pool, GAP and dense leaves are concrete and lifted over the `N` examples. -/
noncomputable def r50InputGradB (N q : Nat) {nCls : Nat}
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 2048 nCls)
    (bnBs : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (xpool : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (b1B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3B : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b5B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7B : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b9B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13B : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14B : Vec (N * (2048 * q * q)) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b15B : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16B : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (m_stem : Fin (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Prop) [DecidablePred m_stem] :
    Vec (N * nCls) → Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :=
  (StableHLO.batchMap N (flatConvStride2Back (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) Ws)
      ∘ bnBs ∘ reluMaskBack m_stem)
  ∘ maxPool3s2FlatBackB N 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xpool
  ∘ b1B ∘ b2B ∘ b3B ∘ b4B ∘ b5B ∘ b6B ∘ b7B ∘ b8B ∘ b9B ∘ b10B ∘ b11B ∘ b12B ∘ b13B ∘ b14B ∘ b15B ∘ b16B
  ∘ StableHLO.batchMap N (gapBack 2048 q q)
  ∘ StableHLO.batchMap N (Proofs.dense (Mat.transpose Wd) (0 : Vec 2048))

/-- **The float ResNet-50 input-gradient skeleton** — `r50InputGradB` with each concrete slot
    replaced by the model's rounded peer and each supplied backward by its float map.
    ⛔ The stem's scatter is `decimateBack`, the SYMMETRIC phase's. -/
noncomputable def r50InputGradBF (N q : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 2048 nCls)
    (bnBsF : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (xpool : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (b1BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b5BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b9BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14BF : Vec (N * (2048 * q * q)) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b15BF : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16BF : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (m_stem : Fin (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Prop) [DecidablePred m_stem] :
    Vec (N * nCls) → Vec (N * (3 * (2 * (2 * (2 * (2 * (2 * q))))) * (2 * (2 * (2 * (2 * (2 * q))))))) :=
  (StableHLO.batchMap N
      (M.flatConvF (h := (2 * (2 * (2 * (2 * (2 * q)))))) (w := (2 * (2 * (2 * (2 * (2 * q)))))) (IR.reverseSwap Ws) (fun _ => 0)
        ∘ decimateBack 64 (2 * (2 * (2 * (2 * q)))) (2 * (2 * (2 * (2 * q)))))
      ∘ bnBsF ∘ reluMaskBack m_stem)
  ∘ M.maxPool3s2FlatBackBF N 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xpool
  ∘ b1BF ∘ b2BF ∘ b3BF ∘ b4BF ∘ b5BF ∘ b6BF ∘ b7BF ∘ b8BF ∘ b9BF ∘ b10BF ∘ b11BF ∘ b12BF ∘ b13BF ∘ b14BF ∘ b15BF ∘ b16BF
  ∘ StableHLO.batchMap N (gapBackF M 2048 q q)
  ∘ StableHLO.batchMap N (M.dense (Mat.transpose Wd) (0 : Vec 2048))

set_option maxRecDepth 100000 in
/-- **The batched ResNet-50 backward float-bridges TO its float skeleton** — one `.comp` thread
    over the concrete endpoints, the supplied stem BatchNorm backward and the sixteen supplied
    bottleneck backwards. `r34_grad_floatBridgesToB` at the bottleneck ladder. -/
noncomputable def r50_grad_floatBridgesToB (N q : Nat) {nCls : Nat} (M : FloatModel)
    (Ws : Kernel4 64 3 7 7) (Wd : Mat 2048 nCls)
    (bnBs bnBsF : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (xpool : Vec (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))))
    (b1B b1BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (64 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b2B b2BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b3B b3BF : Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b4B b4BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (256 * (2 * (2 * (2 * q))) * (2 * (2 * (2 * q))))))
    (b5B b5BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b6B b6BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b7B b7BF : Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b8B b8BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (512 * (2 * (2 * q)) * (2 * (2 * q)))))
    (b9B b9BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b10B b10BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b11B b11BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b12B b12BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b13B b13BF : Vec (N * (1024 * (2 * q) * (2 * q))) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b14B b14BF : Vec (N * (2048 * q * q)) → Vec (N * (1024 * (2 * q) * (2 * q))))
    (b15B b15BF : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (b16B b16BF : Vec (N * (2048 * q * q)) → Vec (N * (2048 * q * q)))
    (m_stem : Fin (N * (64 * (2 * (2 * (2 * (2 * q)))) * (2 * (2 * (2 * (2 * q)))))) → Prop) [DecidablePred m_stem]
    {ws wd : ℝ} (hws : 0 ≤ ws) (hwd : 0 ≤ wd) (hnc : 0 < nCls) (hq : 0 < q)
    (hWs : ∀ o c kh kw, |Ws o c kh kw| ≤ ws) (hWd : ∀ i j, |Wd i j| ≤ wd)
    (hbnBs : FloatBridgesTo bnBs bnBsF)
    (hb1B : FloatBridgesTo b1B b1BF)
    (hb2B : FloatBridgesTo b2B b2BF)
    (hb3B : FloatBridgesTo b3B b3BF)
    (hb4B : FloatBridgesTo b4B b4BF)
    (hb5B : FloatBridgesTo b5B b5BF)
    (hb6B : FloatBridgesTo b6B b6BF)
    (hb7B : FloatBridgesTo b7B b7BF)
    (hb8B : FloatBridgesTo b8B b8BF)
    (hb9B : FloatBridgesTo b9B b9BF)
    (hb10B : FloatBridgesTo b10B b10BF)
    (hb11B : FloatBridgesTo b11B b11BF)
    (hb12B : FloatBridgesTo b12B b12BF)
    (hb13B : FloatBridgesTo b13B b13BF)
    (hb14B : FloatBridgesTo b14B b14BF)
    (hb15B : FloatBridgesTo b15B b15BF)
    (hb16B : FloatBridgesTo b16B b16BF) :
    FloatBridgesTo
      (r50InputGradB N q Ws Wd bnBs xpool b1B b2B b3B b4B b5B b6B b7B b8B b9B b10B b11B b12B b13B b14B b15B b16B m_stem)
      (r50InputGradBF N q M Ws Wd bnBsF xpool b1BF b2BF b3BF b4BF b5BF b6BF b7BF b8BF b9BF b10BF b11BF b12BF b13BF b14BF b15BF b16BF m_stem) := by
  unfold r50InputGradB r50InputGradBF
  have hstem := ((floatBridgesTo_reluMaskBack m_stem).comp hbnBs).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_flatConvStride2Back (h := (2 * (2 * (2 * (2 * q))))) (w := (2 * (2 * (2 * (2 * q))))) M Ws hws
        (Nat.mul_pos (Nat.mul_pos (by norm_num) (by omega)) (by omega)) hWs))
  have h0 := (FloatBridgesTo.batchMap N (floatBridgesTo_linBack M Wd hwd hnc hWd)).comp
    (FloatBridgesTo.batchMap N
      (floatBridgesTo_gapBack M 2048 q q (by norm_num) hq hq))
  have hB17 := h0
  have hB16 := hB17.comp hb16B
  have hB15 := hB16.comp hb15B
  have hB14 := hB15.comp hb14B
  have hB13 := hB14.comp hb13B
  have hB12 := hB13.comp hb12B
  have hB11 := hB12.comp hb11B
  have hB10 := hB11.comp hb10B
  have hB9 := hB10.comp hb9B
  have hB8 := hB9.comp hb8B
  have hB7 := hB8.comp hb7B
  have hB6 := hB7.comp hb6B
  have hB5 := hB6.comp hb5B
  have hB4 := hB5.comp hb4B
  have hB3 := hB4.comp hb3B
  have hB2 := hB3.comp hb2B
  have hB1 := hB2.comp hb1B
  have hMP := hB1.comp
    (floatBridgesTo_maxPool3s2BackB M N 64 (2 * (2 * (2 * q))) (2 * (2 * (2 * q))) xpool (by norm_num) (by omega) (by omega))
  exact hMP.comp hstem

end Proofs
