import sys
k=int(sys.argv[1]); out=sys.argv[2]
src=open("LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtWholeBackCertifiedTieB.lean").read().split("\n")
prefix="\n".join(src[:319])            # everything before the apex docstring (line 320)
outer=[("StableHLO.batchMap B (convNextStageChK 3 w.s1)",
        "⟨cnxStageB_at B 3 w.s1 h1 (cnxSavedB1 B w x),\n     batchMap_differentiableAt _ _ (fun _ => (convNextStageChK_diff 3 w.s1 h1).differentiableAt)⟩"),
       ("StableHLO.batchMap B (cnxDn1 w)",
        "⟨cnxDn1B_at B w hd1 (cnxSavedB2 B w x),\n     batchMap_differentiableAt _ _ (fun _ => (cnxDn1Diff w hd1).differentiableAt)⟩"),
       ("StableHLO.batchMap B (convNextStageChK 3 w.s2)",
        "⟨cnxStageB_at B 3 w.s2 h2 (cnxSavedB3 B w x),\n     batchMap_differentiableAt _ _ (fun _ => (convNextStageChK_diff 3 w.s2 h2).differentiableAt)⟩"),
       ("StableHLO.batchMap B (cnxDn2 w)",
        "⟨cnxDn2B_at B w hd2 (cnxSavedB4 B w x),\n     batchMap_differentiableAt _ _ (fun _ => (cnxDn2Diff w hd2).differentiableAt)⟩"),
       ("StableHLO.batchMap B (convNextStageChK 9 w.s3)",
        "⟨cnxStageB_at B 9 w.s3 h3 (cnxSavedB5 B w x),\n     batchMap_differentiableAt _ _ (fun _ => (convNextStageChK_diff 9 w.s3 h3).differentiableAt)⟩")]
body=("vjp_comp_diff_at (StableHLO.batchMap B (cnxSavedA0 w))\n"
      "    (StableHLO.batchMap B (chanLNTensor3 96 56 56 w.sε w.sγ w.sβ)) x\n"
      "    ⟨cnxStemB_at B w x,\n"
      "     batchMap_differentiableAt _ x (fun _ => (cnxD0 w).differentiableAt)⟩\n"
      "    ⟨cnxStemLNB_at B w hsε (cnxSavedB0 B w x),\n"
      "     batchMap_differentiableAt _ _ (fun _ =>\n"
      "       (chanLNTensor3_diff 96 56 56 w.sε w.sγ w.sβ hsε).differentiableAt)⟩")
for i in range(min(k,len(outer))):
    s,wt=outer[i]; body=f"vjp_comp_diff_at _ ({s}) x\n    ({body})\n    ({wt})"
open(out,"w").write(prefix+f"""

/-- Truncated apex: {k} level(s) above the stem pair. -/
noncomputable def truncApex (B : Nat) {{nC : Nat}} (w : CnxTWeightsCh nC)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn)
    (x : Vec (B * (3 * 224 * 224))) :=
  {body}

end Proofs
""")
