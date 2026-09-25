# Re-audit of the 2026-09-22 proof-quality reports — open findings at 28fc373f (2026-09-24)

Static reading + grep only (no builds). Each report re-checked finding by finding against
current code; statuses: FIXED / PARKED-or-REJECTED (cited to proof_cleanup.md § or audit_v2 /
certlayer_nets) / STILL OPEN / GONE. Per-report tables + quoted current code follow the ranked list.
Nothing here was compiled: every "unused"/"closes by" claim needs one `lake env lean <file>`.

## Counts

| report | FIXED | PARKED/REJ | STILL OPEN | GONE |
|---|---|---|---|---|
| arch_training (A1–A20) | 5 (+3 partly) | sub-parts of A2/A3/A6 (§3.5) | 11 | 0 |
| codegen_certs (33 rows) | 15 | 3 (+1 no-action) | 14 | 0 |
| effnet_vit (21 rows) | 4 (+ all bumps) | 2 (§3.6) | 15 | 0 |
| foundation_float (25 rows) | 10 | 1 (§3.4) | 14 | 0 |
| mobilenet (27 rows) | 13 | 5 | 14 | 0 |
| resnet_small_convnext (21 rows) | 11 | 2 | 9 | 1 |

Every `set_option maxHeartbeats/maxRecDepth` finding in Nets/ is FIXED (grep: zero in Nets/).
Remaining bumps: renderers' 21× `maxRecDepth 4000000` + 2× 1M (codegen report), IBP 6.4M (kept, §1(k)).

## Ranked: still open, low-hanging (trivial/small first, payoff descending)

1. **ConvNeXt imports a retired R34 file** (trivial, low-med) — `Nets/ConvNeXt/ConvNeXtStepTie.lean:4` and
   `ConvNeXtFoldG.lean:4`: `import LeanMlir.Proofs.Nets.ResNet.ResNet34Fold`; only `convStridedW_den`/`convStridedB_den`
   are used (4×, all in ConvNeXtStepTie). Move them to `Foundation/SgdNodes`; FoldG's import looks dead.
2. **`simpa [hf] using` ×14 in the SGD rungs** (trivial–small, medium) — SgdDescentCnn:2455/2457, 4447/4449,
   5028/5030, 5499/5501; Mlp:466/468, 778/780; Linear:251/253. Full simp set over huge terms; try `exact`/`simpa only [hf]`.
3. **Unused `hE₀0` in FloatBridge** (trivial, low) — `have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=` at
   Float/FloatBridge.lean:960/1028/1079/1131/1199 never used afterwards (spot-checked 960); `hC₁0` at 1027.
4. **Unused `DecidableEq` on printer types** (trivial, low; unmeasured) — `Codegen/StableHLOPretty.lean:281`
   (`Raw`) and `:889` (`Tok`) `deriving DecidableEq, …`; no user found by grep (§3.2 noted it, never tried).
5. **`relu6MaskB_shard` unused** (trivial, low) — `Nets/MobileNet/MobileNetV2SyncStepTieB.lean:221`; 24 `_shard`
   proofs close its goal by bare `rfl`. Cite or delete (audit_v2 deleted the R34 twin).
6. **`#eval` writers in StableHLOPretty** (small, low-med) — `StableHLOPretty.lean:4858–4868`, 98 transitive
   importers, three write stray `/tmp/linear_*_v.mlir`; move to a leaf like `CnnArtifacts.lean`.
7. **Bf16GradNodes `_den` ×9 same 5-line body** (small, low-med) — `Foundation/Bf16GradNodes.lean:49–220`;
   one `rnd_sum_backward` lemma makes each a one-liner.
8. **Batched leaf ties `show` the `batchMapAux` layout by hand ×9** (small, medium) — ConvNeXtWholeBackCertifiedTieB
   :205/221/241/258/299/311, ViTWholeBackCertifiedTieB :97/110/126; one row-lift lemma in `BatchMapVJPAt.lean`.
9. **`*LossCot_den` ×6 copies** (small, low-med) — MlpFold:133, CnnFold:150, CifarFold:100, Cifar8StepTie:34,
   Cifar8BnStepTie:25, ConvNeXtStepTie:293 (K=10 fixed) → one `{K}` lemma in `LinearTrainStep.lean`.
10. **`conv2d_padOdd_eq` copies `conv2d`'s let body** (small, medium robustness) — `Architectures/EvenKernelConvBack.lean:96–118`.
11. **γ₇₈₅ `norm_num [u32]` block twice** (trivial, low) — SgdDescentCnn:684–698, :739–751 → `FloatModel.gamma785_le`.
12. **Undocumented `show`s leaning on `denseE`/`WithLp` unfolding** (trivial, low-med, bump-fragile) —
    `LipschitzCertPairSDP.lean:166` (`mlp_gap_eq`), `LipschitzCertInstance.lean:398` (`mlpT_logit_continuous`).
13. **ViT trivia** (trivial) — `ViTVecLNBackCertifiedTie.lean:97–99,127–129` undocumented `show w (finProdFinEquiv …)`
    → `rw [Mat.unflatten_apply, …]`; `ViTBackB0.lean:215–225` hand reindex → `sum_finProdFinEquiv`, `:247–252`
    `show … from by decide` ×3; `ViTStepTie.lean:195` `set_option linter.unusedSimpArgs false`.
14. **Small idioms** (trivial each) — `rndP_mul_four` (`DataParallelSyncBf16.lean:312–315`, `norm_num at this; exact this`);
    bare closing `rfl` without comment at DataParallelSyncBf16 :167/:229; `simpa using abs_sub_le a b 0` ×9 in
    FloatBridge + BnFloatBridge:213; `Finset.sum_const, card_univ, Fintype.card_fin` ×8 → `Fin.sum_const`;
    `ctConv_inj` Nat.mul_comm rewrites (BatchSealKit:790–793 → `linarith`); MobileNetV2Fold:43 `congr 1; congr 1`;
    MobileNetV2StepTieB `*CotIn_eq_vjp` unused `cotN : String` binder (:156/240/263/341);
    MobileNetV2FullBVJP.lean:50 docstring "carries no numerals" is false.
15. **EfficientNet sync `show cInB … (den (SHlo.bnBatchLABack …))` ×5** (small) — `EfficientNetSyncStepTieG.lean:346–397,
    484–495`; one `rfl` lemma per graph. **BN-β clause ×10** in `EfficientNetStepTieG.lean` → `BnBetaTiedB` Prop.
16. **Renderer `maxRecDepth 4000000` ×21 (+2× 1M)** (small, low) — MNv2/MNv4/R34/R50/ENet render files; strip and
    compile, gate = byte-identical artifacts (§1(b) found 57/59 dead by this method).

Medium, worth listing: `pdiv_bnIstdBroadcast` (BatchNorm.lean:558, ~175 lines, CLM-coercion `show` chains — most
bump-fragile open item); `*TiedB` Props restating their `_den` lemmas (GradNodesB, ViTFoldGB, ViTFold; 218
`intro idx; exact …_den` delegations); ConvNeXt's 15 `@[irreducible]` wrappers (heartbeat-era, comparator tier → measure).

Not resurrected (disproved/decided): FwdGraphB `rw`→`simp only` (MNv4FullB:847: 9 min then kernel failure); B-7 BnMode
on eval renders (reversed by 28fc373f); restating capstones over weight records (premise was the bumps, gone via §1(b));
the "SDPA 15 s" and SgdDescentCnn "95 s statement elaboration" timing guesses (§1(q)).

---


# Report: audit_foundation_float.md

## audit_foundation_float.md — re-audit 2026-09-24 (static, HEAD 28fc373f)

| # | finding (audit §) | status | current location | effort / payoff |
|---|---|---|---|---|
| F1 | DataParallelSyncBf16 imports Binary32Instance (float tier on crit path) | FIXED §1(l) | imports `Float.RndP` (DataParallelSyncBf16.lean:2) | — |
| F2 | DPSBf16 plain/strided twin lemmas (6 pairs) → `batchWGrad_*` | STILL OPEN | Foundation/DataParallelSyncBf16.lean:93–260, 318–367 | medium / low (only 2 kinds live in this file today) |
| F3 | DPSBf16 `_shard` closes by `rw …; rfl` across `convWGradShardSum` + `batchShard` unfold | STILL OPEN | DataParallelSyncBf16.lean:167, 229 | trivial (comment) / low |
| F4 | `rndP_mul_four`: `norm_num at this; exact this` | STILL OPEN | DataParallelSyncBf16.lean:312–315 | trivial / low |
| F5 | FloatBridge `nlinarith` → `linarith` (speed claim) | FIXED §1(d)+(l); speed part disproved §4 ("idiom, not speed", −2 s) | 6 `nlinarith only […]` left (505, 1428, 1431–32, 1599–1600) | — |
| F6 | `softmaxF_close` long proof: 4 sandwich copies + `div_mul_cancel₀` chain | PARTIAL: sandwiches now `linarith [abs_le.mp …]` one-liners; the `div_eq_iff … div_mul_cancel₀` chain and 126-line body remain | Float/FloatBridge.lean:1486–1605 (chain 1559–1562, 1579–1581) | small / low |
| F7a | `mlp_*_step_float_close`: dead `have`s (`hE₀0` ×5, `hC₁0` ×2) | STILL OPEN | FloatBridge.lean:960, 1027–1028, 1079, 1131, 1199 | trivial / low |
| F7b | same: layer-1/0 cotangent blocks repeated → `mlp_cot1_close` / `mlp_cot0_close` | STILL OPEN | FloatBridge.lean:1032–1045, 1082, 1135–1152, 1202–1219 | small-medium / low-med |
| F8 | hand-rolled `\|a\| ≤ \|a−b\|+\|b\|` via `simpa using abs_sub_le a b 0` | STILL OPEN (9 sites) | FloatBridge.lean:144, 147, 189, 402, 767, 771, 807, 1511, 1614; BnFloatBridge.lean:213 | trivial / low |
| F9 | `step_bound` / `dense_step_bound` duplicate `htri` block | STILL OPEN | FloatBridge.lean:136–171, 176–211 | small / low |
| F10 | Binary32Instance `#print axioms` ×3 | FIXED §1(l) (only a docstring mention at :82) | — | — |
| F11 | `binary32_linear_sgd_descends_concrete`: `show` dense, non-terminal `simp`, `hSabs`/`hSsq` twin, `hη`/`hη0` twin | STILL OPEN | Float/Binary32Instance.lean:172, 191–207, 209–216 | small / low |
| F12 | `Mat`/`Tensor3` `flatten` `let`s, `_apply`, `@[simp]` round-trips | FIXED §1(m) (Tensor.lean:526–540, 933–947); site migration PARKED §3.4 | — | — |
| F13 | `vjpComp_backward` / `vjpCompAt_backward` | FIXED §1(m) (Tensor.lean:456, 462) | — | — |
| F14a | PerChannelBN `bnchwFwd_apply` (replace `show`s) | FIXED §1(m) (Architectures/PerChannelBN.lean:428) | — | — |
| F14b | PerChannelBN `reassoc*Idx` / `bnchw*Idx` as `let`-defs, 4 copies of the `Prod.mk.eta` round-trip → one `Equiv` | STILL OPEN | Architectures/PerChannelBN.lean:150–177, 397–421 | small / low |
| F15a | DataParallel `meanLoss_apply`/`dpMean_apply`/`lossGrad_apply` | PARKED §3.4 ("Also left … §3.6 territory") | — | — |
| F15b | DataParallel 7-step `rw` chain twice behind `show` → `mean_mean_equiv` | STILL OPEN | Foundation/DataParallel.lean:145–148, 283–285 | small / low |
| F16 | Bf16GradNodes: nine copies of `simp; congr; sum_congr; exact (…).correct` → `rnd_sum_backward` | STILL OPEN | Foundation/Bf16GradNodes.lean:49–220 (9 `_den` theorems) | small / low-med (each new bf16 kind copies it) |
| F17 | `conv2d_padOdd_eq` restates `conv2d`'s let-body in two `have`s | STILL OPEN | Architectures/EvenKernelConvBack.lean:96–118 | small / med (breaks far from cause on any `conv2d` edit) |
| F18 | MuonGeometry reassociate-by-`show … from by simp only [Matrix.mul_assoc]` (10), `hpt4`/`hpt2` dup, `hMle` nlinarith | STILL OPEN | Foundation/MuonGeometry.lean:171, 173, 196, 219, 361–411, 431 | small-medium / low |
| F19a | CrownBound:267 `nlinarith [this]` → `linarith` | FIXED §1(d) (now `linarith [this]`, Certificates/CrownBound.lean:268) | — | — |
| F19b | CrownBound:108–109 → Mathlib lemma names | FIXED-enough §1(l) (`nlinarith only`, :108–109) | — | — |
| F19c | CrownBound:251 `show` through `∘` | FIXED (documented: docstring at :246–248 says why it peels once) | CrownBound.lean:252 | — |
| F20 | `Finset.sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul` → `Fin.sum_const` | STILL OPEN (8 sites repo-wide) | DataParallelSync.lean:476; BnFloatBridge.lean:210, 299; SgdDescentMlp.lean:257, 662, 672; SmoothingMC.lean:75; BatchNorm.lean:704 | trivial / low |

Counts: FIXED 10 (F1, F5, F10, F12, F13, F14a, F19a–c; F12 site-migration part parked), PARKED 1 (F15a), STILL OPEN 14 (F2, F3, F4, F6 partial, F7a, F7b, F8, F9, F11, F14b, F15b, F16, F17, F18, F20 — F7a/F7b counted as one audit finding split in two), GONE 0.

### Still open, ranked (trivial/small, best payoff first)

1. **F17 `conv2d_padOdd_eq` (small / med)** — Architectures/EvenKernelConvBack.lean:96–104 (and 109–116 for `hcol`):
   ```lean
   have hrow : (∑ kw : Fin (kW + 1), padOdd W o c 0 kw *
       (let pH := (kH + 1 - 1) / 2
        let pW := (kW + 1 - 1) / 2
        let hh := (0 : Fin (kH + 1)).val + hi.val
        ...
        else 0)) = 0 := by
     refine Finset.sum_eq_zero (fun kw _ => ?_)
     rw [padOdd_zero_row, zero_mul]
   rw [hrow, zero_add]
   ```
   Copies `conv2d`'s body verbatim; try `simp only [padOdd_zero_row, padOdd_zero_col, zero_mul, Finset.sum_const_zero, zero_add]` after `Fin.sum_univ_succ`. Untested.
2. **F7a dead `have`s (trivial / low)** — FloatBridge.lean:1027–1029 (same shape at 960, 1079, 1131, 1199):
   ```lean
   have hC₁0 : 0 ≤ layerAct d₃ w₂ 0 G := layerAct_nonneg hw₂ le_rfl hG0   -- unused in w1
   have hE₀0 : 0 ≤ layerBudget M.u d₀ w₀ β₀ a 0 :=                        -- unused in all 5
     layerBudget_nonneg M.u_nonneg hw₀ hβ₀ ha le_rfl
   ```
   (grep per theorem range: `hE₀0` occurs once in each of the five, `hC₁0` once in `w1`.)
3. **F16 Bf16GradNodes nine copies (small / low-med)** — Bf16GradNodes.lean:77–83 (×9):
   ```lean
   simp only [denStep, denStepApp]
   congr 1
   apply Finset.sum_congr rfl
   intro n _
   exact (flatConvStride2WeightGradHasVJP b
     (fun j => rnd (batchSlice N (ic * (2 * h) * (2 * w)) x n j))).correct
     (Kernel4.flatten W) (fun j => rnd (batchSlice N (oc * h * w) cot n j)) idx
   ```
   One `rnd_sum_backward` lemma (audit's statement) makes each a one-liner.
4. **F4 `rndP_mul_four` (trivial / low)** — DataParallelSyncBf16.lean:312–315:
   ```lean
   have := rndP_two_pow_mul p 2 x
   norm_num at this
   exact this
   ```
   → `by simpa [show (2:ℝ)^2 = 4 by norm_num] using rndP_two_pow_mul p 2 x`.
5. **F3 undocumented `rfl` (trivial / low)** — DataParallelSyncBf16.lean:166–167 and 228–229:
   ```lean
   rw [den_convWeightGradBBf16_eq_rnd rnd xN cotN, hdy]
   rfl
   ```
   Needs a one-line comment naming the two unfoldings (`convWGradShardSum`, `batchShard` of a mapped vector), or `batchShard_map`.
6. **F8 `simpa using abs_sub_le a b 0` (trivial / low)** — FloatBridge.lean:144:
   ```lean
   have h1 : |st| ≤ |st - S| + |S| := by simpa using abs_sub_le st S 0
   ```
   → `linarith [abs_sub_abs_le_abs_sub st S]` (already the idiom at FloatBridge.lean:112, 796). 9 sites + BnFloatBridge.lean:213.
7. **F20 `Fin.sum_const` (trivial / low)** — e.g. BnFloatBridge.lean:210:
   `rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul …]` → `rw [Fin.sum_const, nsmul_eq_mul]`; 8 sites.
8. **F15b DataParallel twin chain (small / low)** — DataParallel.lean:145–148 and 283–285:
   ```lean
   show (1 / (R : ℝ)) * ∑ r : Fin R, ((1 / (N : ℝ)) * ∑ n : Fin N, ℓ (e (r, n)) θ)
        = (1 / ((R * N : Nat) : ℝ)) * ∑ k : Fin (R * N), ℓ k θ
   rw [← Finset.mul_sum, ← mul_assoc, div_mul_div_comm, one_mul, Nat.cast_mul,
       ← Equiv.sum_comp e (fun k => ℓ k θ), Fintype.sum_prod_type]
   ```
   Extract `mean_mean_equiv`; keep the `show`s (the `_apply` trio is parked §3.4).
9. **F11 Binary32 concrete instance (small / low)** — Binary32Instance.lean:196–207: `hSabs`/`hSsq` identical 5-line bodies ending in non-terminal `simp [Fin.sum_univ_two, x0, oneHot, lbl]; norm_num`; 209–216: `hη`/`hη0` identical `simp only` lists.
10. **F14b PerChannelBN index round-trips (small / low)** — PerChannelBN.lean:167–177 (and 411–421):
    ```lean
    unfold reassocFwdIdx reassocBackIdx
    simp only [Equiv.symm_apply_apply]
    rw [Prod.mk.eta, Equiv.apply_symm_apply, Prod.mk.eta, Equiv.apply_symm_apply]
    ```
    4 copies; `Equiv` build or `Fin.cast` characterisation per audit.
11. **F6 `softmaxF_close` div chain (small / low)** — FloatBridge.lean:1559–1562: `simp only [smKappa]; rw [div_eq_iff hne, add_mul, one_mul, div_mul_cancel₀ _ hne]; ring` → `field_simp; ring`.
12. **F9 `step_bound`/`dense_step_bound` `htri` twin (small / low)** — FloatBridge.lean:155–161 vs 196–202.
13. **F7b `mlp_cot1_close`/`mlp_cot0_close` (small-medium / low-med)** — FloatBridge.lean:1135–1152 = 1202–1219 word-for-word.
14. **F18 MuonGeometry (small-medium / low)** — MuonGeometry.lean:219 `show (V * Uᵀ) * (U * Vᵀ) = V * (Uᵀ * U) * Vᵀ from by simp only [Matrix.mul_assoc]` (10 such); :196 `nlinarith [hMsq]`.
15. **F2 DPSBf16 twins (medium / low)** — two kinds only; a `batchWGrad` abstraction pays only when a third kind lands.

---

# Report: audit_arch_training.md

# Re-audit: audit_arch_training.md (2026-09-24, static read at 28fc373f)

Paths are relative to `LeanMlir/Proofs/`. The line numbers from the old audit are stale:
SgdDescentCnn is 6,284 lines now (was ~7.3k; the float/conv vocabulary moved to
`Architectures/ConvIndex` and `Float/ConvFloat`/`FloatClose` in 9a30d2ad). The softmax Jacobian
moved to `Architectures/Softmax.lean` (4530a0df).

| # | finding (old loc) | status | current location | effort / payoff |
|---|---|---|---|---|
| A1 | rung statements spell out the loss closure + step radius (Cnn:2899 et al.) | FIXED (statements) §1(q). **Residue STILL OPEN**: each rung proof still opens `simp only [stepRadius] at *; unfold <loss> at *`, re-`set`s the closure as `f` and restates hm2'/hmq'/hm3'/hm4' at `unflatten (flatten W₂)` | Training/SgdDescentCnn.lean:2382–2457 (conv2), :4376, :4994, :5469; float rungs :2576–2579; Mlp:438, :739; Linear:231 | medium / low-med |
| A2 | `Conv2Slot.loss_grad_lipschitz` 294 lines, 17 copies of δ, nested chains | FIXED (`set δ`, `gcongr`) §1(q); `mul_nonneg` chains PARKED §3.5 (positivity can't read hyps); `frozen_term_le` extraction STILL OPEN | Training/SgdDescentCnn.lean:1979 (274 lines; `by_cases hA` endgame at ~:2212) | small / low |
| A3 | nested `mul_le_mul_of_nonneg_left` chains (57), 0 `gcongr` | FIXED §1(q) for the deep nests; positivity side PARKED §3.5. Two short nests left | Cnn:2993 (double), Mlp:236–237 (triple) | trivial / low |
| A4 | triple `abs_sum_le_sum_abs` nest at 6 sites; `abs_triple_sum_sub_le` exists late | STILL OPEN — and the one existing lemma was deleted as dead in d3688f85 | Cnn:250–262, :2203–2206, :2813–2822, :2877–2886, :3234–3236, :4027–4030 | small / low-med |
| A5 | `simpa [hf] using …` (full simp set over huge terms) | STILL OPEN, 14 sites | Cnn:2455/2457, :4447/4449, :5028/5030, :5499/5501; Mlp:466/468, :778/780; Linear:251/253 | trivial–small / med |
| A6 | GradBudget non-negativity proved inline (`hη0`) | STILL OPEN (lemma placement); the `positivity` half is PARKED §3.5 | Cnn:2548, :4566 (12-line `hebacknn` restatement), :5817, :6276 | small / low |
| A7 | 16 `dense_close … .trans (denseErr_le_uniform …)` pairs | STILL OPEN (10 in Cnn, 5 in Mlp) | Cnn:199–227, :1328–1340, :4179, :6125; Mlp:901, :913, :1267, :1280, :1297 | small / low-med |
| A8 | duplicated Higham γ₇₈₅ block | STILL OPEN | Cnn:684–698 and :739–751 (`mnist_cnn_convW/convb_step_float_budget`, :655/:719) | trivial / low |
| A9 | file-scope `open … Classical` in 6 files | FIXED §1(l) + §1(r) (`MaxPool2IsArgmax.decidable`). Sub-point "drop `BigOperators`" not done (136 `open … BigOperators` repo-wide; no-op, not a fragility) | — | trivial / very low |
| A10 | `sum_swap_12_3`, `sum_swap_pair_pair` hand calc; `sum_window_cells` 46 lines | STILL OPEN | Cnn:2601, :2611 (`sum_swap_triple_triple` :2626 already on `Fintype.sum_prod_type`); `sum_window_cells` → Architectures/ConvIndex.lean:106 | trivial–small / low |
| A11 | Cifar `maxHeartbeats 1000000` | FIXED §1(b); statement on `cifar8LastConvLoss` + `stepRadius` §1(q) | Training/SgdDescentCifar.lean:83–184 | — |
| A12 | CNN/Depthwise `hasVJP3.correct` near-copies (~290 lines) | FIXED §1(q) (`padTap_indicator`, `sum_fin_ite_add_eq`) | Architectures/CNN.lean:176, :189, :287; Depthwise.lean:187 | — |
| A13 | `pdiv_bnIstdBroadcast` 174 lines, 9 CLM-coercion `show`s | STILL OPEN (unchanged) | Architectures/BatchNorm.lean:558 (shows at :576, :597–603, :624, :661) | medium / med |
| A14 | softmax/CE/oneHot defeq `show`s; `rw [fderiv_apply …]; rfl` ×4 | `_apply` lemmas FIXED §1(m) (Foundation/MLP.lean:275–281). STILL OPEN: `pdiv_eq_fderiv_coord` helper (4 sites) and one RHS `show` unfolding `softmax` | Architectures/Softmax.lean:65–66, :110, :199, :222; BatchNorm.lean:579 | small / low-med |
| A15 | `mhsaQkvW/b` if-chain + six `@[simp]` lemmas with redundant `show … from by decide` | STILL OPEN | Architectures/Attention.lean:1042–1110; consumers Nets/ViT/ViTBackB0.lean:249, :349–359 | trivial (drop the `decide` args) / small–med (`![Wq,Wk,Wv]`) ; low |
| A16 | `sdpa_back_{Q,K,V}_correct`: `unfold …; rfl` through `vjpMatComp` | STILL OPEN; the "large share of 15 s" was a static guess, not measured | Architectures/Attention.lean:535–547, :619–630, :652–667; `vjpMatComp` Foundation/Tensor.lean:638 (no `_backward` lemma) | medium / low (unmeasured) |
| A17 | `Proofs.Real.hasDerivAt_tanh` name collision | FIXED §1(l) | Architectures/LayerNorm.lean:152, :162 | — |
| A18 | `head_diff_ct`: `rfl` across `globalAvgPoolFlat`/`bcell`, restated summand | STILL OPEN | Training/BatchSealKit.lean:820–860 | small / low |
| A19 | `ctConv_inj` `Nat.mul_comm` rw chain | STILL OPEN | Training/BatchSealKit.lean:790–793 | trivial / low |
| A20 | `depthwiseStride2FlatXlaBack_eq_vjp_backward` undocumented `show` + `rfl` | STILL OPEN | Architectures/DepthwiseBackCertifiedTie.lean:68–71 | trivial / low |

Counts: FIXED 6 (A3, A9, A11, A12, A17 + A1's statements), FIXED-with-residue 3 (A1, A2, A14),
STILL OPEN 11 (A4–A8, A10, A13, A15, A16, A18–A20), PARKED only as sub-parts (A2/A3/A6
positivity, §3.5). GONE: none (the `abs_triple_sum_sub_le` lemma that A4 pointed to is gone, but
the nests it targeted are not).

## Still open, ranked (low-hanging first)

### 1. A5 — `simpa [hf]` → `simpa only [hf]` / `exact` (14 sites, trivial–small, med payoff)

Training/SgdDescentCnn.lean:2450–2457 (same shape at :4447/:4449, :5028/:5030, :5499/:5501;
Mlp:466/468, :778/780; Linear:251/253):
```lean
    (fun t ht idx => by
      have h := cnn_conv2_loss_grad_lipschitz b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
        label hh hw ha hx hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (Kernel4.flatten W₂)
        (-(lr • gh)) hD hm2' hmq' hm3' hm4' hsmall t ht idx
      simpa [hf] using h)
    h1 h2
  simpa [hf] using hmain
```
After `set f … with hf` the goal is already stated in `f`, and `sgd_descends`'s conclusion
(`f (x - lr • gh) ≤ f x - lr * (∑ i, gradAt f x i ^ 2) / 2`, SgdDescent.lean:208) matches it, so
the outer one is likely `exact hmain`. The inner one only needs β/`hf`. The unrestricted simp set
is the root-file `@[simp]` hazard the audit named; compile each site before landing.

### 2. A8 — `FloatModel.gamma785_le` (trivial, low)

Training/SgdDescentCnn.lean:684–698, duplicated verbatim at :739–751:
```lean
  have hk1 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 / (1 - ((28 * 28 + 1 : ℕ) : ℝ) * u32)
      ≤ 47/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (28 * 28 + 1) - 1 ≤ 47/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (28 * 28 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  …
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
```

### 3. A19 — `ctConv_inj`'s `hcomm` (trivial, low)

Training/BatchSealKit.lean:790–793:
```lean
  have hcomm : (2 * (2 * w)) * (2 * r.val) + 2 * s.val
      = (2 * (2 * w)) * (2 * r'.val) + 2 * s'.val := by
    rw [Nat.mul_comm (2 * (2 * w)) (2 * r.val), Nat.mul_comm (2 * (2 * w)) (2 * r'.val)]
    exact hnat
```
→ `by linarith [hnat]` (or `by ring_nf; ring_nf at hnat; exact hnat`).

### 4. A15 (trivial half) — drop the redundant `show … from by decide` (low)

Architectures/Attention.lean:1076 and :1084–1086 (and the `_b_eq1/_eq2` twins):
```lean
  simp [Equiv.symm_apply_apply,
        show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
        show (2 : Fin 3) ≠ (1 : Fin 3) from by decide]
```
Core's `Fin` simprocs decide literal `Fin` (in)equalities; `simp [Equiv.symm_apply_apply]` should
close all six. The `![Wq, Wk, Wv]` restatement is the bigger (small–medium) half; it changes the
defs ViTBackB0 consumes (:249, :349–359).

### 5. A20 — DepthwiseBackCertifiedTie `show` (trivial, low)

Architectures/DepthwiseBackCertifiedTie.lean:68–71:
```lean
  funext dy
  show depthwiseFlatBack (h := 2 * h) (w := 2 * w) W (decimateOddBack c h w dy) = _
  rw [depthwiseFlatBack_eq_vjp_backward hkH hkW W b x]
  rfl
```
→ `rw [depthwiseStride2FlatXlaBack, Function.comp_apply]` (or the def's equation lemma) plus a
one-line comment on the closing `rfl`.

### 6. A4 — `abs_triple_sum_le` (small, low-med)

Six sites of the same nest; the lemma the audit wanted moved up was deleted as dead in d3688f85.
Training/SgdDescentCnn.lean:250–261:
```lean
  calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi|
      ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW, …| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW, …| :=
        Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, |…| :=
        Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.abs_sum_le_sum_abs _ _
```
Also :2203–2206, :2813–2822, :2877–2886, :3234–3236, :4027–4030. Add
`abs_triple_sum_le (f : α → β → γ → ℝ) : |∑ a, ∑ b, ∑ c, f a b c| ≤ ∑ a, ∑ b, ∑ c, |f a b c|`
in ConvIndex (a leaf) and collapse each calc's first three steps.

### 7. A10 — `sum_swap_12_3` / `sum_swap_pair_pair` (trivial, low)

Training/SgdDescentCnn.lean:2601–2624 use `Finset.sum_congr rfl fun _ _ => Finset.sum_comm`
chains; `sum_swap_triple_triple` right below (:2626–2632) already has the 3-line
`Fintype.sum_prod_type` / `Finset.sum_comm` proof. `sum_window_cells` (ConvIndex.lean:106, ~45
lines) → `Fintype.sum_equiv` along `winRowEquiv`/`winColEquiv` — small.

### 8. A14 residue — `pdiv_eq_fderiv_coord` (small, low-med)

Architectures/Softmax.lean:63–66:
```lean
  have h_swap : fderiv ℝ (softmax (c' + 1)) z (basisVec i) j =
                fderiv ℝ (fun z' : Vec (c' + 1) => softmax (c' + 1) z' j) z (basisVec i) := by
    rw [fderiv_apply (softmax_differentiable (c' + 1) z) j]
    rfl
```
Same step at Softmax.lean:199, :222 and BatchNorm.lean:579. Also Softmax.lean:110 still
`show`s the RHS with `softmax` unfolded (`(Real.exp (z j) / S) * …`) — `rw [softmax_apply]` exists.

### 9. A7 — `FloatModel.dense_close_layer` (small, low-med)

Training/SgdDescentCnn.lean:1328–1340:
```lean
  have hE3close : ∀ l, |Z3F l - Z3 l| ≤ E3 := fun l =>
    (M.dense_close W₃ b₃ PF PR E2 E2nn hPool l).trans
      (M.denseErr_le_uniform hw₃ E2nn hW₃ hb₃ hMpool l)
  have hRelu3 : ∀ l, |relu d₃ Z3F l - relu d₃ Z3 l| ≤ E3 :=
    fun l => relu_close _ _ _ hE3close l
  have hE4close : ∀ q, |Z4F q - Z4 q| ≤ E4 := fun q =>
    (M.dense_close W₄ b₄ (relu d₃ Z3F) (relu d₃ Z3) E3 E3nn hRelu3 q).trans
      (M.denseErr_le_uniform hw₄ E3nn hW₄ hb₄ hM3 q)
```
15 pairs (Cnn 10, Mlp 5). One lemma in `Float/FloatClose.lean` (leaf).

### 10. A6 — GradBudget `_nonneg` lemmas beside the defs (small, low)

Training/SgdDescentCnn.lean:4566–4591 carries a 12-line restatement of a sub-budget:
```lean
    have hebacknn : (0:ℝ) ≤ ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ *
            (FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ + …
```
Four sites (:2548, :4566, :5817, :6276); only `cnnConv2CotBudget_nonneg` (:1167) is a lemma.
Moving them does not remove the `mul_nonneg` terms (§3.5: `positivity` can't use the weight
hypotheses), only relocates them next to the def.

### 11. A18 — `head_diff_ct` (small, low)

Training/BatchSealKit.lean:833–838 (`rw [row_batchMap, row_batchMap]; rfl` across
`globalAvgPoolFlat`/`bcell`) and :850–854 (restated summand via `show … from by rw [hgap ci]; ring`)
then `Finset.sum_eq_single_of_mem`. → `globalAvgPoolFlat_bcell` lemma + `Fintype.sum_eq_single`.

### 12. A13 — `pdiv_bnIstdBroadcast` (medium, med)

Architectures/BatchNorm.lean:558 (~175 lines), unchanged:
```lean
    show ((ContinuousLinearMap.proj k : Vec (n' + 1) →L[ℝ] ℝ) - mean_clm) y = _
    …
    show y k - mean_clm y = _
    show y k - (((n' + 1 : Nat) : ℝ)⁻¹ • …
    show y k - ((n' + 1 : Nat) : ℝ)⁻¹ * ∑ i' : Fin (n' + 1), y i' = _
  …
        rw [Finset.sum_eq_single i]
        …
        · intro h; exact absurd (Finset.mem_univ i) h]
  …
  rw [show bnIstd (n' + 1) x ε = 1 / Real.sqrt (bnVar (n' + 1) x + ε) from rfl]
```
The CLM-coercion `show`s are the most Mathlib-bump-fragile thing left in scope. Split
`bnVar_hasFDerivAt` + `sum_sub_bnMean`, `simp [basisVec_apply, Finset.sum_ite_eq']` for the
Kronecker collapses.

### 13. A1 residue — rungs stop unfolding the names they state (medium, low-med)

Training/SgdDescentCnn.lean:2382–2413: `simp only [stepRadius] at *; unfold cnnConv2KernelLoss at *`,
then `set f := fun v' => crossEntropy …` (the closure re-typed), then four
`have hmX' … := by rw [Kernel4.unflatten_flatten]; exact hmX …` restatements (~35 lines per rung).
Stating `sgd_descends` / the `margin*_keeps_offkink` lemmas on `stepRadius` and the named loss
would delete them; touches the non-rung lemmas §3.5 chose to keep on closures, so it is a
statement-level change to shared Training lemmas, not a local edit.

### 14. A16 — `vjpMatComp_backward` for the SDPA ties (medium, unmeasured)

Architectures/Attention.lean:543–547:
```lean
  rw [← (sdpaQChainHasVJP n d K V).correct Q dOut i j]
  -- Goal: sdpaBackQ ... = (sdpaQChainHasVJP ...).backward Q dOut i j
  unfold sdpaBackQ sdpaDScores sdpaDScaled sdpaDWeights sdpaWeights
    sdpaQChainHasVJP
  rfl
```
Robustness only; profile Attention before claiming a speed win (the audit's 15 s attribution was
a guess).

### Not worth it
- A9's `open … BigOperators` (136 sites repo-wide): a no-op open, not a fragility.
- A2's `frozen_term_le` extraction: the 274-line proof already carries section comments; low value.
- A3's two remaining short nests (Cnn:2993, Mlp:236–237): `gcongr` would do it, but they're one-offs.

---

# Report: audit_codegen_certs.md

# Re-audit: audit_codegen_certs.md (2026-09-24, HEAD 28fc373f, static read + grep only)

## Table

| # | finding (audit §) | status | current location / evidence | effort / payoff |
|---|---|---|---|---|
| A.1-1/2 | 215-arm `den` + `simp only [den]` cost | FIXED | proof_cleanup §1(g) (`denStep`/`denStepApp`, 343 s → 83 s) | — |
| A.1-3 / A.3-5 | unused `deriving DecidableEq` on `Raw`/`Tok` | STILL OPEN | `Codegen/StableHLOPretty.lean:281` (Raw), `:889` (Tok); grep finds no `==`/`decide`/`DecidableEq` user of either in LeanMlir/tests/apps. §3.2 notes it "not measured" (not parked) | trivial / low (build time of the printer, unmeasured) |
| A.1-4 / A.3-6 | `emitTok` one giant def → per-family defs | STILL OPEN (partly shrunk) | `StableHLOPretty.lean:1368`, still ~3,130 lines; 91737514 cut −411 (bf16/fp8 twins → `emitContract`) but no per-family split | medium / low (§3.2: emitTok 14 s + 8.5 s compile) |
| A.1-5 / A.2 row 1 (writers half) | `#eval` artifact writers (+ 3 stray `/tmp/linear_*_v.mlir`) in a non-leaf root file | STILL OPEN | `StableHLOPretty.lean:4858–4862` (`/tmp/linear_{fwd,back,train_step}_v.mlir`), `:4868–` (9 `verified_mlir/*` writers); StableHLOPretty has 98 transitive importers (44 in LeanMlir). §3.2 called this "the cheap half"; `CnnArtifacts.lean`/`MlpArtifacts.lean` are the template | small / low-med |
| A.2 row 1 (graphs half) | chapter graphs + `cnnBackGraph_faithful` + CifarCNN import out of root | FIXED | fa689fb3: `Nets/Small/ChapterGraphTies.lean:109`; StableHLO imports no net (import list now Foundation/Architectures/Training.Optim only) | — |
| A.2 row 2 | `StableHLOPretty.lean` printer split | FIXED | 9da1a817 (audit_v2 §5) | — |
| A.2 row 3 | proof modules importing renderers (CnnRender/LinearTrainStep/SyncBnSites) | FIXED (mostly) | 64eb7943 + CnnArtifacts; residual: `Foundation/DataParallelNode` (uses `skel`, 16 downstream) and `Nets/Small/LinearFold` still import StableHLOPretty | — (residual: medium / low) |
| A.2 deep | `BatchableOp`-style descriptors for `SHlo` unary/binary ops | STILL OPEN | `SHlo` at `StableHLO.lean:327`, still ~215 ctors; audit_v2 §5 declined only a *regroup*, not this | large / med (not low-hanging) |
| A.3-1 | file-wide `maxHeartbeats 4000000` | FIXED | §1(g) | — |
| A.3-2 | `cnnBackGraph_faithful` 2M bump | FIXED | §1(g); now in `ChapterGraphTies.lean` | — |
| A.3-3 | `simp only [… den …]` ×113 → node lemmas | FIXED | §1(g) (dsimprocs rather than node lemmas; same effect) | — |
| A.3-4 | `bnBack_faithful` / `bnPerChannelBack_faithful` undocumented `show` | STILL OPEN | `StableHLO.lean:3188`, `:3639` | trivial / low |
| B-1 | render modules `maxRecDepth 4000000` (unexplained, copied) | STILL OPEN | 21 × 4M + 2 × 1M: MobileNetV2RenderB 567/624/654/1158/1226; ResNet50RenderB 496/552/1140/1183/1204; MobileNetV4RenderB 599/936; EfficientNetRender 728/865/880/900/1059/1160; ResNet34RenderB 183,237 (1M), 1309/1344/1364 | small (strip-and-compile, artifacts byte-identical) / low (a limit, not a cost) |
| B-2 | CnnRender `#eval` writers → leaf | FIXED | `Codegen/CnnArtifacts.lean` (44 writers; nothing imports it) | — |
| B-2b | same split for R50/R34/ENet/ViT/ConvNeXt/IRPrint renders | STILL OPEN (audit itself rated low) | #eval counts: ViTRenderB 46, R50 45, ConvNeXtRenderB 42, ENet 38, R34 34, MNv2 17, MNv4 11, IRPrint 51; importers are tests + `FwdGraphTextTies` only | small each / low |
| B-3 | `parseStack_toToks` 19 s | no action (audit: nothing to do) | `StableHLOParse.lean:199` | — |
| B-4 | `rmsBufNext_eps_placement_at_zero` `show`s | STILL OPEN (moved) | `Training/Optim/RmsPropStep.lean:147,149` | trivial / low |
| B-5 | `LambTriple` `show`s (+ inline `show … from rfl`) | STILL OPEN | `Codegen/LambTriple.lean:101`, `:127–128` | trivial / low |
| B-6 | `keepProb_last` / `clipFactor_accum` positional `rw` chains | STILL OPEN (moved) | `Training/DropPath.lean:184`, `Training/Optim/GradClip.lean:206` | trivial / low |
| B-7 | EfficientNetRenderPC vs PCEval duplicated, parametrise by `BnMode` | REJECTED (by later design) | audit_v2 §5 hazard row (l.413) + 28fc373f: eval chains *dropped* the BN-mode parameter (`bnEvalSite`), the `.train` arm was the per-example-BN defect; PCEval's name drift is documented and left (audit_v2 §5 "Left as they were") | — |
| C.1 (IBP conv / Instance / pair-SDP data) | per-entry ℝ `simp` → ℚ + one `decide +kernel` | FIXED | §1(h) (IBP conv), §1(i) (Instance, Scorecard, Float), §1(j) (FullNets) | — |
| C.1-4 | drop emitted `try norm_num` | STILL OPEN | still emitted by 6 generators: `lipschitz_cert_scorecard_ibp.py:282`, `lipschitz_cert_float.py:124`, `lipschitz_cert_pair_sdp.py:309,315`, `lipschitz_cert_pair_sdp_full.py:247,253`, `trained_cnn_witness.py:380,404`, `trained_cnn_seal.py:128,188`; 818 `try norm_num` in committed Certificates files | medium (regen + heavy recompile, SDPFull ~5 min/13.5 GB, build by name) / low (robustness only) |
| C.2 | FullNets matrix-level Gram lemma | PARKED | proof_cleanup §3.3 + §4 (~20 s of 61 s at best; `List.getD` 784-wide kernel timeout) | — |
| C.3 (bumps) | pair-SDP 64M heartbeats | FIXED | §1(k) (0 `set_option` left in the four SDP files) | — |
| C.3 (shape) | LDLᵀ-coefficient `linarith [sq_nonneg …]` → ℚ matrix identity / diag-dominance kernel check | STILL OPEN | `LipschitzCertScorecardSDPFull.lean:64–67` (90+-digit coefficients); generators `lipschitz_cert_pair_sdp{,_full}.py`. SDPFull* are in no lib (disabled tier) | large / low (SDP ~70 s; SDPFull not built) |
| C.4 / D.2 | `binomTailNum` quadratic | FIXED | §1(a) | — |
| D.1 | general dense layer out of LipschitzCertInstance | FIXED | 528fd25f: `Certificates/DenseEuclid.lean` (`denseE_lipschitzL2*`, `certified_at_eps`, `sum_sq_matTvec_eq`, …); `LipschitzCertPairSDP` imports DenseEuclid; Instance has 3 importers | — |
| D.3 | `pi_gaussian_np_shift` integrability repetition | FIXED | 26280a26 (audit_v2 §6): `SmoothingGaussian.lean:255` `mul_bdd`, `:258–269` `Integrable.of_mem_Icc` | — |
| D.4a | `mlp_gap_eq` undocumented `show` through `denseE`/`WithLp` | STILL OPEN | `Certificates/LipschitzCertPairSDP.lean:166–167` | trivial / low-med (WithLp defeq is the bump-fragile kind) |
| D.4b | `mlpT_logit_continuous` undocumented `show` | STILL OPEN (moved) | `Certificates/LipschitzCertInstance.lean:398–399` | trivial / low-med |
| D.5 | extract `two_mul_inner_le_sq_add` from `pair_sq_bound` | STILL OPEN (optional; audit called the proof acceptable) | `LipschitzCertPairSDP.lean:115` (`have hsq`) | small / low |
| D.6 | `LipschitzL2` vs Mathlib `LipschitzWith` | OUT OF SCOPE / kept by design | rubric: Mathlib reuse is another audit; docstring `LipschitzCert.lean:38–41` now states the choice (`ℝ≥0∞`-valued vs explicit ε–δ) | — |
| clean | `SmoothingCP` `simp only [Nat.choose_self, …]` → `simp` | STILL OPEN (optional) | `SmoothingCP.lean:203` | trivial / very low |

Counts: FIXED 15 · PARKED 1 · REJECTED/out-of-scope 2 · no-action 1 · STILL OPEN 14 (of which 4 are optional/very-low or large).

## Still open, ranked (trivial/small first)

1. **Drop `DecidableEq` from `Raw`/`Tok`** — trivial; unmeasured build-time win on the printer that 98 modules wait for. No user found (grep of `==`/`decide`/`DecidableEq` over Raw/Tok in LeanMlir/tests/apps).
   ```
   LeanMlir/Proofs/Codegen/StableHLOPretty.lean:281   deriving DecidableEq, Repr, Inhabited   -- Raw
   LeanMlir/Proofs/Codegen/StableHLOPretty.lean:889   deriving DecidableEq, Repr              -- Tok
   ```
   (`OptKind` at :4593 is tiny; leave it.) Measure with the profiler before/after per §0.

2. **Move StableHLOPretty's `#eval` writers to a leaf `StableHLOArtifacts.lean`** (CnnArtifacts is the template; update `regen_verified_mlir.sh` module list + proofs.yml render guard per memory notes). Also deletes three stray `/tmp` writes that run on every rebuild of a 98-importer module.
   ```
   StableHLOPretty.lean:4858  #eval IO.FS.writeFile "/tmp/linear_fwd_v.mlir"
   StableHLOPretty.lean:4860  #eval IO.FS.writeFile "/tmp/linear_back_v.mlir"
   StableHLOPretty.lean:4862  #eval IO.FS.writeFile "/tmp/linear_train_step_v.mlir"
   StableHLOPretty.lean:4868  #eval (do … IO.FS.writeFile "verified_mlir/linear_fwd.mlir" … cifar8w_bn_fwd.mlir)
   ```
   small / low-med.

3. **`show` through `denseE`/`WithLp` defeq (D.4)** — the Mathlib-bump-fragile kind; `denseE_apply`/`reluE_apply` exist.
   ```
   LipschitzCertPairSDP.lean:166  show (∑ t, W2 i t * (reluE (denseE W1 x)) t)
                                      - (∑ t, W2 j t * (reluE (denseE W1 x)) t) = _
   LipschitzCertInstance.lean:398 show Continuous fun x : EuclideanSpace ℝ (Fin 49) =>
                                      ∑ k : Fin 8, W2t j k * max (∑ l, W1t k l * x l) 0
   ```
   → `simp only [Function.comp_apply, denseE_apply, reluE_apply]` / `unfold mlpT; fun_prop`-style. trivial / low-med.

4. **BN backward `show`s → `den_*` rfl lemmas (A.3-4)**
   ```
   StableHLO.lean:3188  show bnGradInput n ε γ x (den e) i = _
   StableHLO.lean:3639  show bnPerChannelTensor3GradInput oc h w ε γ x (den e) i = _
   ```
   → add `@[simp] theorem bnBack_den … := rfl` beside the other 69 `den_*`/`*_faithful` rfl lemmas and `rw`. ⚠ StableHLO is a root file (205 transitive dependents): batch with other root edits (§0). trivial / low.

5. **Optimizer-file `show`s / positional rw chains (B-4, B-5, B-6)**
   ```
   Training/Optim/RmsPropStep.lean:147  · show μ * buf i + g i / Real.sqrt (rmsSqNext ρ sq g i + ε) = _
   Training/Optim/RmsPropStep.lean:149  · show μ * buf i + g i / (Real.sqrt (rmsSqNext ρ sq g i) + ε) = _
   Codegen/LambTriple.lean:101          show scalarOf (fun _ => (0 : ℝ) + gradSumSq θ) = gradSumSq θ
   Codegen/LambTriple.lean:127–128      show (sgdParam lr θ (lambScale (scalarOf (fun _ => (0 : ℝ))) _), _, _) = _
                                        rw [show scalarOf (fun _ => (0 : ℝ)) = 0 from rfl, lambScale_zero_weight]
   Training/DropPath.lean:184           rw [keepProb, Nat.cast_sub (by omega), Nat.cast_one, mul_div_assoc, div_self hd, mul_one]
   Training/Optim/GradClip.lean:206     rw [clipFactor, clipFactor, hsq, ← mul_add, mul_div_mul_left _ _ (ne_of_gt hk)]
   ```
   trivial each / low. (RmsPropStep and GradClip are imported by StableHLO → root-cost rebuild; batch.)

6. **Render `maxRecDepth 4000000` ×21 (+1M ×2)** — strip-and-compile/bisect (ViT/ConvNeXt/Cnn renders manage with 4000–8000); gate = artifacts byte-identical. small / low (limit, not cost; §0's "bump comment ≠ still needed" rule found 57/59 dead elsewhere).
   ```
   ResNet50RenderB.lean:496   set_option maxRecDepth 4000000 in
   ResNet34RenderB.lean:183   set_option maxRecDepth 1000000 in
   ```

7. **Remaining render `#eval` → leaf splits (B-2b)** — small each / low (importers are tests + FwdGraphTextTies).

8. **`try norm_num` in 6 generators (C.1-4)** — medium (regenerate + recompile heavy tiers; SDPFull only by name) / low.

9. **`two_mul_inner_le_sq_add` extraction (D.5)**, **SmoothingCP:203 `simp`** — optional, low.

Not low-hanging: `emitTok` per-family split (medium/low), `SHlo` descriptor refactor (large), SDP LDL → ℚ kernel identity (large, SDPFull not in any lib), DataParallelNode/LinearFold still importing the printer (medium/low).

---

# Report: audit_effnet_vit.md

# audit_effnet_vit.md — re-audit at HEAD 28fc373f (2026-09-24)

Static read + grep only. `grep -rn "set_option max" LeanMlir/Proofs/Nets/{ViT,EfficientNet}` is EMPTY:
every heartbeat / recDepth bump the report cites is gone (§1(b), c2e5467a). So every "heartbeats"
smell below is FIXED; what is left is the non-budget part of each finding.

| # | finding | status | current location | effort / payoff |
|---|---|---|---|---|
| V1 | `vit_net_tiedGB` / `vit_net_tied_certified`: 16M/400k bumps; 192 block binders, 28 lets at numerals; `BlockParamsV` unused | bumps FIXED (§1(b)); statement shape OPEN but not low-hanging | `Nets/ViT/ViTStepTieGB.lean:390` (192 binders still, `-- block 1…12`), `ViTStepTie.lean:292`; `vit_net_tiedGB` is a blueprint node (content.tex:11636), `vit_net_tied_certified` a yaml row | large / low (no budget left to save; statement + blueprint + yaml change) |
| V2 | 12 ViTStepTieGB defs (`blkSaves`, `cAtt`, `cQ`…`cM1`, `vitBlockTiedGB`, ties) restate a 16-param block list; should take `BlockParamsV` | STILL OPEN | `ViTStepTieGB.lean:83–265`; same in `ViTStepTie.lean:43–184` | medium / med (prerequisite for V1) |
| V3 | `mhsaClean_backward_collapseMH`: 4M bump; two undocumented 13-/6-line `show`s; 20-line if-form written 3×; `hproj0/1/2` one lemma at c=0,1,2 | bump FIXED; rest STILL OPEN | `ViTBackB0.lean:276–420` (shows at 283/297, if-form at 366/389, hproj at 345–359) | medium / med |
| V4 | `mhsaBackGraphMH_faithful`: 2M bump; `rw [show ∑… = ∑… from by sum_congr…]` restates an 18-line sum; `hQbr/hKbr/hVbr` triplicated | bump FIXED; rest STILL OPEN | `ViTBackB0.lean:570–684` (rw-show at 666–678) | small / low-med |
| V5 | `qkv_back_fanin_MH`: 1.6M bump; `show (1:Fin 3) ≠ 0 from by decide` ×3 + `ite_true`/`ite_false` | bump FIXED; simp list STILL OPEN | `ViTBackB0.lean:247–252` | trivial / low (bump-churn insurance) |
| V6a | `sum_heads_3d` hand-rolls the `finProdFinEquiv` reindex | STILL OPEN | `ViTBackB0.lean:215–225` | trivial / low |
| V6b | `mulVec_headPadMat` hand-rolled reindex | FIXED (uses `sum_finProdFinEquiv`, audit v2 §6 72dfe870) | `ViTBackB0.lean:495` | — |
| V6c | `mhsaBackFlat_eq_mhsa_vjp`: 3× `← Equiv.sum_comp` with explicit motives + `Fintype.sum_prod_type` ×3 | STILL OPEN | `ViTMhsaBackCertifiedTie.lean:92–103` | small / low-med (−12 lines; rw order needs one try) |
| V7 | `attnSubFlat_tie_v` / `mlpSubFlat_tie_v`: `show w (finProdFinEquiv …)` + `rw [Prod.mk.eta, Equiv.apply_symm_apply]` ×2 | STILL OPEN | `ViTVecLNBackCertifiedTie.lean:97–99, 127–129` | trivial / low-med (undocumented defeq; `Mat.unflatten_apply` now exists, Tensor.lean:529) |
| V8 | `*TiedB` Props restate their `_den` lemma verbatim (4 in ViTFoldGB, 4 in ViTFold) | STILL OPEN | `ViTFoldGB.lean:285–330` vs `:79–150`; `ViTFold.lean:145–181` vs `:31–140`; 22 `intro …; exact …_den` lines in ViTStepTieGB, 13 in ViTStepTie | small / med |
| V9 | `vit_cls_den` under `linter.unusedSimpArgs false`; `clsGrad_denB`'s unrestricted `simp` at 192/197; numeral shapes | STILL OPEN (linter + `simp`); variable-`D` part low payoff | `ViTStepTie.lean:195`; `ViTFoldGB.lean:241` | trivial (linter/simp) / low; variable D = medium/low |
| E1 | `efficientnetInputGradBFull_eq_efficientnetB_full_vjp` + `…ForwardB_full_vjp`: 4M/800k bumps; closing `rfl` at numerals; `(by decide)` for `0 < 112` | bumps FIXED (§1(b)); `(by decide)` ×2 remains | `EfficientNetFullWholeBackCertifiedTie.lean:168` | trivial / negligible (don't bother) |
| E2 | `efficientnetForwardBFullHasVJP` re-derives the generic apex with 36 `have`s + 16 `vjpComp _ _`; bridged by `backward_unique` | recDepth bump FIXED; duplication STILL OPEN | `EfficientNetFullB0.lean:371–440` (def), bridge `EfficientNetFullWholeBackCertifiedTie.lean:210–219` | small–medium / med (−~60 lines, bridge → rfl; move `efficientnetBFullHasVJP` into FullB0) |
| E3 | ENet capstones (`efficientnet_net_syncTiedG`, `_tiedG`, `_tied`): 4M/100k bumps; 37–55 lets at literal widths | bumps FIXED (§1(b)); statement shape OPEN but not low-hanging | `EfficientNetSyncStepTieG.lean` (comparator-tier decl), `EfficientNetStepTieG.lean`, `EfficientNetStepTie.lean` | large / low (statement change; syncTiedG is in tests/comparator tier) |
| E4 | 49 `0 < ε` binders × 7 statements | FIXED §1(o) (`B0Weights.EpsPos`) | — | — |
| E5 | `cbsB_back_eq` / `dwbsB_back_eq` / `dwbsSB_back_eq` / `projB_back_eq` / `hdCotIn_eq_vjp`: same 6-line proof with an undocumented `show` ×5 | STILL OPEN | `EfficientNetSyncStepTieG.lean:346–397, 484–495` | small / med |
| E6a | 25 `*_smul` chain lemmas | FIXED §1(p) (`IsHomog`) | — | — |
| E6b | 30 `*_shard` chain lemmas / `IsShardwise` | PARKED §3.6 ("`IsShardwise` … was not built") | — | — |
| E6c | SyncStepTieG without `variable` sections | FIXED §1(p) (sections added; measured +21 lines / −6% bytes) | — | — |
| E7 | `hN hh hw` threaded for `nhw_ne_zero` → `[NeZero _]` | PARKED §3.6 ("`[NeZero N]` … was not done") | — | — |
| E8 | MBConv tail chain written 7× (StepTie ×3, StepTieG ×3, Sync `EnTail`); `EnTail` → leaf, `enetTailTiedG` once | STILL OPEN | `EfficientNetStepTieG.lean:58–341`, `EfficientNetStepTie.lean:66–360`, `EnTail` at `EfficientNetSyncStepTieG.lean:82` | medium–large / med |
| E9 | BN-β clause inline (no `BnBetaTiedB` Prop) | STILL OPEN | 10 × `den (SHlo.bnBetaGradB` in `EfficientNetStepTieG.lean` (e.g. :93–97); lemma `bnBetaGradB_den` now in `Foundation/GradNodesB.lean:140` | small / low-med |
| E10 | 5 × `simp only [den, batchMap, batchMapHasVJP, …]; rfl` with drifting unfold lists | STILL OPEN (moved; `den` → `denStepApp`) | `Foundation/BatchedBackLinks.lean:84–180`; `batchMapHasVJP` at `Foundation/BatchMapVJPAt.lean:139` | small / low-med |

Counts: FIXED 4 whole (V6b, E4, E6a, E6c) + heartbeat halves of V1/V3/V4/V5/E1/E2/E3 all fixed; PARKED 2 (E6b, E7);
STILL OPEN 14 rows (V1-shape, V2, V3, V4, V5, V6a, V6c, V7, V8, V9, E2, E5, E8, E9, E10 — V1/E3 shape = large, not
recommended); E1 residue negligible.

## Still open, ranked (low-hanging first)

1. **V7 — `show` over `Mat.unflatten`'s body, twice** (trivial). `ViTVecLNBackCertifiedTie.lean:97–99`:
   ```lean
   have hw : Mat.unflatten w (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2 = w idx := by
     show w (finProdFinEquiv ((finProdFinEquiv.symm idx).1, (finProdFinEquiv.symm idx).2)) = w idx
     rw [Prod.mk.eta, Equiv.apply_symm_apply]
   ```
   and `:127–129` (`hv`). Fix: `rw [Mat.unflatten_apply, Prod.mk.eta, Equiv.apply_symm_apply]` (the `_apply`
   lemma landed in §1(m), Tensor.lean:529) — removes the undocumented defeq.

2. **V5 + V6a — `qkv_back_fanin_MH` / `sum_heads_3d`** (trivial). `ViTBackB0.lean:219–225, 247–252`:
   ```lean
   rw [← Equiv.sum_comp (finProdFinEquiv : Fin heads × Fin (3*d) ≃ Fin (heads * (3*d))) f]
   rw [Fintype.sum_prod_type]
   apply Finset.sum_congr rfl; intro h _
   rw [← Equiv.sum_comp (finProdFinEquiv : Fin 3 × Fin d ≃ Fin (3*d)) (fun kk => f (finProdFinEquiv (h, kk)))]
   rw [Fintype.sum_prod_type]
   …
     show (1 : Fin 3) ≠ (0 : Fin 3) from by decide,
     show (2 : Fin 3) ≠ (0 : Fin 3) from by decide,
     show (2 : Fin 3) ≠ (1 : Fin 3) from by decide, ite_true, ite_false]
   ```
   Fix: `sum_heads_3d := by rw [sum_finProdFinEquiv]; exact Finset.sum_congr rfl fun h _ => sum_finProdFinEquiv _`;
   simp list → `Fin.reduceEq, reduceIte` (or `↓reduceIte`). Bump already gone, so this is rename insurance only.

3. **V9 — `vit_cls_den` linter switch + `clsGrad_denB` bare `simp`** (trivial, one compile to prune).
   `ViTStepTie.lean:195`: `set_option linter.unusedSimpArgs false in` over
   `simp only [denStep, denStepApp, batchSlice, clsSliceFlat, clsTokenGrad]; rw [Fin.sum_univ_one]; rfl`;
   `ViTFoldGB.lean:241`: `simp [batchSlice, batchMap, clsSliceFlat, Equiv.symm_apply_apply]`. Drop the switch and
   prune; `simp?` → `simp only`. Needs one `lake env lean` to see which args fire.

4. **E5 — five graph-back `show`s** (small, med). `EfficientNetSyncStepTieG.lean:350–355` (×5, through :495):
   ```lean
   have hg := cbsBackBatchedGraph_faithful W b ε hε γ β x (.operand "" dy)
   rw [den_operand] at hg
   rw [← hg]
   show cInB N W b (den (SHlo.bnBatchLABack _ _ _ ε γ _ _)) = _
   rw [den_bnBatchLABack_eq_bnBackB _ _ _ ε hε γ β]
   rfl
   ```
   Fix: one `rfl` structural lemma per graph (`den_cbsBackBatchedGraph : den (cbsBackBatchedGraph …) = cInB N W b
   (den (SHlo.bnBatchLABack …))`) next to the graph defs, then each proof is one `rw` chain. Minimum: a comment.

5. **V4 — restated 18-line sum in `mhsaBackGraphMH_faithful`** (small). `ViTBackB0.lean:666–678`:
   ```lean
   rw [show (∑ h : Fin (hm1 + 1), ((den (SHlo.denseRowBack "%Wq" Wq …) j + …) + …))
       = ∑ h : Fin (hm1 + 1), ((Mat.flatten (fun r c => …sdpaBackQ…) j + …) + …)
     from by
       apply Finset.sum_congr rfl; intro h _; rw [hQbr h, hKbr h, hVbr h]]
   ```
   Fix: `refine (Finset.sum_congr rfl fun h _ => ?_).trans ?_` + `rw [hQbr h, hKbr h, hVbr h]`; no `simp_rw`
   (it would name `den`, §1(g) trap).

6. **V6c — three motive-carrying `Equiv.sum_comp`s** (small). `ViTMhsaBackCertifiedTie.lean:92–103`:
   `rw [← Equiv.sum_comp (finProdFinEquiv : Fin h × Fin dh ≃ Fin (h * dh)) (fun k => Wq c k * mhsaSdpaBackQ …), …]`
   then `rw [Fintype.sum_prod_type, ×3]`. Fix: `simp only [sum_finProdFinEquiv (m := h) (n := dh)]` or three
   `rw [sum_finProdFinEquiv]` (rewrite order to check).

7. **E10 — five `simp only [denStepApp, batchMap, batchMapHasVJP, …]; rfl`** (small). `Foundation/BatchedBackLinks.lean:97–99`:
   ```lean
   simp only [denStepApp, batchMap, batchMapHasVJP, flatConvHasVJP, HasVJPMat.toHasVJP,
     rowwiseHasVJPMat, HasVJP3.toHasVJP, conv2dHasVJP3]
   rfl
   ```
   (lists differ: :121, :143, :160 omit the per-op `HasVJP3`). Fix: `batchMapHasVJP_backward` apply lemma at
   `BatchMapVJPAt.lean:139`, proved once. ⚠ Foundation file: check its importer count first (§0 rule).

8. **E9 — `BnBetaTiedB`** (small). `EfficientNetStepTieG.lean:93–97` (10 copies):
   ```lean
   ∧ (∀ o : Fin mid,
         den (SHlo.bnBetaGradB (N := N) (oc := mid) (h := h) (w := w) (.operand cotN (reassocB N mid h w cotEc))) o
           = ∑ j : Fin (mid * (N * (h * w))),
               pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe (fun _ => 0) β' (fun _ => 0))
                    be o j * bnchwFwd N mid h w (reassocB N mid h w cotEc) j)
   ```
   Fix: `def BnBetaTiedB` beside `bnBetaGradB_den` (`Foundation/GradNodesB.lean:140`, 9 importers) or in the
   EfficientNet leaf; the proofs' `intro o; exact …bnBetaGradB_den …` lines become one term.

9. **V8 — Tied Props duplicating `_den` statements** (small, med). `ViTFoldGB.lean:285–296` restates
   `rowDenseWeightGradB_den` (:116–126) verbatim; same for 3 more pairs and 4 in `ViTFold.lean`. Fix: state
   `rowDenseWeightGradB_den … : RowDenseWTiedB …` (callers applying `… i j` still elaborate — the def unfolds).
   Check `tests/AuditAxioms*` pins of the `_den` names first.

10. **E2 — duplicate B0 apex** (small–medium, med). `EfficientNetFullB0.lean:371–440` builds the witness with 36
    `have`s and `have e1 := vjpComp _ _ dS d1 vS v1 …`; `EfficientNetFullWholeBackCertifiedTie.lean:219`
    bridges with `(funext fun dy => HasVJP.backward_unique _ _ x dy)`. Fix: move `efficientnetBFullHasVJP`
    (FullWholeBackCertifiedTie.lean:57) into FullB0 and define the concrete witness as its instance. ⚠ the §1(m)
    trap: ENet's conv witnesses are defeq-typed, so an instantiation may not unify syntactically — try it, keep
    `backward_unique` if it doesn't close.

Not low-hanging (medium–large): V2 (`BlockParamsV` through 12 ViT defs), V3 (if-form ×3, `qkvSlabBack`,
`mhsaClean_backward_apply`), E8 (MBConv tail ×7 → `EnTail` leaf). Not recommended: V1 / E3 statement redesign
(bumps already gone; E3's `efficientnet_net_syncTiedG` is in the comparator tier; ENet/ViT whole-net folding
belongs to the certlayer_nets.md thread), E1's `(by decide)`.

---

# Report: audit_mobilenet.md

# audit_mobilenet.md — re-audit against HEAD 28fc373f (2026-09-24)

Static read + grep only. Directory: `LeanMlir/Proofs/Nets/MobileNet/` (25 files, 12,761 lines; was 26 / 14,125).
`grep maxHeartbeats|maxRecDepth` in the directory: **0 option lines** (one comment mention at
`MobileNetV4FullBSeal.lean:65`). Every heartbeat finding below is therefore closed.

## Table

| # | finding (audit line) | status | current location | effort / payoff |
|---|---|---|---|---|
| 1 | V2 tie `mnv2InputGradB_eq_mobilenetv2B_full_vjp` — 1M hb + maxRecDepth 800k | FIXED (bumps out, §1(b)) | `MobileNetV2WholeBackCertifiedTieB.lean:211` | — |
| 1b | …its whole-chain closing `rfl` → rfl-peel `_backward` lemma at variable stages (R34 template `r34BFullHasVJPAt_backward`, §1(f)) | STILL OPEN (residue) | `MobileNetV2WholeBackCertifiedTieB.lean:313-317` | small / low (compiles without bumps; robustness only) |
| 2 | V2 `mnv2InputGradB_correct` bump + 119-line restatement → `HasVJPAt.correct_of_eq` | bump FIXED; `correct_of_eq` STILL OPEN | `MobileNetV2WholeBackCertifiedTieB.lean:323-432` (proof body 418-432) | small / low (statement is pinned in blueprint + AuditAxioms, so only the 12-line `exact (apex …).correct` term shrinks; same shape in R34 `ResNet34BackCertifiedTieB.lean:524` and V4 `:395`) |
| 3 | three N-stage apexes (R34/V2/V4) → `OpaquePrefix` | PARKED — `certlayer_nets.md` (the "next thread" in proof_cleanup ▶ Start here); R34/R50 already one CertLayer (6778f8c9); certlayer_nets.md:195 "MNv2 would not change that [line count]" | `MobileNetV2WholeBackCertifiedTieB.lean:65`, `MobileNetV4WholeBackCertifiedTieB.lean:131` | — |
| 4 | `mobilenetv2ForwardBFull_eq_slots` 19-def `simp only` + private `comp3_assoc` | PARKED — proof_cleanup §3.6 ("MobileNetV2 `eq_slots` / the 18 `mnv2PreB*_apply` lemmas stay"); `comp3_assoc` now carries a docstring explaining the variable-level trick | `MobileNetV2WholeBackCertifiedTieB.lean:442, 456-482` | — |
| 5 | V4 tie `mnv4InputGradB_eq_mnv4B_full_vjp` 2M hb; `_correct` 2M; `eq_slots` maxRecDepth | FIXED (bumps out); closing `rfl` residue same as 1b | `MobileNetV4WholeBackCertifiedTieB.lean:251, 378-389, 395, 620` | 1b-style peel: small / low |
| 6 | `MobileNetV2FullBVJP` file-wide `maxHeartbeats 1000000` | FIXED | — | — |
| 6b | …its docstring "this tier carries no numerals" is false (`mnv2PreB*` are at 112/56/28/14/7) | STILL OPEN | `MobileNetV2FullBVJP.lean:50` vs `:237-290` | trivial / low |
| 7 | `mobilenetv2ForwardBFullHasVJPAt` 38 binders / 51 `have`s → V4 shape | FIXED — §1(o) (`MNV2PosB`, `MNV2SmoothAtB`, `vjpCompDiffAt`, `…_differentiableAt` exported) | `MobileNetV2FullBVJP.lean:297, 320, 347` | — |
| 8 | `mnv2PreB0…17` + 18 `_apply` + 19-name `rw` | PARKED — §3.6 | `MobileNetV2FullBVJP.lean:237-290, 440-520` | — |
| 9 | `mnv2ResidB_differentiableAt` `show biPath` / no `residual_apply`/`residual_differentiableAt` | FIXED — §1(m) (`Architectures/Residual.lean:182,185`; `MobileNetV2FullBVJP.lean:180` uses it, `MobileNetV2FullBSeal.lean:216` uses `residual_apply`) | residue: `MobileNetV2FullPaperEval.lean:238` `unfold ivExpOnlyEvalW residual biPath`; V4 `resid_id` `MobileNetV4FullBSeal.lean:250` `show L.fwd v k + v k = v k` (CertLayer.residual; `CertLayer.residual_fwd` exists) | trivial / low |
| 10 | `mnv2_net_tiedB` 1.6M hb, `g` embedded → binder; same for `mnv2_net_syncTiedB` | FIXED — §1(o) | `MobileNetV2StepTieB.lean`, `MobileNetV2SyncStepTieB.lean` | — |
| 11 | `*CotIn_eq_vjp` ×4: local `hd` re-proves `den_operand`; unused `cotN : String` binder | STILL OPEN | `MobileNetV2StepTieB.lean:156-165, 240-250, 263-272, 341-351` | trivial / low (only consumers are 4 `#print axioms` pins, `tests/AuditAxioms.lean:1915-1918` — names unchanged) |
| 12 | `relu6MaskB` vs `reluMaskB` → mask-generic IB cotangent chain | STILL OPEN | `MobileNetV2StepTieB.lean:83`, `Foundation/BatchedBackLinks.lean:443` | large / high-lines but invasive (touches plain/_smul/Sync/_shard chains of 2+ nets); not a low-hanging item |
| 13 | 53 `*_smul` hand chains → `Homog` predicate | FIXED — §1(p) (`IsHomog`, 25 uses V2 Sync, 27 V4 Sync); the per-link `rw` chains remain as the proofs of `IsHomog` statements, by design | `MobileNetV2SyncStepTieB.lean:110-113` | — |
| 14a | `*_shard` lemmas close with bare `rfl`; `relu6MaskB_shard` declared but never used | STILL OPEN | 11 bare `rfl`s in `MobileNetV2SyncStepTieB.lean` (e.g. `:295, 308, 382, 395, 408, 420, 488, 501, 516, 598`), 13 in `MobileNetV4SyncStepTieB.lean`; `relu6MaskB_shard` at `MobileNetV2SyncStepTieB.lean:221` (only other mention: docstring `:30`) | trivial / low — either cite it (`exact relu6MaskB_shard _ _ r`) or delete it as dead (audit_v2 §4 already deleted the R34 twin `reluMaskB_shard`) |
| 14b | `nhw_ne_zero hN hh hw` pair re-derived 23× → one `nhw_pair` | PARKED — §3.6 (`[NeZero N]`/threading not done; `IsShardwise` parked) | `MobileNetV2SyncStepTieB.lean:287, 300, 374, …` | — |
| 15 | seal `ed*` carrier steps (22 V2 / 15 V4) → `EDiff_ctConvBn`/`ctDwBn` kit | FIXED — §1(n) (`eDiff_convBn`/`dwBn`/`dwS2Bn`/…) | — | — |
| 16 | seal tail (`sealX_nonconstant`/`_jacobian_nonzero`) ×4 nets | FIXED — §1(n) (`ne_of_ray_readout`, `fderiv_ne_zero_of_ray_readout`) | `MobileNetV2FullBSeal.lean:1028-1036`, V4 `:1026-1034` | — |
| 17 | `seal_differentiableAt` re-derives the 18-step chain | FIXED — §1(o)/(r) (one term each) | `MobileNetV2FullBSeal.lean:1021`, V4 `:1020` | — |
| 18 | `sealExpB_eq`/`sealStridedB_eq`/`sealNoExpB_eq` undocumented 5-line `show` re-spelling the block | STILL OPEN | `MobileNetV2FullBSeal.lean:229, 249, 264` (+ `head_eq_dense` `:988`, V4 `sealCTStrided_eq` `:277`); V4's `pc*` `show`s (`:813-907`) now carry a section comment (`:801-806`) — acceptable | small / low (add `mnv2ExpOnlyB_apply`/`mnv2StridedB_apply`/`mnv2NoExpB_apply` rfl lemmas at variable shapes, or one comment per site) |
| 19a | V4 `sealUib_ok`/`sealUibStrided_ok` slot discharge written out 3× | STILL OPEN | `MobileNetV4FullBSeal.lean:302-306, 308-312, 324-328` | small / low (`postDWSlot_ok`/`preDWSlot_ok` lemma) |
| 19b | `Rr_pos` 15/22-deep hand-nested `mul_pos` | STILL OPEN (documented — comment explains why not `repeat' apply mul_pos`) | `MobileNetV2FullBSeal.lean:929-…`, `MobileNetV4FullBSeal.lean:770-…` | trivial / low: `repeat' (first \| exact rf_pos _ _ \| apply mul_pos)` tries `rf_pos` before splitting, which is exactly the objection the comment raises (unmeasured) |
| 20 | `relu6LinearPart_apply` / `pdiv_relu6` / `relu6HasVJPAt` `ite_eq_*` chains | FIXED — moved to `Foundation/MLP.lean:409-455`: `split_ifs <;> rfl`, `pdiv_of_hasFDerivAt_mask`, `simp_rw …; simp` | — | — |
| 21 | 7 copies of stage VJP + back-graph faithfulness skeleton | STILL OPEN (co-located now: `Foundation/BatchedStageLayers.lean` §2.5 of audit_v2, but still separate `bnRelu6StageHasVJPAt` `:58` / `bnReluStageHasVJPAt` `:287` and `cbr…`/`dwbr…`/`cbRelu…` `_faithful` `:152, 173, 199, 337, 412, 469`) | medium / low (each copy is 6-10 lines; a generic `bnActStage` saves little) |
| 22 | `mobilenetv2FwdGraphPaperEval_faithful` maxRecDepth 20000 + closing whole-net `rfl` | bump FIXED; closing `rfl` at 224/112 literals remains | `MobileNetV2FullPaperEval.lean:305-312` | medium / low (stem/head graph lemmas; compiles without the bump) |
| 23 | `mobilenetv2FwdGraphBFull_faithful` 19-entry ordered `rw` → order-free `simp only` | REJECTED by measurement — `MobileNetV4FullB.lean:847-850`: the `simp only` spelling "elaborates for ~9 minutes and then dies in the KERNEL"; the audit itself said "if the kernel cost rises, keep `rw`" | `MobileNetV2FullB.lean:356-363`, `MobileNetV4FullB.lean:844` | — |
| M1 | `have h112 : 0 < 112 := by norm_num` ×5 per capstone | STILL OPEN | `MobileNetV2SyncB.lean:308-312`, `MobileNetV2SyncStepTieB.lean:1011…`, `MobileNetV4SyncStepTieB.lean:1303-1305`, `MobileNetV4SyncB.lean:679` | trivial / very low (idiom) |
| M2 | `MobileNetV2Fold.lean` `show depthwiseWeightSgdDen …` relies on `den` unfolding | STILL OPEN — ⚠ the audit's `simp only [den]` fix is now FORBIDDEN (§1(g) / proof_cleanup ▶ trap 2: 233 s `den.eq_def`); correct fix is a one-line comment or `denStep` | `MobileNetV2Fold.lean:58, 68` | trivial / low |
| M3 | `congr 1; congr 1` → `congr 2` | STILL OPEN | `MobileNetV2Fold.lean:43-44` | trivial / very low |

**Counts (27 rows incl. sub-rows):** FIXED 13 (1, 5, 6, 7, 9, 10, 13, 15, 16, 17, 20, + bump halves of 2 and 22) ·
PARKED/REJECTED 5 (3, 4, 8, 14b, 23) · STILL OPEN 14 (1b, 2-correct_of_eq, 5-peel, 6b, 11, 12, 14a, 18, 19a, 19b, 21, 22-rfl, M1, M2, M3 — 1b/5-peel counted as one item below) · GONE 0.
(The MNv2 per-example legacy chain deleted in 5c4c9101 carried none of this audit's findings.)

## Still open, ranked (low-hanging first)

### 1. `*CotIn_eq_vjp` ×4 — drop `cotN`, use `den_operand` (trivial)
`MobileNetV2StepTieB.lean:156-165` (same at 240, 263, 341):
```lean
theorem mnv2NoExpCotIn_eq_vjp (N h w : Nat) {ic oc : Nat} (p : IVWNoExp ic oc)
    (hq : IVNoExpPos p) (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w)))
    (hs : IVNoExpSmoothAtB N h w p xin) (cotN : String) :
    mnv2NoExpCotIn N h w p xin dyOut = (mnv2NoExpBHasVJPAt N h w p hq xin hs).backward dyOut := by
  have h := mnv2NoExpBackGraph_faithful p hq xin (.operand cotN dyOut) hs
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
```
`den_operand` is `@[simp]` at `Codegen/StableHLO.lean:2354`. `cotN` is not in the statement; the
only consumers are the four `#print axioms` lines (`tests/AuditAxioms.lean:1915-1918`), so the
binder can go with no caller edits. Payoff low (idiom).

### 2. `relu6MaskB_shard` — cite it or delete it (trivial)
`MobileNetV2SyncStepTieB.lean:221`:
```lean
theorem relu6MaskB_shard {R N n : Nat} (PRE DY : Vec ((R * N) * n)) (r : Fin R) :
    relu6MaskB (N * n) (batchShard R N n PRE r) (batchShard R N n DY r)
      = batchShard R N n (relu6MaskB ((R * N) * n) PRE DY) r := rfl
```
while `:293-295` (and ~10 more) close exactly that goal with a bare `rfl`:
```lean
  unfold mnv2NoExpSyncCotDn
  rw [mnv2NoExpSyncCotPc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl
```
audit_v2 §4 deleted the R34 twin `reluMaskB_shard` as dead; either do the same here or make the
`rfl`s `exact relu6MaskB_shard _ _ r` so the defeq has a name. Payoff low.

### 3. `MobileNetV2FullBVJP.lean:50` false docstring (trivial)
```
⭐ `N` is a variable throughout: this tier carries no numerals.
```
but `:237-239` `mnv2PreB0 … := mnv2StemB N 112 112 …` (and the whole prefix ladder is at
112/56/28/14/7). Reword to "`N` is a variable throughout; the widths are the paper's". Payoff low
(doc accuracy).

### 4. `Rr_pos` nests (trivial, unmeasured)
`MobileNetV2FullBSeal.lean:929-935` / `MobileNetV4FullBSeal.lean:770-776`:
```lean
theorem Rr_pos (nCls : Nat) (t : ℝ) : 0 < Rr nCls t := by
  unfold Rr
  -- ⚠ not `repeat' apply mul_pos`: `rf` is itself a product, so `mul_pos` splits inside it and
  -- leaves `0 < 1/64` goals `rf_pos` cannot close. One factor per carrier BatchNorm, explicitly.
  exact mul_pos (rf_pos _ _)
    (mul_pos (rf_pos _ _)
```
`repeat' (first | exact rf_pos _ _ | apply mul_pos)` tries `rf_pos` first, so it never splits
inside `rf` — the comment's objection doesn't apply to it. Replaces a 22-deep (V2) / 15-deep (V4)
term that must mirror `Rr`'s parenthesisation. Needs one compile to confirm.

### 5. Residual residue (trivial)
`MobileNetV2FullPaperEval.lean:238`:
```lean
  unfold ivExpOnlyEvalW residual biPath
  rfl
```
`MobileNetV4FullBSeal.lean:248-252`:
```lean
    (CertLayer.residual L).fwd v = v := by
  funext k
  show L.fwd v k + v k = v k
  rw [hL]
```
→ `rw [CertLayer.residual_fwd, residual_apply, hL]` (both lemmas exist: `Architectures/Residual.lean:182`,
`CertLayer.residual_fwd` used at `MobileNetV4FullB.lean:680`). Payoff low.

### 6. `MobileNetV2Fold.lean` show + congr (trivial) — ⚠ not the audit's fix
`:43-44` `congr 1` / `congr 1` → `congr 2`. `:58` / `:68`:
```lean
  show depthwiseWeightSgdDen b x W lr cot idx = _
  exact mnv2_render_depthwiseW_flat_certified b x W cot lr idx
```
Add a comment ("`den` of this constructor is `depthwiseWeightSgdDen` by its match arm"); do NOT
use the audit's `simp only [den]` (233 s `den.eq_def`, proof_cleanup §1(g)).

### 7. `correct_of_eq` for the three `_correct` apex readings (small)
`MobileNetV2WholeBackCertifiedTieB.lean:418-432`, `MobileNetV4WholeBackCertifiedTieB.lean:515-530`,
`ResNet34BackCertifiedTieB.lean:524-535` all do
```lean
  rw [congrFun (mnv2InputGradB_eq_mobilenetv2B_full_vjp N Ws … h_head) dy]
  exact (mobilenetv2PaperPCHasVJPAt (mnv2StemB N 112 112 Ws bs εs γs βs) b1 … x
          ⟨…⟩ hb1 … ⟨…⟩ ⟨…⟩ ⟨…⟩).correct dy i
```
re-typing the 12-line apex term from the tie's statement. A `HasVJPAt.correct_of_eq (hf) (hB : B = hf.backward)`
leaf lemma lets the `exact` be `.correct_of_eq (tie …) dy i` with the apex term inferred. Statements
stay (blueprint + AuditAxioms pin them). Cross-net (R34 is another fork's file). Payoff low-med.

### 8. V2/V4 tie closing `rfl` → `_backward` rfl-peel (small)
`MobileNetV2WholeBackCertifiedTieB.lean:313-317`:
```lean
  unfold mnv2InputGradB
  rw [mnv2StemBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbrBBack_eq_vjp_backward (by decide) (by decide) Wh bh εh hεh γh βh _ h_head,
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  rfl
```
(V4 `:378-389` same.) Template: `r34BFullHasVJPAt_backward` (`ResNet34BackCertifiedTieB.lean:259`, §1(f)).
Bumps are already gone, so this is robustness against the §0 "never a closing `rfl` through the
concrete chain" rule, not speed. Overlaps certlayer_nets.md (if V2/V4 become one CertLayer, it goes).

### 9. Seal block-collapse `show`s (small)
`MobileNetV2FullBSeal.lean:229-235` (and 249, 264, 988; V4 277):
```lean
  show projB N (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0) 1 (kv oc (1 / 64)) (kv oc 0)
      (StableHLO.dwbrB N (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0) 1 (kv mid (1 / 64))
        (kv mid 3)
        (StableHLO.cbrB N (h := h) (w := w) (ctK mid ic 1 1 1) (kv mid 0) 1 (kv mid (1 / 64))
          (kv mid 3) v)) = _
  rw [cbrB_eq _ _ hm, dwbrB_eq _ _ hm]
  rfl
```
Undocumented; `mnv2ExpOnlyB_apply`-style rfl lemmas at variable shapes in `MobileNetV2FullB.lean`
would make each a `rw`. Low payoff.

### 10. V4 slot discharge ×3 (small)
`MobileNetV4FullBSeal.lean:302-306` (and 308, 324):
```lean
  · show (mnv4PreDWSlot (h := s.h) (w := s.h) N s.preDWk _ _ _ _ _ _).ok _
    unfold mnv4PreDWSlot
    by_cases hk : s.preDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; exact bne N s.ic s.h s.h hm _
```
→ `preDWSlot_ok` / `postDWSlot_ok` lemmas. Low payoff.

### Not low-hanging (listed for completeness)
* #12 mask-generic inverted-residual cotangent chain — large, invasive.
* #21 stage-VJP/back-graph skeleton — medium, each copy is short; low payoff.
* #22 PaperEval closing `rfl` at literal shapes — medium; compiles without a bump now.
* M1 `have h112 : 0 < 112 := by norm_num` — idiom only.

---

# Report: audit_resnet_small_convnext.md

# Re-audit: audit_resnet_small_convnext.md (at 28fc373f, static read + grep only)

Scope check first: `grep -rn "set_option max" LeanMlir/Proofs/Nets/{ResNet,Small,ConvNeXt}` returns
NOTHING — every heartbeat/recDepth finding in this report is gone (§1(b), §1(f)).

| # | finding (audit line) | status | current location | effort / payoff |
|---|---|---|---|---|
| 1 | R34/R50 whole-back tie: 800k recDepth + 1–2M heartbeats, closing `rfl` (:21) | FIXED — §1(f) (R34 peel `r34BFullHasVJPAt_backward`), §1(b) (R50 bumps out) | `ResNet34BackCertifiedTieB.lean:259, 345`; R50 tie `ResNet50WholeBackCertifiedTieB.lean:150` still closes by bare `rfl` but at binder `q`, no bump | — |
| 2 | two prefix vocabularies `r34PreK`/`r50PreK` + `_apply` vs `opaqueA_K` (:65) | PARKED — proof_cleanup §3.1 → certlayer_nets.md §4.3 "Retiring the prefixes" (public vocab, comparator-tier change) | `ResNet34FullBVJP.lean:210-…`, `Foundation/OpaquePrefix.lean` | — |
| 3 | ConvNeXt capstones 16M/400k + ~160 loose binders (:93) | FIXED (the smell: bumps out, §1(b); speed premise disproved — capstones elaborate in ~s). Residual: binder list still loose, not over `CnxTWeightsCh` — large, low payoff, `cnx_net_tiedGB` is in the comparator tier | `ConvNeXtStepTie.lean:461`, `ConvNeXtStepTieGB.lean:381` | (large / low) |
| 4 | hand-rolled `@[irreducible]` wrappers + `unfold` (:121) | STILL OPEN | `ConvNeXtStepTieGB.lean:302-352` (4 `*TiedGBAt`), `ConvNeXtStepTie.lean:305-426` (11) | small / low (needs a measure: are they still needed without the bumps?) |
| 5 | R34/R50 (+sync) StepTieB 1.6M bumps, let telescopes (:141) | FIXED — §1(b) (no bumps; lets remain, 38–54 per file, harmless now) | `ResNet34StepTieB.lean:454` | — |
| 6 | `*TiedB` Props restate `_den` lemmas; 218 `intro idx; exact …_den` delegations (:158) | STILL OPEN (the `_den` lemmas moved to `Foundation/GradNodesB.lean`, audit_v2 §2.4; the duplication did not change) | `Foundation/GradNodesB.lean:48` (`convWGradB_den`) vs `:209` (`ConvWTiedB`); 218 sites / 13 files (EffNet 68, ViT 38, MNv2 22, Cifar8(Bn) 32, R34 14, R50 11, CNX 21, Cifar/Cnn 12) | medium / medium |
| 7 | 151 `_smul` lemmas with spelled-out `fun i => s * dy i` (:182) | FIXED — §1(p) (`IsHomog`) | `Foundation/DataParallelSync.lean` | — |
| 8 | file-wide `maxHeartbeats 1000000` in R34/R50 FullBVJP (:211) | FIXED — §1(b) | — | — |
| 9 | whole-net `HasVJPAt` built in tactic mode (`have` of data), forcing a 2nd term-mode apex (:222) | ResNet FIXED (6778f8c9: apex = `(r34NetLayer …).vjp`, `ResNet34FullBVJP.lean:488`, `ResNet50FullBVJP.lean:423`). STILL OPEN for `CifarCNN.lean:77/441/803` and `ConvNeXtFullT.lean:294` (+ its term-mode twin `convNextForwardTChVjpChain`, `ConvNeXtWholeBackCertifiedTie.lean:520`) | see left | medium / low (compiles fine; duplication only) |
| 10 | `seal_differentiableAt` rebuilds the apex's differentiability (:249) | FIXED — §1(t) (+ §1(r) MNv4, §1(o) MNv2) | — | — |
| 11 | ~77 per-slot seal lemmas `nn/pc/sc/ed/cn` ×2 nets (:269) | `cn*` GONE (3550f132, continuity by `fun_prop`); `nn/pc/sc_/ed` STILL OPEN: 63 in R34, 62 in R50 | `ResNet34FullBSeal.lean:392-713` | medium / low (fast today; line count only) |
| 12 | seal `show`s over `r34IdB = relu ∘ residual`, no `relu_residual_const` (:297) | PARTIAL: `residual_apply` (Architectures/Residual.lean:182) now used; `projB_zero_const` moved to BatchSealKit (audit_v2 §2.5). STILL OPEN: the `show relu (…) (residual _ v) k = …` unfolds + 4 copies of the relu-shift closing | `ResNet34FullBSeal.lean:168, 187, 190`; `ResNet50FullBSeal.lean:202, 237, 240, 264, 267, 299-365` | small / low |
| 13 | batched leaf ties `show` the `batchMapAux` index layout (6 CNX + 3 ViT) (:323) | STILL OPEN | `ConvNeXtWholeBackCertifiedTieB.lean:205, 221, 241, 258, 299, 311`; `ViTWholeBackCertifiedTieB.lean:97, 110, 126`; lemma would go in `Foundation/BatchMapVJPAt.lean` (after `batchMapHasVJPAt`, :83) | small / medium |
| 14 | file-wide `maxRecDepth 100000` in both CNX whole-back ties (:352) | FIXED — §1(b) | — | — |
| 15 | `vjpCompDiffAt`(+peel) in net files; `BackwardMaps` imports `Nets.ResNet.ResNet34` (:368) | FIXED — §1(m) (both in `Foundation/Tensor.lean:456-481`); BackwardMaps imports no net (audit_v2 §2.6; ResNet34.lean deleted 5c4c9101); `ResNet34BackCertifiedTie → CifarCNN` edge gone (now `Architectures/ConvBackCertifiedTie`) | — | — |
| 15b | (same finding's tail) ConvNeXt files import `ResNet34Fold` for shared leaf lemmas | STILL OPEN | `ConvNeXtStepTie.lean:4`, `ConvNeXtFoldG.lean:4` → `Nets/ResNet/ResNet34Fold.lean` (85 lines, a RETIRED-artifact file whose only content is `convStridedW_den`/`convStridedB_den`, used only by ConvNeXtStepTie ×4) | trivial / low-med |
| 16 | Small `*_conv_tied_certified` maxRecDepth 4k→32k doubling with let depth (:393) | FIXED — §1(b) (no option left in Nets/Small) | — | — |
| 17 | CifarCNN re-derives the raw-point pool VJP with `rw [← hpt]` (:415) | STILL OPEN (22 sites). `maxPoolFlatHasVJPAt'` now lives in `Nets/Small/ChapterGraphTies.lean:87` (audit_v2 §2.6), which IMPORTS CifarCNN — still unreachable from it | `CifarCNN.lean:151-156, 603-608, 936-…`; `MnistCNN.lean:138-140` | small / low-med (⚠ changes the apexes' `.backward` spelling — check `rfl` consumers) |
| 18 | 6 `*LossCot_den` copies (+5 `*_tied_totalloss`) (:437) | STILL OPEN | `MlpFold.lean:133`, `CnnFold.lean:150`, `CifarFold.lean:100`, `Cifar8StepTie.lean:34`, `Cifar8BnStepTie.lean:25`, `ConvNeXtStepTie.lean:293` (K = 10 hard-coded) | small / low-med |
| 19 | `cbReluBackBatchedGraph_faithful` closes by definitional `simp only` (:461) | FIXED — §1(m) (`vjpCompAt_backward` exists, `Tensor.lean:456`, and is in the set; decl moved to `Foundation/BatchedStageLayers.lean:337`) | — | — |
| 20 | `_correct` corollaries copy 16 PProd / 35 bundle binders (:484) | FullBVJP half FIXED — §1(t) (`R34PosB`/`R34SmoothAtB`, `ResNet34FullBVJP.lean:499`). WholeBack half PARKED — certlayer_nets §4.3 ("binder resolutions for R34"); `r34InputGradB_eq_r34B_full_vjp` is in the comparator tier (`gen_comparator_tier.py:64`) | `ResNet34BackCertifiedTieB.lean:444-…` | — |

Counts: FIXED 11 (1, 3, 5, 7, 8, 10, 14, 15, 16, 19, + ResNet half of 9) · PARKED 2 (2, 20) · STILL OPEN 8
(4, 6, 9-rest, 11, 12, 13, 15b, 17, 18 — 9 rows counting 9's residue) · GONE 1 (`cn*` part of 11).
Not resurrected: 3's and 5's "the bump pays for elaboration → restate over records" (the bumps went
with no restatement, §1(b)); 1's "~10 s" prediction (measured 3 s, §1(f)).

## Still open, ranked (low-hanging first)

### A. #15b — move `convStridedW_den`/`convStridedB_den` out of the retired `ResNet34Fold` (trivial, low-med)

`Nets/ResNet/ResNet34Fold.lean:1-3, 56-85`:
```lean
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2Close
import LeanMlir.Proofs.Nets.Small.CifarFold
import LeanMlir.Proofs.Foundation.SgdNodes
...
namespace Proofs.ResNet34PoC
theorem convStridedW_den … theorem convStridedB_den …
end Proofs.ResNet34PoC
```
Its docstring says the artifact is "⛔ RETIRED (2026-09-06)". Only `ConvNeXtStepTie.lean` uses the two
lemmas (4 refs); `ConvNeXtFoldG.lean:4` imports the file and uses neither (possibly a dead import —
confirm by compiling). They are per-example SGD node lemmas, which is what `Foundation/SgdNodes.lean`
holds (audit_v2 §2.5 moved `CifarPoC.convW/B_den` there). Move them with names kept, drop the
sideways ConvNeXt → ResNet edge. ⚠ SgdNodes must be able to import `mnv2_render_stem_conv{W,b}_certified`'s home.

### B. #13 — one `batchMapAux` row-lift lemma (small, medium)

`ConvNeXtWholeBackCertifiedTieB.lean:220-225` (the shape of all nine):
```lean
  funext dy idx
  show chanLNTensor3Back c h w ε γ (Mat.unflatten v (finProdFinEquiv.symm idx).1)
      (fun c => dy (finProdFinEquiv ((finProdFinEquiv.symm idx).1, c)))
      (finProdFinEquiv.symm idx).2 = _
  rw [chanLNTensor3Back_eq_chanLN_vjp (β := β) ε hε γ]
  rfl
```
Each copy re-spells `batchMapHasVJPAt`'s backward field (`Foundation/BatchMapVJPAt.lean:87-90`).
Add `batchMapAux_eq_batchMapHasVJPAt_backward` (statement as in the audit, :342-347) plus a
point-independent twin for `batchMapHasVJP` (stem, dense); each leaf tie becomes a term. The
lemma goes in `BatchMapVJPAt.lean` (a leaf).

### C. #18 — one generic `softmaxCELossCot_den` (small, low-med)

`Small/CnnFold.lean:150-159` (five others identical up to the forward):
```lean
theorem cnnLossCot_den {ic c h w d1 nClasses kH kW : Nat} (nlogN ohN : String)
    (W₁ …) (b₅ : Vec nClasses) (x : Vec (ic*(2*h)*(2*w))) (label : Fin nClasses) :
    den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe
            (.operand nlogN (mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x))))
          (.operand ohN (oneHot nClasses label)))
      = fun j => softmax nClasses (mnistCnnNoBnForward …) j - oneHot nClasses label j := by
  funext j; simp only [denStepApp, softmax]
```
`ConvNeXtStepTie.lean:293` already has the logits-generic form at `K = 10`. Make it `{K}` in
`Nets/Small/LinearTrainStep.lean` (next to `lossCot_eq_softmax_sub_onehot`, :34); the six become
one-term instances (names kept for callers/pins).

### D. #12 — `show` over `r34IdB`'s definition in the seals (small, low)

`ResNet34FullBSeal.lean:167-169`:
```lean
  funext k
  show relu (N * (c * h * w)) (residual _ v) k = v k + 1
  rw [relu_id_of_pos (fun i => by rw [hres i]; linarith [hv i]), hres k]
```
Same at :190 (`residualProj`), R50 :202/:240/:267. Add `r34IdB_apply`/`r34DownB_apply` (rfl) and a
`relu_residual_shift` lemma in `Training/BatchSealKit.lean` (4 copies of the closing).

### E. #4 — `@[irreducible]` wrappers (small, low; measure first)

`ConvNeXtStepTieGB.lean:300-352`:
```lean
/-! ## `@[irreducible]` wrappers — keep the 22-deep capstone thread opaque (the r34/mnv2 heartbeat lesson) -/
@[irreducible] def cnxHeadChTiedGBAt … : Prop := cnxHeadChTiedGB N xN …
theorem cnx_head_ch_tiedGBAt … := by
  unfold cnxHeadChTiedGBAt
  exact cnx_head_ch_tiedGB …
```
The "heartbeat lesson" they encode predates §1(b). Try deleting them (15 sites) and compiling; if
still needed, `irreducible_def`. `cnx_net_tiedGB` is comparator-tier — the wrappers appear in its
statement, so removing them is a tier regen.

### F. #17 — CifarCNN's 11 pool re-derivations (small, low-med)

`Small/CifarCNN.lean:603-608`:
```lean
  have hpt1 : Tensor3.flatten (Tensor3.unflatten z1 : Tensor3 c1 …) = z1 := Tensor3.flatten_unflatten z1
  have mp1_v : HasVJPAt (maxPoolFlat c1 (2*(2*(2*h))) (2*(2*(2*w)))) z1 := by
    rw [← hpt1]; exact maxPoolFlatHasVJPAt _ hp1
  have mp1_d : DifferentiableAt ℝ (maxPoolFlat c1 …) z1 := by
    rw [← hpt1]; exact maxPoolFlat_differentiableAt _ hp1 hc1 (by omega) (by omega)
```
Move `maxPoolFlatHasVJPAt'` (`ChapterGraphTies.lean:87`) down into `MnistCNN.lean` or
`Architectures/CNN.lean`, add `maxPoolFlat_differentiableAt'`, and swap the 11 blocks. ⚠ The
apexes are tactic-mode data (#9), so the `.backward` of the result changes spelling
(`maxPoolBackFlat` instead of an `Eq.mpr`-transported one); grep `cifarCnn*HasVJPAt` consumers
that close by `rfl` before switching.

### G. #6 — `*TiedB` Props vs `_den` lemmas (medium, medium)

`Foundation/GradNodesB.lean:48` `theorem convWGradB_den … (idx) : den (…) idx = ∑ n, ∑ j, …` and
`:209` `def ConvWTiedB … : Prop := ∀ idx, den (…) idx = …` — two texts of one formula.
`ResNet34StepTieB.lean:304-308`:
```lean
  unfold r34IdTiedB
  intro r1 c1 c2 cotA cotC2 cotN1 cotC1
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact ResNet34PoCB.convWGradB_den xN cotN p.b₁ xin p.W₁ cotC1 idx
  · intro o;   exact ResNet34PoCB.convBGradB_den cotN p.W₁ xin p.b₁ cotC1 o
```
Cheapest form without renaming: add `…_tied : ConvWTiedB … := convWGradB_den …` term lemmas in
GradNodesB (one per Prop) and swap the 218 sites file by file; the `refine`/`intro` scaffolding
then collapses to anonymous-constructor terms. Mechanical but touches 13 files incl. 4 comparator-
tier modules (proofs only, statements unchanged).

### H. #11 — seal per-slot lemmas (medium, low)

`ResNet34FullBSeal.lean:392-395, 464-467, 711-713`:
```lean
theorem nn9 (nCls : Nat) (t : ℝ) : ∀ k, 0 ≤ r34Pre9 2 (sealW nCls) (sealX t) k := by
  intro k; rw [r34Pre9_apply]; exact r34IdB_nonneg 2 14 14 256 _ _ k
theorem pc9 … := by rw [r34Pre9_apply]; exact sealIdB_eq 2 14 14 256 (by norm_num) _ (nn8 nCls t)
theorem ed9 … := by rw [pc9]; exact eDiff_shift _ _ 1 (ed8 nCls t)
```
125 lemmas across two files; fast today. A `sealIdSlot` step lemma would cut them ~4→1 per slot, but
it names `r34PreK` (parked vocabulary, #2) — do it only alongside certlayer_nets work.

### I. #9 residue — tactic-mode apexes in CifarCNN ×3 and ConvNeXtFullT (medium, low)

`ConvNeXtFullT.lean:294-…` (`convNextForwardTChHasVJP … := by have st_diff … have e1 := vjpComp …`)
still forces the term-mode twin `convNextForwardTChVjpChain` (`ConvNeXtWholeBackCertifiedTie.lean:520`).
Rewriting the apex in term mode (as ResNet's now is) would let the twin go. Only duplication; no
compile-time cost recorded.

---
