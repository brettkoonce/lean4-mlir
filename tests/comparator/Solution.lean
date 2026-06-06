import LeanMlir.Proofs.Tensor
import LeanMlir.Proofs.MLP
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.BatchNorm
import LeanMlir.Proofs.Residual
import LeanMlir.Proofs.Depthwise
import LeanMlir.Proofs.SE
import LeanMlir.Proofs.LayerNorm
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.ConvNeXt
import LeanMlir.Proofs.EfficientNet

open Proofs
open scoped Real

-- Foundation ────────────────────────────────────────────────────────

theorem chk_pdiv_comp {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m) (hf : DifferentiableAt ℝ f x)
    (hg : DifferentiableAt ℝ g (f x))
    (i : Fin m) (k : Fin p) :
    pdiv (g ∘ f) x i k =
    ∑ j : Fin n, pdiv f x i j * pdiv g (f x) j k :=
  pdiv_comp f g x hf hg i k

theorem chk_pdiv_add {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k + g y k) x i j
    = pdiv f x i j + pdiv g x i j :=
  pdiv_add f g x hf hg i j

theorem chk_pdiv_mul {m n : Nat} (f g : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => f y k * g y k) x i j
    = pdiv f x i j * g x j + f x j * pdiv g x i j :=
  pdiv_mul f g x hf hg i j

theorem chk_pdiv_id {n : Nat} (x : Vec n) (i j : Fin n) :
    pdiv (fun y : Vec n => y) x i j = if i = j then 1 else 0 :=
  pdiv_id x i j

theorem chk_pdiv_const {m n : Nat} (c : Vec n) (x : Vec m)
    (i : Fin m) (j : Fin n) :
    pdiv (fun _ : Vec m => c) x i j = 0 :=
  pdiv_const c x i j

theorem chk_pdiv_reindex {a b : Nat} (σ : Fin b → Fin a) (x : Vec a)
    (i : Fin a) (j : Fin b) :
    pdiv (fun y : Vec a => fun k : Fin b => y (σ k)) x i j =
    if i = σ j then 1 else 0 :=
  pdiv_reindex σ x i j

theorem chk_pdiv_finset_sum {m n : Nat} {α : Type*} [DecidableEq α]
    (S : Finset α) (f : α → Vec m → Vec n) (x : Vec m)
    (hdiff : ∀ s ∈ S, DifferentiableAt ℝ (f s) x)
    (i : Fin m) (j : Fin n) :
    pdiv (fun y k => ∑ s ∈ S, f s y k) x i j =
    ∑ s ∈ S, pdiv (f s) x i j :=
  pdiv_finset_sum S f x hdiff i j

theorem chk_pdivMat_rowIndep {m n p : Nat} (g : Vec n → Vec p)
    (h_g_diff : Differentiable ℝ g)
    (A : Mat m n) (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin p) :
    pdivMat (fun M : Mat m n => fun r => g (M r)) A i j k l =
    if i = k then pdiv g (A i) j l else 0 :=
  pdivMat_rowIndep g h_g_diff A i j k l

-- Mat-level ─────────────────────────────────────────────────────────

theorem chk_pdivMat_comp {a b c d e f : Nat}
    (F : Mat a b → Mat c d) (G : Mat c d → Mat e f)
    (A : Mat a b)
    (hF_diff : DifferentiableAt ℝ
      (fun v : Vec (a * b) => Mat.flatten (F (Mat.unflatten v))) (Mat.flatten A))
    (hG_diff : DifferentiableAt ℝ
      (fun u : Vec (c * d) => Mat.flatten (G (Mat.unflatten u))) (Mat.flatten (F A)))
    (i : Fin a) (j : Fin b) (k : Fin e) (l : Fin f) :
    pdivMat (G ∘ F) A i j k l =
    ∑ p : Fin c, ∑ q : Fin d,
      pdivMat F A i j p q * pdivMat G (F A) p q k l :=
  pdivMat_comp F G A hF_diff hG_diff i j k l

theorem chk_pdivMat_matmul_left_const {m p q : Nat} (C : Mat m p) (B : Mat p q)
    (i : Fin p) (j : Fin q) (k : Fin m) (l : Fin q) :
    pdivMat (fun B' : Mat p q => Mat.mul C B') B i j k l =
    if l = j then C k i else 0 :=
  pdivMat_matmul_left_const C B i j k l

theorem chk_pdivMat_scalarScale {m n : Nat} (s : ℝ) (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin m) (l : Fin n) :
    pdivMat (fun M : Mat m n => fun r c => s * M r c) A i j k l =
    if i = k ∧ j = l then s else 0 :=
  pdivMat_scalarScale s A i j k l

theorem chk_pdivMat_transpose {m n : Nat} (A : Mat m n)
    (i : Fin m) (j : Fin n) (k : Fin n) (l : Fin m) :
    pdivMat (fun M : Mat m n => Mat.transpose M) A i j k l =
    if j = k ∧ i = l then 1 else 0 :=
  pdivMat_transpose A i j k l

-- Ch 3 MLP ──────────────────────────────────────────────────────────

theorem chk_pdiv_dense {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (i : Fin m) (j : Fin n) :
    pdiv (dense W b) x i j = W i j :=
  pdiv_dense W b x i j

theorem chk_pdiv_dense_W {m n : Nat} (b : Vec n) (x : Vec m) (W : Mat m n)
    (i : Fin m) (j' : Fin n) (j : Fin n) :
    pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
         (Mat.flatten W) (finProdFinEquiv (i, j')) j =
      if j = j' then x i else 0 :=
  pdiv_dense_W b x W i j' j

theorem chk_pdiv_dense_b {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m)
    (i j : Fin n) :
    pdiv (fun b' : Vec n => dense W b' x) b i j = if i = j then 1 else 0 :=
  pdiv_dense_b W b x i j

theorem chk_dense_weight_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) (j : Fin n) :
    Mat.outer x dy i j =
      ∑ k : Fin n,
        pdiv (fun v : Vec (m * n) => dense (Mat.unflatten v) b x)
             (Mat.flatten W) (finProdFinEquiv (i, j)) k * dy k :=
  dense_weight_grad_correct W b x dy i j

theorem chk_dense_bias_grad_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin n) :
    dy i =
      ∑ j : Fin n, pdiv (fun b' : Vec n => dense W b' x) b i j * dy j :=
  dense_bias_grad_correct W b x dy i

-- Ch 5 BN ───────────────────────────────────────────────────────────

theorem chk_pdiv_bnAffine (n : Nat) (γ β : ℝ) (v : Vec n) (i j : Fin n) :
    pdiv (bnAffine n γ β) v i j =
      if i = j then γ else 0 :=
  pdiv_bnAffine n γ β v i j

theorem chk_pdiv_bnCentered (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (bnCentered n) x i j =
      (if i = j then (1 : ℝ) else 0) - 1 / (n : ℝ) :=
  pdiv_bnCentered n x i j

theorem chk_pdiv_bnIstdBroadcast (n : Nat) (ε : ℝ) (hε : 0 < ε) (x : Vec n)
    (i j : Fin n) :
    pdiv (bnIstdBroadcast n ε) x i j =
      -(bnIstd n x ε)^3 * (x i - bnMean n x) / (n : ℝ) :=
  pdiv_bnIstdBroadcast n ε hε x i j

theorem chk_pdiv_bnNormalize (n : Nat) (ε : ℝ) (hε : 0 < ε)
    (x : Vec n) (i j : Fin n) :
    pdiv (bnNormalize n ε) x i j =
      bnIstd n x ε / (n : ℝ) *
        ((n : ℝ) * (if i = j then 1 else 0) - 1 - bnXhat n ε x i * bnXhat n ε x j) :=
  pdiv_bnNormalize n ε hε x i j

-- Ch 9 LayerNorm + GELU ─────────────────────────────────────────────

theorem chk_pdiv_gelu (n : Nat) (x : Vec n) (i j : Fin n) :
    pdiv (gelu n) x i j =
    if i = j then geluScalarDeriv (x i) else 0 :=
  pdiv_gelu n x i j

-- Ch 10 Attention ───────────────────────────────────────────────────

theorem chk_pdiv_softmax (c : Nat) (z : Vec c) (i j : Fin c) :
    pdiv (softmax c) z i j =
    softmax c z j * ((if i = j then 1 else 0) - softmax c z i) :=
  pdiv_softmax c z i j

theorem chk_softmaxCE_grad (c : Nat) (logits : Vec c) (label : Fin c) (j : Fin c) :
    pdiv (fun (z : Vec c) (_ : Fin 1) => crossEntropy c z label) logits j 0
    = softmax c logits j - oneHot c label j :=
  softmaxCE_grad c logits label j

theorem chk_sdpa_back_Q_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_Q n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun Q' => sdpa n d Q' K V) Q i j k l * dOut k l :=
  sdpa_back_Q_correct n d Q K V dOut i j

theorem chk_sdpa_back_K_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_K n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun K' => sdpa n d Q K' V) K i j k l * dOut k l :=
  sdpa_back_K_correct n d Q K V dOut i j

theorem chk_sdpa_back_V_correct (n d : Nat) (Q K V dOut : Mat n d)
    (i : Fin n) (j : Fin d) :
    sdpa_back_V n d Q K V dOut i j =
    ∑ k : Fin n, ∑ l : Fin d,
      pdivMat (fun V' => sdpa n d Q K V') V i j k l * dOut k l :=
  sdpa_back_V_correct n d Q K V dOut i j

-- Public correctness theorems for canonical-witness defs ────────────

theorem chk_relu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  relu_has_vjp_correct n x dy i

theorem chk_mlp_has_vjp_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀) (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp W₀ b₀ W₁ b₁ W₂ b₂).backward x dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  mlp_has_vjp_correct W₀ b₀ W₁ b₁ W₂ b₂ x dy i

theorem chk_maxPool2_has_vjp3_correct {c h w : Nat}
    (x : Tensor3 c (2*h) (2*w)) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp3 (c := c) (h := h) (w := w)).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  maxPool2_has_vjp3_correct x dy ci hi wi

theorem chk_depthwise_has_vjp3_correct {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c)
    (x : Tensor3 c h w) (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    (depthwise_has_vjp3 (h := h) (w := w) W b).backward x dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (depthwiseConv2d W b : Tensor3 c h w → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  depthwise_has_vjp3_correct W b x dy ci hi wi

theorem chk_residual_has_vjp_correct {n : Nat}
    (f : Vec n → Vec n) (hf_diff : Differentiable ℝ f) (hf : HasVJP f)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (residual_has_vjp f hf_diff hf).backward x dy i =
    ∑ j : Fin n, pdiv (residual f) x i j * dy j :=
  residual_has_vjp_correct f hf_diff hf x dy i

theorem chk_residualProj_has_vjp_correct {m n : Nat}
    (proj f : Vec m → Vec n)
    (hproj_diff : Differentiable ℝ proj) (hf_diff : Differentiable ℝ f)
    (hproj : HasVJP proj) (hf : HasVJP f)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (residualProj_has_vjp proj f hproj_diff hf_diff hproj hf).backward x dy i =
    ∑ j : Fin n, pdiv (residualProj proj f) x i j * dy j :=
  residualProj_has_vjp_correct proj f hproj_diff hf_diff hproj hf x dy i

theorem chk_seBlock_has_vjp_correct {n : Nat}
    (gate : Vec n → Vec n) (hg_diff : Differentiable ℝ gate) (hg : HasVJP gate)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (seBlock_has_vjp gate hg_diff hg).backward x dy i =
    ∑ j : Fin n, pdiv (seBlock gate) x i j * dy j :=
  seBlock_has_vjp_correct gate hg_diff hg x dy i

theorem chk_gelu_has_vjp_correct (n : Nat) (x : Vec n) (dy : Vec n) (i : Fin n) :
    (gelu_has_vjp n).backward x dy i =
    ∑ j : Fin n, pdiv (gelu n) x i j * dy j :=
  gelu_has_vjp_correct n x dy i

theorem chk_layerNorm_has_vjp_correct (n : Nat) (ε γ β : ℝ) (hε : 0 < ε)
    (x : Vec n) (dy : Vec n) (i : Fin n) :
    (layerNorm_has_vjp n ε γ β hε).backward x dy i =
    ∑ j : Fin n, pdiv (layerNormForward n ε γ β) x i j * dy j :=
  layerNorm_has_vjp_correct n ε γ β hε x dy i

theorem chk_mhsa_has_vjp_mat_correct (N heads d_head : Nat)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (mhsa_has_vjp_mat N heads d_head Wq Wk Wv Wo bq bk bv bo).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (mhsa_layer N heads d_head Wq Wk Wv Wo bq bk bv bo)
              X i j k l * dY k l :=
  mhsa_has_vjp_mat_correct N heads d_head Wq Wk Wv Wo bq bk bv bo X dY i j

theorem chk_transformerBlock_has_vjp_mat_correct
    (N heads d_head mlpDim : Nat)
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (X : Mat N (heads * d_head)) (dY : Mat N (heads * d_head))
    (i : Fin N) (j : Fin (heads * d_head)) :
    (transformerBlock_has_vjp_mat N heads d_head mlpDim ε γ1 β1 hε
        Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2).backward X dY i j =
    ∑ k : Fin N, ∑ l : Fin (heads * d_head),
      pdivMat (transformerBlock N heads d_head mlpDim ε γ1 β1
                 Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2)
              X i j k l * dY k l :=
  transformerBlock_has_vjp_mat_correct N heads d_head mlpDim ε γ1 β1 hε
    Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 X dY i j

theorem chk_vit_full_has_vjp_correct
    (ic H W patchSize N mlpDim heads d_head kBlocks nClasses : Nat)
    (W_conv : Kernel4 (heads * d_head) ic patchSize patchSize)
    (b_conv : Vec (heads * d_head))
    (cls_token : Vec (heads * d_head))
    (pos_embed : Mat (N + 1) (heads * d_head))
    (ε γ1 β1 : ℝ) (hε : 0 < ε)
    (Wq Wk Wv Wo : Mat (heads * d_head) (heads * d_head))
    (bq bk bv bo : Vec (heads * d_head))
    (γ2 β2 : ℝ)
    (Wfc1 : Mat (heads * d_head) mlpDim) (bfc1 : Vec mlpDim)
    (Wfc2 : Mat mlpDim (heads * d_head)) (bfc2 : Vec (heads * d_head))
    (γF βF : ℝ)
    (Wcls : Mat (heads * d_head) nClasses) (bcls : Vec nClasses)
    (x : Vec (ic * H * W)) (dy : Vec nClasses) (i : Fin (ic * H * W)) :
    (vit_full_has_vjp ic H W patchSize N mlpDim heads d_head kBlocks nClasses
        W_conv b_conv cls_token pos_embed ε γ1 β1 hε
        Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls).backward x dy i =
    ∑ j : Fin nClasses,
      pdiv (vit_full ic H W patchSize N mlpDim heads d_head kBlocks nClasses
              W_conv b_conv cls_token pos_embed ε γ1 β1
              Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls)
           x i j * dy j :=
  vit_full_has_vjp_correct ic H W patchSize N mlpDim heads d_head kBlocks nClasses
    W_conv b_conv cls_token pos_embed ε γ1 β1 hε
    Wq Wk Wv Wo bq bk bv bo γ2 β2 Wfc1 bfc1 Wfc2 bfc2 γF βF Wcls bcls x dy i

/-- **`cnn_has_vjp_at` contract**: the pointwise (smooth-input) variant
for the full ResNet-style CNN — backward equals the `pdiv`-contracted
Jacobian. CNN analogue of `vit_full_has_vjp_correct`; the underlying
`.correct` chains `vjp_comp_at` through stem → maxpool → residual
blocks → global-avg-pool → dense under the smoothness hypotheses. -/
theorem chk_cnn_has_vjp_at_correct
    {ic c oc h w kHs kWs kH₁ kW₁ kH₂ kW₂ kH₁' kW₁' kH₂' kW₂' kHp kWp nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (W₁ : Kernel4 c c kH₁ kW₁) (b₁ : Vec c) (W₂ : Kernel4 c c kH₂ kW₂) (b₂ : Vec c)
    (e₁ g₁ bb₁ e₂ g₂ bb₂ : ℝ) (he₁ : 0 < e₁) (he₂ : 0 < e₂)
    (W₁' : Kernel4 oc c kH₁' kW₁') (b₁' : Vec oc) (W₂' : Kernel4 oc oc kH₂' kW₂') (b₂' : Vec oc)
    (Wp : Kernel4 oc c kHp kWp) (bp : Vec oc)
    (f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip : ℝ) (hf₁ : 0 < f₁) (hf₂ : 0 < f₂) (hfp : 0 < fp)
    (Wd : Mat oc nClasses) (bd : Vec nClasses)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (x : Vec (ic * (2*h) * (2*w)))
    (h_stem : ∀ k, bnForward (c * (2*h) * (2*w)) εs γs βs (flatConv Ws bs x) k ≠ 0)
    (h_mp : MaxPool2Smooth (Tensor3.unflatten
              (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x) : Tensor3 c (2*h) (2*w)))
    (h_rb1 : ∀ k, bnForward (c * h * w) f₁ hh₁ i₁
        (flatConv W₁ b₁
          (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0)
    (h_rb1o : ∀ k,
        ((bnForward (c * h * w) f₂ hh₂ i₂ ∘ flatConv W₂ b₂) ∘
          (relu (c * h * w) ∘ bnForward (c * h * w) f₁ hh₁ i₁ ∘ flatConv W₁ b₁))
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k
          + (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x)) k ≠ 0)
    (h_rb2 : ∀ k, bnForward (oc * h * w) e₁ g₁ bb₁
        (flatConv (h := h) (w := w) W₁' b₁'
          ((rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) : Vec (c*h*w))) k ≠ 0)
    (h_rb2o : ∀ k,
        ((bnForward (oc * h * w) fp hhp ip ∘ flatConv (h := h) (w := w) Wp bp)
          (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
            (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k)
        + ((bnForward (oc * h * w) e₂ g₂ bb₂ ∘ flatConv (h := h) (w := w) W₂' b₂') ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) e₁ g₁ bb₁ ∘ flatConv (h := h) (w := w) W₁' b₁'))
            (rblk (h := h) (w := w) W₁ b₁ W₂ b₂ f₁ hh₁ i₁ f₂ hh₂ i₂
              (maxPoolFlat c h w (cbr (h := 2*h) (w := 2*w) Ws bs εs γs βs x))) k ≠ 0)
    (dy : Vec nClasses) (i : Fin (ic * (2*h) * (2*w))) :
    (cnn_has_vjp_at Ws bs εs γs βs hεs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂ he₁ he₂
        W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip hf₁ hf₂ hfp Wd bd
        hc hh hw x h_stem h_mp h_rb1 h_rb1o h_rb2 h_rb2o).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (cnnForward Ws bs εs γs βs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂
                W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip Wd bd)
             x i j * dy j :=
  cnn_has_vjp_at_correct Ws bs εs γs βs hεs W₁ b₁ W₂ b₂ e₁ g₁ bb₁ e₂ g₂ bb₂ he₁ he₂ W₁' b₁' W₂' b₂' Wp bp f₁ hh₁ i₁ f₂ hh₂ i₂ fp hhp ip hf₁ hf₂ hfp Wd bd hc hh hw x h_stem h_mp h_rb1 h_rb1o h_rb2 h_rb2o dy i

theorem chk_conv2d_has_vjp3_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Tensor3 ic h w) (dy : Tensor3 oc h w)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w) :
    (conv2d_has_vjp3 W b).backward x dy ci hi wi =
      ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        pdiv3 (conv2d W b) x ci hi wi co ho wo * dy co ho wo :=
  conv2d_has_vjp3_correct W b x dy ci hi wi

theorem chk_globalAvgPoolFlat_has_vjp_correct (c h w : Nat)
    (x : Vec (c*h*w)) (dy : Vec c) (i : Fin (c*h*w)) :
    (globalAvgPoolFlat_has_vjp c h w).backward x dy i =
      ∑ j : Fin c, pdiv (globalAvgPoolFlat c h w) x i j * dy j :=
  globalAvgPoolFlat_has_vjp_correct c h w x dy i

theorem chk_mobilenetv2_has_vjp_at_correct
    {ic c mid₁ oc mid₂ h w kHs kWs
     kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
     kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 mid₁ c kHe₁ kWe₁) (be₁ : Vec mid₁) (e₁ ge₁ be1 : ℝ) (he₁ : 0 < e₁)
    (Wd₁ : DepthwiseKernel mid₁ kHd₁ kWd₁) (bd₁ : Vec mid₁) (d₁ gd₁ bd1 : ℝ) (hd₁ : 0 < d₁)
    (Wp₁ : Kernel4 c mid₁ kHp₁ kWp₁) (bp₁ : Vec c) (p₁ gp₁ bp1 : ℝ) (hp₁ : 0 < p₁)
    (We₂ : Kernel4 mid₂ c kHe₂ kWe₂) (be₂ : Vec mid₂) (e₂ ge₂ be2 : ℝ) (he₂ : 0 < e₂)
    (Wd₂ : DepthwiseKernel mid₂ kHd₂ kWd₂) (bd₂ : Vec mid₂) (d₂ gd₂ bd2 : ℝ) (hd₂ : 0 < d₂)
    (Wp₂ : Kernel4 oc mid₂ kHp₂ kWp₂) (bp₂ : Vec oc) (p₂ gp₂ bp2 : ℝ) (hp₂ : 0 < p₂)
    (Wh : Mat oc nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w))
    (h_stem : ∀ k, (bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 0 ∧
                     bnForward (c * h * w) εs γs βs (flatConv Ws bs x) k ≠ 6))
    (h_b1e : ∀ k, (bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) e₁ ge₁ be1
        (flatConv We₁ be₁
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)) k ≠ 6))
    (h_b1d : ∀ k, (bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₁ * h * w) d₁ gd₁ bd1
        (depthwiseFlat Wd₁ bd₁ (ivExpand (h := h) (w := w) We₁ be₁ e₁ ge₁ be1
          ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    (h_b2e : ∀ k, (bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) e₂ ge₂ be2
        (flatConv We₂ be₂
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x))) k ≠ 6))
    (h_b2d : ∀ k, (bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 0 ∧
                   bnForward (mid₂ * h * w) d₂ gd₂ bd2
        (depthwiseFlat Wd₂ bd₂ (ivExpand (h := h) (w := w) We₂ be₂ e₂ ge₂ be2
          ((residual (invresBody (h := h) (w := w) We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1))
            ((relu6 (c*h*w) ∘ bnForward (c*h*w) εs γs βs ∘ flatConv Ws bs) x)))) k ≠ 6))
    (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (mobilenetv2_has_vjp_at Ws bs εs γs βs hεs
        We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
        We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂ Wh bh
        x h_stem h_b1e h_b1d h_b2e h_b2d).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (mobilenetv2Forward Ws bs εs γs βs
                We₁ be₁ e₁ ge₁ be1 Wd₁ bd₁ d₁ gd₁ bd1 Wp₁ bp₁ p₁ gp₁ bp1
                We₂ be₂ e₂ ge₂ be2 Wd₂ bd₂ d₂ gd₂ bd2 Wp₂ bp₂ p₂ gp₂ bp2 Wh bh)
             x i j * dy j :=
  mobilenetv2_has_vjp_at_correct Ws bs εs γs βs hεs
    We₁ be₁ e₁ ge₁ be1 he₁ Wd₁ bd₁ d₁ gd₁ bd1 hd₁ Wp₁ bp₁ p₁ gp₁ bp1 hp₁
    We₂ be₂ e₂ ge₂ be2 he₂ Wd₂ bd₂ d₂ gd₂ bd2 hd₂ Wp₂ bp₂ p₂ gp₂ bp2 hp₂ Wh bh
    x h_stem h_b1e h_b1d h_b2e h_b2d dy i

theorem chk_convnext_has_vjp_at_correct
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (convnext_has_vjp_at Wst bst εst γst βst hεst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd hεhd Wd bd x).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (convNextForward Wst bst εst γst βst
          Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
          Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
          εhd γhd βhd Wd bd) x i j * dy j :=
  convnext_has_vjp_at_correct Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd x dy i

theorem chk_convnext_has_vjp_correct
    {ic c cExp h w kH kW nClasses : Nat}
    (Wst : Kernel4 c ic 1 1) (bst : Vec c) (εst γst βst : ℝ) (hεst : 0 < εst)
    (Wdw₁ : DepthwiseKernel c kH kW) (bdw₁ : Vec c) (εn₁ γn₁ βn₁ : ℝ) (hεn₁ : 0 < εn₁)
    (Wex₁ : Kernel4 cExp c 1 1) (bex₁ : Vec cExp)
    (Wpr₁ : Kernel4 c cExp 1 1) (bpr₁ : Vec c) (γls₁ : Vec (c * h * w))
    (Wdw₂ : DepthwiseKernel c kH kW) (bdw₂ : Vec c) (εn₂ γn₂ βn₂ : ℝ) (hεn₂ : 0 < εn₂)
    (Wex₂ : Kernel4 cExp c 1 1) (bex₂ : Vec cExp)
    (Wpr₂ : Kernel4 c cExp 1 1) (bpr₂ : Vec c) (γls₂ : Vec (c * h * w))
    (εhd γhd βhd : ℝ) (hεhd : 0 < εhd)
    (Wd : Mat c nClasses) (bd : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (convnext_has_vjp Wst bst εst γst βst hεst
      Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
      Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
      εhd γhd βhd hεhd Wd bd).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (convNextForward Wst bst εst γst βst
          Wdw₁ bdw₁ εn₁ γn₁ βn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
          Wdw₂ bdw₂ εn₂ γn₂ βn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
          εhd γhd βhd Wd bd) x i j * dy j :=
  convnext_has_vjp_correct Wst bst εst γst βst hεst
    Wdw₁ bdw₁ εn₁ γn₁ βn₁ hεn₁ Wex₁ bex₁ Wpr₁ bpr₁ γls₁
    Wdw₂ bdw₂ εn₂ γn₂ βn₂ hεn₂ Wex₂ bex₂ Wpr₂ bpr₂ γls₂
    εhd γhd βhd hεhd Wd bd x dy i

theorem chk_efficientnet_has_vjp_at_correct
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (efficientnet_has_vjp_at Ws bs εs γs βs hεs
        We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
        We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
        Wh bh x).backward dy i =
      ∑ j : Fin nClasses,
        pdiv (efficientnetForward Ws bs εs γs βs
                We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
                We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
                Wh bh)
             x i j * dy j :=
  efficientnet_has_vjp_at_correct Ws bs εs γs βs hεs
    We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
    We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
    Wh bh x dy i

theorem chk_efficientnet_has_vjp_correct
    {ic c cmid₁ cout cmid₂ h w kHs kWs kHe₁ kWe₁ kHd₁ kWd₁ kHp₁ kWp₁
      kHe₂ kWe₂ kHd₂ kWd₂ kHp₂ kWp₂ r₁ r₂ nClasses : Nat}
    (Ws : Kernel4 c ic kHs kWs) (bs : Vec c) (εs γs βs : ℝ) (hεs : 0 < εs)
    (We₁ : Kernel4 cmid₁ c kHe₁ kWe₁) (be₁ : Vec cmid₁) (εe₁ γe₁ βe₁ : ℝ) (hεe₁ : 0 < εe₁)
    (Wd₁ : DepthwiseKernel cmid₁ kHd₁ kWd₁) (bd₁ : Vec cmid₁) (εd₁ γd₁ βd₁ : ℝ) (hεd₁ : 0 < εd₁)
    (Ws₁₁ : Mat cmid₁ r₁) (bs₁₁ : Vec r₁) (Ws₁₂ : Mat r₁ cmid₁) (bs₁₂ : Vec cmid₁)
    (Wp₁ : Kernel4 c cmid₁ kHp₁ kWp₁) (bp₁ : Vec c) (εp₁ γp₁ βp₁ : ℝ) (hεp₁ : 0 < εp₁)
    (We₂ : Kernel4 cmid₂ c kHe₂ kWe₂) (be₂ : Vec cmid₂) (εe₂ γe₂ βe₂ : ℝ) (hεe₂ : 0 < εe₂)
    (Wd₂ : DepthwiseKernel cmid₂ kHd₂ kWd₂) (bd₂ : Vec cmid₂) (εd₂ γd₂ βd₂ : ℝ) (hεd₂ : 0 < εd₂)
    (Ws₂₁ : Mat cmid₂ r₂) (bs₂₁ : Vec r₂) (Ws₂₂ : Mat r₂ cmid₂) (bs₂₂ : Vec cmid₂)
    (Wp₂ : Kernel4 cout cmid₂ kHp₂ kWp₂) (bp₂ : Vec cout) (εp₂ γp₂ βp₂ : ℝ) (hεp₂ : 0 < εp₂)
    (Wh : Mat cout nClasses) (bh : Vec nClasses)
    (x : Vec (ic * h * w)) (dy : Vec nClasses) (i : Fin (ic * h * w)) :
    (efficientnet_has_vjp (h := h) (w := w) Ws bs εs γs βs hεs
        We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
        We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
        Wh bh).backward x dy i =
      ∑ j : Fin nClasses,
        pdiv (efficientnetForward Ws bs εs γs βs
                We₁ be₁ εe₁ γe₁ βe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁
                We₂ be₂ εe₂ γe₂ βe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂
                Wh bh)
             x i j * dy j :=
  efficientnet_has_vjp_correct (h := h) (w := w) Ws bs εs γs βs hεs
    We₁ be₁ εe₁ γe₁ βe₁ hεe₁ Wd₁ bd₁ εd₁ γd₁ βd₁ hεd₁ Ws₁₁ bs₁₁ Ws₁₂ bs₁₂ Wp₁ bp₁ εp₁ γp₁ βp₁ hεp₁
    We₂ be₂ εe₂ γe₂ βe₂ hεe₂ Wd₂ bd₂ εd₂ γd₂ βd₂ hεd₂ Ws₂₁ bs₂₁ Ws₂₂ bs₂₂ Wp₂ bp₂ εp₂ γp₂ βp₂ hεp₂
    Wh bh x dy i

-- Pointwise (`_at`) variants ────────────────────────────────────────

theorem chk_relu_has_vjp_at_correct (n : Nat) (x : Vec n)
    (h_smooth : ∀ k, x k ≠ 0) (dy : Vec n) (i : Fin n) :
    (relu_has_vjp_at n x h_smooth).backward dy i =
    ∑ j : Fin n, pdiv (relu n) x i j * dy j :=
  relu_has_vjp_at_correct n x h_smooth dy i

theorem chk_mlp_has_vjp_at_correct {d₀ d₁ d₂ d₃ : Nat}
    (W₀ : Mat d₀ d₁) (b₀ : Vec d₁)
    (W₁ : Mat d₁ d₂) (b₁ : Vec d₂)
    (W₂ : Mat d₂ d₃) (b₂ : Vec d₃)
    (x : Vec d₀)
    (h_smooth_0 : ∀ k, dense W₀ b₀ x k ≠ 0)
    (h_smooth_1 : ∀ k, dense W₁ b₁ (relu d₁ (dense W₀ b₀ x)) k ≠ 0)
    (dy : Vec d₃) (i : Fin d₀) :
    (mlp_has_vjp_at W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1).backward dy i =
    ∑ j : Fin d₃, pdiv (mlpForward W₀ b₀ W₁ b₁ W₂ b₂) x i j * dy j :=
  mlp_has_vjp_at_correct W₀ b₀ W₁ b₁ W₂ b₂ x h_smooth_0 h_smooth_1 dy i

theorem chk_maxPool2_has_vjp_at3_correct {c h w : Nat}
    (x : Tensor3 c (2 * h) (2 * w)) (h_smooth : MaxPool2Smooth x)
    (dy : Tensor3 c h w)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    (maxPool2_has_vjp_at3 x h_smooth).backward dy ci hi wi =
    ∑ co : Fin c, ∑ ho : Fin h, ∑ wo : Fin w,
      pdiv3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
            x ci hi wi co ho wo * dy co ho wo :=
  maxPool2_has_vjp_at3_correct x h_smooth dy ci hi wi

/-- **`mnistLinear_has_vjp_correct` contract**: whole-model VJP for the
Chapter-2 linear classifier — the degenerate simplest case of the
per-architecture capstones (a linear classifier is one dense layer). -/
theorem chk_mnistLinear_has_vjp_correct {m n : Nat} (W : Mat m n) (b : Vec n)
    (x : Vec m) (dy : Vec n) (i : Fin m) :
    (dense_has_vjp W b).backward x dy i =
      ∑ j : Fin n, pdiv (mnistLinear W b) x i j * dy j :=
  mnistLinear_has_vjp_correct W b x dy i
