# Citation list for the attribution pass

This is the list of works the attribution fixes will cite, gathered from the `### attribution`
sections of `slice_{A,C,F,G,N1,N2,P,X}.md` (including slice_X's census and X-att-1..6), plus a grep
of `LeanMlir/`, `Bestiary/`, `demos/`, `apps/`, `ffi/`, `jax/*.lean`, `README.md` and `TRUST.md`
for methods named without a source (marked **new** in the Finding IDs column). This first pass
records one web link per work. Every link was checked on 2026-09-30. arXiv ids were checked through
the arXiv API (`export.arxiv.org/api/query?id_list=…`), with the returned title compared against the
work. DOIs were checked by doi.org content negotiation (CSL-JSON title, year, first author). All other
URLs were fetched and their page title checked, and the RMSProp slides were checked by extracting
their text. The book has no `\cite` keys and no bibliography: it credits inline, as author-year
followed by an `\href` to arXiv or a DOI. The **Book** column therefore gives the `content.tex` line
of that inline credit, and "—" means the book does not credit the work. **In Lean?** records whether any
file under the searched roots credits the work (author and year, or the id): "no", "name only", or
the file that already credits it.

## Architectures & ops

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| He, Zhang, Ren, Sun 2016, *Deep Residual Learning for Image Recognition* | https://arxiv.org/abs/1512.03385 | `LeanMlir/Proofs/Architectures/Residual.lean`, `Proofs/Nets/ResNet/ResNet34FullB.lean`, `ResNet50FullB.lean`, `LeanMlir/Verified/NetsCore.lean`, `LeanMlir/ReferenceNets.lean` | 5025, 13619 | partial: inline "He et al." in `ResNet34FullB`/`ResNet50FullB`/`MaxPool3s2`/`StridedConv` decl docstrings, `Bestiary/ResNet.lean` full; no module docstring | A-attr-2, N1-attr-1, X-att-3 |
| Ioffe & Szegedy 2015, *Batch Normalization* | https://arxiv.org/abs/1502.03167 | `LeanMlir/Proofs/Architectures/BatchNorm.lean` | 3770 (author-year, no link) | no | A-attr-2 |
| Ba, Kiros, Hinton 2016, *Layer Normalization* | https://arxiv.org/abs/1607.06450 | `LeanMlir/Proofs/Architectures/LayerNorm.lean`, `Proofs/Nets/ViT/ViTVecLN.lean` | 9757 | no | A-attr-2, N2-attr-3 |
| Hendrycks & Gimpel 2016, *Gaussian Error Linear Units (GELUs)* (incl. tanh approximation) | https://arxiv.org/abs/1606.08415 | `LeanMlir/Proofs/Architectures/LayerNorm.lean` | 9770 | no | A-attr-2, N2-attr-3 |
| Ramachandran, Zoph, Le 2017, *Searching for Activation Functions* (Swish) | https://arxiv.org/abs/1710.05941 | `LeanMlir/Proofs/Architectures/LayerNorm.lean`, `LeanMlir/Types.lean` (activation enum) | — | no | A-attr-2 |
| Howard et al. 2019, *Searching for MobileNetV3* (h-swish / h-sigmoid) | https://arxiv.org/abs/1905.02244 | `LeanMlir/MlirCodegen.lean` (h-swish emit, :604), `LeanMlir/VjpOracleNets.lean` (`mbConvV3`) | — | name only ("MobileNet V3") | **new** |
| Touvron et al. 2021, *Going deeper with Image Transformers* (CaiT, layer-scale) | https://arxiv.org/abs/2103.17239 | `LeanMlir/Proofs/Architectures/LayerNorm.lean` (`layerScale`) | — | no | A-attr-2 |
| Vaswani et al. 2017, *Attention Is All You Need* | https://arxiv.org/abs/1706.03762 | `LeanMlir/Proofs/Architectures/Attention.lean`, `Proofs/Nets/ViT/ViTMultiHead.lean` | 8398, 8549 (author-year) | partial: `Types.lean`, `Ddpm.lean` (sinusoidal embedding) | A-attr-2, N2-attr-3 |
| Dosovitskiy et al. 2021, *An Image is Worth 16x16 Words* (ViT) | https://arxiv.org/abs/2010.11929 | `LeanMlir/Proofs/Architectures/Attention.lean`, `Proofs/Nets/ViT/ViTDepthK.lean`, `Proofs/Codegen/ViTRender.lean`, `Verified/NetsCore.lean` | 9599 | yes: `apps/imagenette/MainViTVerified.lean` only | A-attr-2, G-attr-1, N2-attr-3, X-att-3 |
| Touvron et al. 2021, *Training data-efficient image transformers & distillation through attention* (DeiT) | https://arxiv.org/abs/2012.12877 | `Proofs/Nets/ViT/ViTDepthK.lean` (DeiT-Ti config), `Verified/NetsCore.lean` | 12187 | yes: `apps/ablation/MainAblation.lean` only | N2-attr-3, X-att-3 |
| Su et al. 2021, *RoFormer: Enhanced Transformer with Rotary Position Embedding* (RoPE) | https://arxiv.org/abs/2104.09864 | `LeanMlir/MlirCodegen.lean` (RoPE emit, :1351) | — | name only | **new** |
| Hu, Shen, Sun 2018, *Squeeze-and-Excitation Networks* | https://arxiv.org/abs/1709.01507 | `LeanMlir/Proofs/Architectures/SE.lean`, `Proofs/Nets/EfficientNet/EfficientNetFullB0.lean` | 8390 | no | A-attr-2, N2-attr-2 |
| Chollet 2017, *Xception: Deep Learning with Depthwise Separable Convolutions* | https://arxiv.org/abs/1610.02357 | `LeanMlir/Proofs/Architectures/Depthwise.lean` | 13866 | name only (`Bestiary/Xception.lean` full) | A-attr-2 |
| Howard et al. 2017, *MobileNets* | https://arxiv.org/abs/1704.04861 | `LeanMlir/Proofs/Architectures/Depthwise.lean` | 6698 | no | A-attr-2 |
| Sandler et al. 2018, *MobileNetV2: Inverted Residuals and Linear Bottlenecks* | https://arxiv.org/abs/1801.04381 | `Proofs/Nets/MobileNet/MobileNetV2.lean`, `MobileNetV2FullB.lean`, `Verified/NetsCore.lean`, `Bestiary/DeepLabV3Plus.lean` (backbone) | 6700 | yes: `Proofs/Codegen/MobileNetV2RenderB.lean`, `jax/MainMobilenetV2Imagenet.lean` | N2-attr-1, X-att-3, X-att-6 |
| Qin et al. 2024, *MobileNetV4: Universal Models for the Mobile Ecosystem* | https://arxiv.org/abs/2404.10518 | `Proofs/Nets/MobileNet/MobileNetV4Spec.lean`, `MobileNetV4FullB.lean`, `Proofs/Codegen/MobileNetV4RenderB.lean`, `Verified/NetsCore.lean` | 8055 | no (timm credited in 11 files) | G-attr-1, N2-attr-1, X-att-3 |
| Tan & Le 2019, *EfficientNet: Rethinking Model Scaling for CNNs* | https://arxiv.org/abs/1905.11946 | `Proofs/Nets/EfficientNet/EfficientNet.lean`, `EfficientNetFullB0.lean`, `Proofs/Codegen/EfficientNetRender/Basic.lean`, `Verified/NetsCore.lean` | 8562 | yes: `apps/imagenette/MainEfficientNetVerified.lean` only | G-attr-1, N2-attr-2, X-att-3 |
| Liu et al. 2022, *A ConvNet for the 2020s* (ConvNeXt) | https://arxiv.org/abs/2201.03545 | `Proofs/Nets/ConvNeXt/ConvNeXt.lean`, `ConvNeXtFullT.lean`, `Proofs/Architectures/ChannelLN.lean`, `Proofs/Codegen/ConvNeXtRender.lean`, `Verified/NetsCore.lean` | 9608 | yes: `ReferenceNets.lean`, `Types.lean`, `apps/imagenette/MainConvNeXtVerified.lean` | A-attr-2, G-attr-1, N1-attr-1, X-att-3 |
| Ronneberger, Fischer, Brox 2015, *U-Net* | https://arxiv.org/abs/1505.04597 | `LeanMlir/ReferenceNets.lean` (`unetBrats`), `demos/MainUnetBratsTrain.lean`, `LeanMlir/MlirCodegen.lean` | 14947 | yes: `Bestiary/UNet.lean` (author-year only), `Bestiary/Diffusion.lean`, `Pix2Pix.lean` | X-att-3, X-att-5, X-att-6 |
| He et al. 2015, *Delving Deep into Rectifiers* (He init) | https://arxiv.org/abs/1502.01852 | `LeanMlir/F32Array.lean` (`heInit`), `LeanMlir/ParamLayouts.lean` | — | name only ("He init") | **new** |
| Glorot & Bengio 2010, *Understanding the difficulty of training deep feedforward neural networks* | https://proceedings.mlr.press/v9/glorot10a.html | `LeanMlir/ParamLayouts.lean` (dense Glorot), `LeanMlir/Types.lean` (Xavier-uniform) | — | name only | **new** |
| Milletari, Navab, Ahmadi 2016, *V-Net* (soft Dice loss) | https://arxiv.org/abs/1606.04797 | `LeanMlir/Types.lean` (`perPixelDice`, :359) | — | no | **new** |

## Optimizers & regularisers

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Kingma & Ba 2015, *Adam: A Method for Stochastic Optimization* | https://arxiv.org/abs/1412.6980 | `LeanMlir/Proofs/Training/Optim/AdamStep.lean` | 5747 | no (only `Bestiary/VAE.lean`, for VAE) | A-attr-1 |
| Loshchilov & Hutter 2019, *Decoupled Weight Decay Regularization* (AdamW) | https://arxiv.org/abs/1711.05101 | `Proofs/Training/Optim/AdamStep.lean`, `LeanMlir/Verified/Train.lean` | 5782 | yes: `jax/Jax/Codegen.lean` only | A-attr-1 |
| Reddi, Kale, Kumar 2018, *On the Convergence of Adam and Beyond* | https://arxiv.org/abs/1904.09237 | (already cited) | — | yes: `AdamStep.lean:10` | A-attr-1 |
| Tieleman & Hinton 2012, *Neural Networks for Machine Learning*, lecture 6e (RMSProp) | https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf | `Proofs/Training/Optim/RmsPropStep.lean` | — (RMSProp named, no source) | no | A-attr-1 |
| Polyak 1964, *Some methods of speeding up the convergence of iteration methods* (heavy ball) | https://doi.org/10.1016/0041-5553(64)90137-5 | `Proofs/Training/Optim/SgdMomentumStep.lean` | — | no | A-attr-1 |
| Nesterov 1983, *A method of solving a convex programming problem with convergence rate O(1/k²)* | https://www.mathnet.ru/eng/dan46009 | `Proofs/Training/Optim/SgdMomentumStep.lean` | — (4699 names "Nesterov") | no | A-attr-1 |
| Sutskever, Martens, Dahl, Hinton 2013, *On the importance of initialization and momentum in deep learning* | https://proceedings.mlr.press/v28/sutskever13.html | `Proofs/Training/Optim/SgdMomentumStep.lean` (the form implemented) | — | no | A-attr-1 |
| Pascanu, Mikolov, Bengio 2013, *On the difficulty of training Recurrent Neural Networks* (global-norm clipping) | https://arxiv.org/abs/1211.5063 | `Proofs/Training/Optim/GradClip.lean` | — | no | A-attr-1 |
| Nesterov 2004, *Introductory Lectures on Convex Optimization* (Lemma 1.2.3, descent lemma) | https://doi.org/10.1007/978-1-4419-8853-9 | `Proofs/Training/SgdDescent/Basic.lean` | — | no | A-attr-1 |
| You et al. 2020, *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes* (LAMB) | https://arxiv.org/abs/1904.00962 | `Proofs/Nets/ResNet/ResNet50StepTieB.lean` | 6581 area (author-year) | yes: `Proofs/Training/Optim/Lamb.lean:7`, `Types.lean` | N1-attr-1 |
| Huang et al. 2016, *Deep Networks with Stochastic Depth* | https://arxiv.org/abs/1603.09382 | `Proofs/Training/DropPath.lean`, `Proofs/Nets/EfficientNet/EfficientNetFullB0Drop.lean`, `Verified/NetsCore.lean` | — | yes: `Types.lean:705`, `apps/ablation/MainAblation.lean`, `jax/Jax/Codegen.lean` | A-attr-1, N2-attr-2, X-att-3 |
| Larsson, Maire, Shakhnarovich 2017, *FractalNet* ("drop-path") | https://arxiv.org/abs/1605.07648 | `Proofs/Training/DropPath.lean` | — | no | A-attr-1 |
| Srivastava et al. 2014, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* | https://jmlr.org/papers/v15/srivastava14a.html | `Proofs/Training/DropPath.lean`, `Proofs/Nets/EfficientNet/EfficientNetFullB0Drop.lean` | — | no | A-attr-1, N2-attr-2 |
| Izmailov et al. 2018, *Averaging Weights Leads to Wider Optima* (SWA) | https://arxiv.org/abs/1803.05407 | `LeanMlir/Types.lean` (`useSWA`, :738) | — | yes: `ffi/f32_helpers.c` only | **new** |

## Losses & training recipes

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Szegedy et al. 2016, *Rethinking the Inception Architecture for Computer Vision* (§7 label smoothing) | https://arxiv.org/abs/1512.00567 | `LeanMlir/Proofs/Foundation/SmoothedBatchLoss.lean`, `SmoothedLossCot.lean`, `LeanMlir/Types.lean` | 5792 (as Inception-v3, not for label smoothing) | no (only `Bestiary/Inception.lean`, for v3) | P-A-1 |
| Wightman, Touvron, Jégou 2021, *ResNet strikes back* (RSB A3: BCE, LAMB recipe) | https://arxiv.org/abs/2110.00476 | `Proofs/Nets/ResNet/ResNet50StepTieB.lean`, `Bestiary/ResNet.lean` | 13671, 17047 | yes: `jax/MainResnet50Imagenet.lean` | N1-attr-1, X-att-6 |
| Zhang et al. 2018, *mixup: Beyond Empirical Risk Minimization* | https://arxiv.org/abs/1710.09412 | `LeanMlir/Types.lean` (augmentation flags) | 10769 | yes: `F32Array.lean`, `ffi/f32_helpers.c` | X census |
| Yun et al. 2019, *CutMix* | https://arxiv.org/abs/1905.04899 | `LeanMlir/Types.lean` | 10770 | yes: `F32Array.lean`, `ffi/f32_helpers.c` | X census |
| Cubuk et al. 2020, *RandAugment* | https://arxiv.org/abs/1909.13719 | (already cited in `Types.lean`) | 10774 | yes: `Types.lean`, `F32Array.lean`, `ffi` | X census |
| Cubuk et al. 2019, *AutoAugment: Learning Augmentation Policies from Data* | https://arxiv.org/abs/1805.09501 | (already cited, `Types.lean:679`) | — | yes (author-year) | — |
| Zhong et al. 2020, *Random Erasing Data Augmentation* | https://arxiv.org/abs/1708.04896 | `LeanMlir/Types.lean` | 10772 | yes: `F32Array.lean`, `ffi` | **new** (Types.lean) |
| Loshchilov & Hutter 2017, *SGDR* (cosine schedule) | https://arxiv.org/abs/1608.03983 | `LeanMlir/Types.lean` (`cosineDecay`, :524) | 5755 | no | **new** |
| Goyal et al. 2017, *Accurate, Large Minibatch SGD* (linear warmup / lr scaling) | https://arxiv.org/abs/1706.02677 | `LeanMlir/Types.lean` (`warmupEpochs`, :525) | 5770 | no | **new** |
| Lin et al. 2017, *Focal Loss for Dense Object Detection* (focal CE, prior-bias init) | https://arxiv.org/abs/1708.02002 | (already credited) | 5628 area, 13284 | yes: `Types.lean:401`, `SpecHelpers.lean:94` (author-year) | — |
| Zheng et al. 2020, *Distance-IoU Loss* | https://arxiv.org/abs/1911.08287 | `LeanMlir/Types.lean` (:536), `LeanMlir/MlirCodegen.lean`, `demos/probes/MainDiouLossProbe.lean` | — | no | X-att-5 |

## Certification & robustness

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Gowal et al. 2018, *On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models* | https://arxiv.org/abs/1810.12715 | `LeanMlir/Proofs/Certificates/IntervalBound.lean`, `Proofs/Foundation/IntervalBoundConv.lean`, `IntervalBoundConvQ.lean`, `formalization.yaml` references | — | no | C-att-1, F-at-3 |
| Mirman, Gehr, Vechev 2018, *Differentiable Abstract Interpretation for Provably Robust Neural Networks* | https://proceedings.mlr.press/v80/mirman18b.html | `Proofs/Foundation/IntervalBoundConv.lean` | — | no | F-at-3 |
| Zhang et al. 2018, *Efficient Neural Network Robustness Certification with General Activation Functions* (CROWN) | https://arxiv.org/abs/1811.00866 | `Proofs/Certificates/CrownBound.lean` (add the id), `formalization.yaml` | — | author-year only: `CrownBound.lean:9` | C-att-1 |
| Zhang et al. 2020, *Towards Stable and Efficient Training of Verifiably Robust Neural Networks* (CROWN-IBP) | https://arxiv.org/abs/1906.06316 | `Proofs/Certificates/CrownBound.lean`, `formalization.yaml` | — | name only | C-att-1 |
| Clopper & Pearson 1934, *The use of confidence or fiducial limits illustrated in the case of the binomial* | https://doi.org/10.1093/biomet/26.4.404 | `Proofs/Certificates/Smoothing/CP.lean`, `LeanMlir/Verified/Smoothing.lean`, `formalization.yaml` | — | name only (6 files) | C-att-1, X-att-4 |
| Madry et al. 2018, *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD) | https://arxiv.org/abs/1706.06083 | `LeanMlir/Verified/Attack.lean` | — | no | X-att-4 |
| Miyato et al. 2018, *Spectral Normalization for GANs* | https://arxiv.org/abs/1802.05957 | `LeanMlir/Verified/Attack.lean` (`projectSpectral`) | — | no | X-att-4 |
| Delattre et al. 2023, *Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration* (optional) | https://arxiv.org/abs/2305.16173 | `Proofs/Certificates/…` `denseE_lipschitzL2_gram{,2}` host module, only if the idea came from there | — | no | C (courtesy note) |
| Cohen, Rosenfeld, Kolter 2019, *Certified Adversarial Robustness via Randomized Smoothing* | https://arxiv.org/abs/1902.02918 | (already credited) | — | yes: 12 files incl. `Verified/Smoothing.lean` | — |
| Salman et al. 2019, *Provably Robust Deep Learning via Adversarially Trained Smoothed Classifiers* | https://arxiv.org/abs/1906.04584 | (already credited) | — | yes | — |
| Tsuzuku, Sato, Sugiyama 2018, *Lipschitz-Margin Training* | https://arxiv.org/abs/1802.04034 | (already credited) | — | yes | — |
| Fazlyab et al. 2019, *Efficient and Accurate Estimation of Lipschitz Constants for DNNs* (LipSDP) | https://arxiv.org/abs/1906.04893 | (already credited) | — | yes: `LipschitzCert/PairSDP.lean` | — |
| Hoeffding 1963, *Probability Inequalities for Sums of Bounded Random Variables* | https://doi.org/10.1080/01621459.1963.10500830 | `Proofs/Certificates/Smoothing/MC.lean` (add the reference to the name) | — | name only | — |

Note: the book cites none of the certification papers (no Cohen, Gowal, CROWN or Clopper–Pearson hit
in `content.tex`).

## Numerics & statistics

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Chan, Golub, LeVeque 1982, *Updating Formulae and a Pairwise Algorithm for Computing Sample Variances* (COMPSTAT; the pairwise combine rule) | https://doi.org/10.1007/978-3-642-51461-6_3 | `Proofs/Foundation/DataParallel/Sync.lean`, `SyncKit.lean`, `Proofs/Architectures/BatchNorm.lean` (`bnVar_shard_chan`), `LeanMlir/SyncBnCheck.lean` | — | name only ("Chan's", ~17 files) | A-attr-2, F-at-1, X-att-4 |
| Chan, Golub, LeVeque 1983, *Algorithms for Computing the Sample Variance: Analysis and Recommendations* (Am. Stat.) | https://doi.org/10.1080/00031305.1983.10483115 | same as above (companion, the survey the reports name) | — | no | F-at-1, X-att-4 |
| Micikevicius et al. 2022, *FP8 Formats for Deep Learning* (E4M3) | https://arxiv.org/abs/2209.05433 | `LeanMlir/E4M3Quant.lean` | — | no | X-att-4 |
| Higham 2008, *Functions of Matrices: Theory and Computation* (ch. 8, Newton–Schulz) | https://doi.org/10.1137/1.9780898717778 | `Proofs/Foundation/Muon/NewtonSchulz.lean`, `Muon/Geometry.lean` | — | name only ("Higham", 10 files, no title/year) | F-at-2 |

## Muon / Shampoo geometry

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Bernstein & Newhouse 2024, *Old Optimizer, New Norm: An Anthology* | https://arxiv.org/abs/2409.20325 | `Proofs/Foundation/Muon/Geometry.lean` (the steepest-descent table, `shampoo_eq_muon`) | — | no (only `formalization.yaml:224`) | F-at-2 |
| Jordan et al. 2024, *Muon: An optimizer for hidden layers in neural networks* (blog) | https://kellerjordan.github.io/posts/muon/ | `Proofs/Foundation/Muon/NewtonSchulz.lean`, `Muon/Geometry.lean` | — | author-year only: `NewtonSchulz.lean:313` "(Jordan 2024)" | F-at-2 |
| Gupta, Koren, Singer 2018, *Shampoo: Preconditioned Stochastic Tensor Optimization* | https://arxiv.org/abs/1802.09568 | `Proofs/Foundation/Muon/Geometry.lean` | — | name only | F-at-2 |

## RL / generative / physics demos

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Ho, Jain, Abbeel 2020, *Denoising Diffusion Probabilistic Models* | https://arxiv.org/abs/2006.11239 | `demos/MainDiffusion2d.lean` | 15630 | yes: `LeanMlir/Ddpm.lean`, `Bestiary/Diffusion.lean` | X-att-5 |
| Lipman et al. 2023, *Flow Matching for Generative Modeling* | https://arxiv.org/abs/2210.02747 | `LeanMlir/Ddpm.lean` (:86), `demos/MainDiffusion2d.lean` | 15731 | yes: `Bestiary/BoltzmannGenerator.lean` only | X-att-4, X-att-5 |
| Liu, Gong, Liu 2023, *Flow Straight and Fast* (rectified flow / reflow) | https://arxiv.org/abs/2209.03003 | `LeanMlir/Ddpm.lean` | 15830 | yes: `Bestiary/BoltzmannGenerator.lean` | X-att-4 |
| Tong et al. 2024, *Improving and generalizing flow-based generative models with minibatch optimal transport* | https://arxiv.org/abs/2302.00482 | `LeanMlir/Ddpm.lean` | 15833 | yes: `ffi/f32_helpers.c` only | X-att-4 |
| Pooladian et al. 2023, *Multisample Flow Matching* | https://arxiv.org/abs/2304.14772 | `LeanMlir/Ddpm.lean` | — | no | X-att-4 |
| Noé, Olsson, Köhler, Wu 2019, *Boltzmann generators* | https://doi.org/10.1126/science.aaw1147 | `demos/MainDiffusion2d.lean` | 15754 | yes: `Bestiary/BoltzmannGenerator.lean`, `Bestiary/README.md` | X-att-5 |
| Carleo & Troyer 2017, *Solving the quantum many-body problem with artificial neural networks* | https://doi.org/10.1126/science.aag2302 | `demos/MainNqsIsing.lean` | 16191 | no | X-att-5 |
| Mnih et al. 2015, *Human-level control through deep reinforcement learning* (DQN) | https://doi.org/10.1038/nature14236 | `demos/MainBlackjackDqn.lean` | 16513 | yes: `demos/MainPongDqn.lean`, `demos/README.md` | X-att-5 |
| van Hasselt, Guez, Silver 2016, *Deep Reinforcement Learning with Double Q-learning* | https://arxiv.org/abs/1509.06461 | `demos/MainPongDqn.lean`, `demos/MainBlackjackDqn.lean` | 16591 | no | X-att-5 |
| Silver et al. 2017, *Mastering the game of Go without human knowledge* (AlphaGo Zero; PUCT, the 40-block net) | https://doi.org/10.1038/nature24270 | `Bestiary/AlphaZero.lean`, `ffi/f32_helpers.c` (PUCT, :3648) | 16666, 16820 | no | X-att-5, X-att-6 |
| Silver et al. 2018, *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* (AlphaZero) | https://doi.org/10.1126/science.aar6404 | `Bestiary/AlphaZero.lean` (add title/venue) | 16667, 16817 (arXiv 1712.01815) | yes (author-year): `Bestiary/AlphaZero.lean`, `demos/MainAlphaZeroTtt.lean` | X-att-6 |
| Nair, *alpha-zero-general* (software) | https://github.com/suragnair/alpha-zero-general | `demos/MainAlphaZeroTtt.lean`, `demos/README.md` (:596), book RL paragraph | — | name only: `demos/README.md` | X-att-2 |
| Redmon et al. 2016, *You Only Look Once* (YOLOv1) | https://arxiv.org/abs/1506.02640 | `demos/MainYolov1NeuDetFpn.lean`, `MainYolov1VisdroneFpn.lean`, `MainYolov1NeuDet448.lean` | 14174 | yes: `Bestiary/YOLO.lean` | X-att-5 |
| Lin et al. 2017, *Feature Pyramid Networks for Object Detection* | https://arxiv.org/abs/1612.03144 | `demos/MainYolov1NeuDetFpn.lean`, `MainYolov1VisdroneFpn.lean` | 13280 (author-year) | partial: `Types.lean`, `Bestiary/MaskRCNN.lean` | X-att-5 |

## Bestiary gaps

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Jumper et al. 2021, *Highly accurate protein structure prediction with AlphaFold* | https://doi.org/10.1038/s41586-021-03819-2 | `Bestiary/Evoformer.lean` | 16432 | author-year only | X-att-6 |
| Gu & Dao 2023, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* | https://arxiv.org/abs/2312.00752 | `Bestiary/Mamba.lean` | 15215 | author-year only | X-att-6 |
| Zhang, Zhou, Lin, Sun 2018, *ShuffleNet* | https://arxiv.org/abs/1707.01083 | `Bestiary/ShuffleNet.lean` | 13973 | author-year only | X-att-6 |
| Liu et al. 2021, *Swin Transformer* | https://arxiv.org/abs/2103.14030 | `Bestiary/SwinT.lean` | 9602, 14061 | author-year only | X-att-6 |
| Ronneberger et al. 2015, *U-Net* | https://arxiv.org/abs/1505.04597 | `Bestiary/UNet.lean` (add title/id) | 14947 | author-year only | X-att-6 |
| Liu, Li, Wu, Lee 2023, *Visual Instruction Tuning* (LLaVA) | https://arxiv.org/abs/2304.08485 | `Bestiary/LLaVA.lean` (add id) | 16396 | title, no id | X-att-6 |
| Liu, Li, Li, Lee 2024, *Improved Baselines with Visual Instruction Tuning* (LLaVA-1.5) | https://arxiv.org/abs/2310.03744 | `Bestiary/LLaVA.lean` | 16397 | no | X-att-6 |
| Chen et al. 2018, *Encoder-Decoder with Atrous Separable Convolution* (DeepLab v3+) | https://arxiv.org/abs/1802.02611 | `Bestiary/DeepLabV3Plus.lean` (add id; also cite Sandler 2018 for the backbone) | 14972 | author-venue only | X-att-6 |
| Szegedy et al. 2015, *Going Deeper with Convolutions* (Inception v1) | https://arxiv.org/abs/1409.4842 | `Bestiary/Inception.lean` | 5634, 13776 | author-year only | X census (partial) |
| Szegedy et al. 2017, *Inception-v4, Inception-ResNet and the Impact of Residual Connections* | https://arxiv.org/abs/1602.07261 | `Bestiary/Inception.lean` | 13834 | author-year only | X census (partial) |
| Redmon & Farhadi 2018, *YOLOv3: An Incremental Improvement* | https://arxiv.org/abs/1804.02767 | `Bestiary/YOLO.lean` (add id) | — | author-year only | **new** |
| Ultralytics, *YOLOv5* (software) | https://github.com/ultralytics/yolov5 | `Bestiary/YOLO.lean` (:236) | — | name only | X-att-6 |
| Ultralytics, *ultralytics* (YOLOv8 / YOLO11, software) | https://github.com/ultralytics/ultralytics | `Bestiary/YOLO.lean` (:279) | 17237 | name only | X-att-6 |
| TensorFlow Model Garden (software) | https://github.com/tensorflow/models | `Bestiary/DeepLabV3Plus.lean` (:108) | — | name only | X census |
| Karpathy, *nanoGPT* (software) | https://github.com/karpathy/nanoGPT | `Bestiary/GPT.lean` (:38) | 15275 | name only | X census |

## Software & libraries acknowledged

| Work | Link | Cite in | Book | In Lean? | Findings |
|---|---|---|---|---|---|
| Wightman, *PyTorch Image Models* (timm) | https://github.com/huggingface/pytorch-image-models | `README.md` ("Built on"), `TRUST.md`, book intro; `Verified/NetsCore.lean` (timm parity) | — (named at 6351 etc., never linked) | name only (44 mentions) | X-att-1 |
| JAX (Bradbury et al. 2018, software) | https://github.com/jax-ml/jax | `README.md`, book intro | — (named) | name only | X-att-1 |
| OpenXLA XLA (incl. PJRT) | https://github.com/openxla/xla | `README.md`, `TRUST.md`, book intro | — (named at 184) | name only | X-att-1 |
| OpenXLA StableHLO | https://github.com/openxla/stablehlo | `README.md`, `TRUST.md` | — | name only | X-att-1 |
| IREE | https://github.com/iree-org/iree | `README.md`, `TRUST.md`, book intro | — (named at 192, 17805) | name only | X-att-1 |
| mathlib Community 2020, *The Lean Mathematical Library* (CPP 2020) | https://arxiv.org/abs/1910.09336 | `README.md`, book intro (146, 165) | — (named) | no | X-att-1 |
| Mathlib4 (software) | https://github.com/leanprover-community/mathlib4 | `README.md` | — | name only | X-att-1 |
| de Moura & Ullrich 2021, *The Lean 4 Theorem Prover and Programming Language* (CADE-28) | https://doi.org/10.1007/978-3-030-79876-5_37 | `README.md`, book intro | — | no | **new** (companion to X-att-1) |
| doc-gen4 | https://github.com/leanprover/doc-gen4 | `README.md` | — | name only | X-att-1 |
| leanblueprint (Massot) | https://github.com/PatrickMassot/leanblueprint | `README.md`, book front matter | — (line 4 loads the package) | no | X-att-1 |
| plasTeX | https://github.com/plastex/plastex | `README.md` | — | name only | X-att-1 |
| torchvision | https://github.com/pytorch/vision | `Proofs/Nets/ResNet/ResNet50FullB.lean` (ResNet v1.5), `README.md` | 13649, 17056 (named) | name only (21 files) | N1-attr-1, X census |
| Gymnasium (Farama Foundation) | https://github.com/Farama-Foundation/Gymnasium | (already credited, by name) `LeanMlir/Blackjack.lean` | — | name only | — |
| leanprover/comparator | https://github.com/leanprover/comparator | (already linked in the book) | 17925 | name only: `LeanMlir/Proofs/README.md` | — |

## Unverified

None. Every link above resolved, and its title matched the work on 2026-09-30. For Nesterov 1983,
the Math-Net.Ru page (Dokl. Akad. Nauk SSSR 269(3), 543–547) is the canonical record. The paper has
no DOI, and the English translation (Soviet Math. Dokl. 27) has no stable link.

Not linked. The Stanford tech report STAN-CS-79-773 (Chan, Golub & LeVeque 1979) that F-at-1 names
has no stable URL. The COMPSTAT 1982 DOI above is the published version of that report, so it is
the one to cite.

## Checked, not needed

- **Classical named algorithms whose name is the credit** (optional DOIs, all verified):
  Box–Muller 1958 (https://doi.org/10.1214/aoms/1177706645; `LeanMlir/Ddpm.lean`, `F32Array.lean`,
  `ffi`), Marsaglia–Tsang gamma 2000 (https://doi.org/10.1145/358407.358414; `ffi` only), xorshift
  (Marsaglia 2003, https://doi.org/10.18637/jss.v008.i14; xorshift64* is Vigna,
  https://arxiv.org/abs/1402.6246), Metropolis et al. 1953 (https://doi.org/10.1063/1.1699114;
  `demos/MainNqsIsing.lean`), Kuhn 1955 Hungarian method (https://doi.org/10.1002/nav.3800020109;
  `ffi`, `Bestiary/DETR.lean`), Wilson 1927 interval
  (https://doi.org/10.1080/01621459.1927.10502953; `Verified/Train.lean:181`), Lanczos 1964 gamma
  (https://doi.org/10.1137/0701008) and Lentz 1976 continued fraction
  (https://doi.org/10.1364/AO.15.000668; both `Verified/Smoothing.lean`, named), Keys 1981 bicubic
  (https://doi.org/10.1109/TASSP.1981.1163711; the repo follows PIL's implementation and credits
  PIL).
- **Acklam's probit approximation.** It is named in `Verified/Smoothing.lean:12` and
  `PhiBounds.lean`. It was only ever published on a personal web page that is no longer hosted, so
  the name is the credit.
- **Polyak & Juditsky 1992 (weight averaging / EMA),** https://doi.org/10.1137/0330046, verified.
  The EMA in `Types.lean` is a recipe knob, not a method the proofs state, so it needs no citation.
- **Textbook material:** power iteration (`Verified/Attack.lean`), minimax and alpha-beta
  (`TicTacToe.lean`, `ffi`), Gram/Schatten operator-norm bounds, the Sutton & Barto blackjack env
  (already credited), Grad-CAM/CAM, DDIM and score-SDE, improved DDPM, SWAG, Mixup/CutMix/RandAugment
  in `ffi` (already fully credited).
- **Datasets:** the book appendix (content.tex:16968–17220) already credits every dataset. (The
  TinyStories demo was deleted 2026-09-30, so its row is gone.) Licences are a separate X-att-5 item and need no citation.
