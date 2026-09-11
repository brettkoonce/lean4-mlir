# bestiary_candidates.md — architectures not yet in the bestiary, 2026-09-11

About 190 candidates, grouped by domain, one line each, for cross-correlating
against the external paper index. The tag is what the entry would cost the
reader in the bestiary's own currency:

- **0** — composes from the kit as it stands (Part 1 primitives + the
  bestiary-only ones in `Types.lean`).
- **1: x** — one bundled primitive to add, named.
- **prose** — the novelty is a loss, a recipe, a sampler or an inference
  procedure, not a layer; the entry is a paragraph beside an existing spec.

Already covered, so not listed: the chapter nets (MLP, CNN, ResNet-34/50,
MobileNetV2/V4, EfficientNet-B0, ConvNeXt-T/S/B, ViT-Ti/S/B, TinyGPT) and the
41 bestiary files — AlexNet, AlphaGo, AlphaZero, BERT, CLIP, CycleGAN, DCGAN,
DeepLabV3+, DenseNet, DETR, DDPM, Evoformer, GPT, Highway, Inception, LeNet,
LLaVA, Mamba, Mask R-CNN, MobileViT, MuZero, NeRF, Nyströmformer, Pix2Pix,
QANet, ResNet, SAM, SegFormer, ShuffleNet v1/v2, SqueezeNet, Stable Diffusion,
Swin, UNet, VAE, VGG, WaveNet, Whisper, WRN, Xception, YOLO.

## Vision backbones and classifiers

| entry (year) | what it is, and what the spec would show | cost |
|---|---|---|
| Network in Network (2013) | 1×1 convs as per-pixel MLPs, global average pooling instead of FC; the GAP the whole book uses was born here | 0 |
| ZFNet (2013) | AlexNet re-tuned through deconvolutional visualisation; the first "look inside" paper | 0 |
| PreAct ResNet (2016) | BN-ReLU-conv ordering makes the skip a true identity; one-line reorder of Ch 5's block | 0 |
| Stochastic depth (2016) | drop whole residual blocks in training; the drop-path the ImageNet recipes already use | prose |
| ResNeXt (2017) | grouped-conv bottleneck, "cardinality" as the third axis after depth and width | 1: grouped conv |
| SENet (2018) | squeeze-excitation on ResNet-50; the 2017 ImageNet winner, Ch 7's SE block on Ch 5's net | 0 |
| DPN (2017) | dual path: a ResNet stream and a DenseNet stream in one block | 1: dual-path block |
| NASNet (2018) | searched normal and reduction cells stacked by hand | 1: NAS cell |
| MnasNet (2019) | mobile NAS with latency in the reward; the direct ancestor of MobileNetV3 and EfficientNet | 0 |
| MobileNetV1 (2017) | the plain depthwise-separable stack; Ch 6 minus the inverted residual | 0 |
| MobileNetV3 (2019) | NAS + SE + hard-swish, `mbConvV3` already in the kit | 0 |
| GhostNet (2020) | cheap linear "ghost" feature maps from a few real ones | 1: ghost module |
| RegNet (2020) | a design space, not a net: quantised linear widths, grouped convs | 1: grouped conv |
| Res2Net (2021) | hierarchical residual splits inside one bottleneck | 1: res2 block |
| HRNet (2019) | parallel streams at four resolutions with repeated fusion; segmentation and pose workhorse | 1: multi-resolution fusion |
| EfficientNetV2 (2021) | fused-MBConv early stages + progressive training; `fusedMbConv` is in the kit | 0 |
| NFNet (2021) | normaliser-free ResNets: weight-standardised convs + adaptive gradient clipping | 1: WS conv |
| BiT (2020) | ResNet-v2 + GroupNorm + weight standardisation at transfer scale | 1: GroupNorm |
| CaiT (2021) | LayerScale + class-attention layers for deep ViTs; `layerScale` is in the kit | 0 |
| PVT (2021) | pyramid ViT with spatial-reduction attention; SegFormer's MiT descends from it | 1: SR attention |
| CvT (2021) | convolutional token embedding and convolutional projections inside attention | 1: conv projection |
| LeViT / PiT (2021) | conv stem, shrinking attention stages; pooling between transformer stages | 1: attention pooling |
| Twins (2021) | locally-grouped + globally-subsampled attention alternating | 1: LSA/GSA |
| CoAtNet (2021) | MBConv stages then transformer stages; the "conv early, attend late" recipe | 0 |
| MLP-Mixer (2021) | token-mixing and channel-mixing MLPs, no attention, no conv | 1: token mixing |
| ResMLP / gMLP (2021) | affine-normalised MLP mixer; spatial gating unit | 1: spatial gating |
| ConvMixer (2022) | patch embedding then depthwise/pointwise mixing; "patches are all you need" | 0 |
| ConvNeXt V2 (2023) | global response normalisation + fully-convolutional MAE pretraining | 1: GRN |
| Deformable ConvNets v1/v2 (2017/19) | learned sampling offsets per kernel tap | 1: deformable conv |
| InternImage (2023) | deformable-conv-v3 backbone at ViT scale | 1: deformable conv |
| RepVGG (2021) | multi-branch in training, re-parameterised to plain 3×3 at inference | prose |
| MobileOne / FastViT / EfficientViT (2022–23) | mobile backbones built on structural re-parameterisation and cheap attention | 1: reparam block |
| Vision Mamba / VMamba (2024) | state-space blocks on image patches; `mambaBlock` on `patchEmbed` | 0 |
| DLA (2018) | deep layer aggregation: iterative and hierarchical merges across stages | 1: aggregation node |
| Non-local networks (2018) | self-attention blocks inside a CNN; the pre-ViT attention-in-vision paper | 0 |
| CBAM (2018) | channel + spatial attention module | 1: spatial attention |
| Spatial Transformer Networks (2015) | a learned affine warp module with a differentiable sampler | 1: grid sample |
| Capsule Networks (2017) | routing by agreement between capsules; historical, never scaled | 1: routing |
| Dilated ResNet (2017) | dilation in place of stride keeps resolution for dense prediction | 1: dilated conv |

## Self-supervised and representation learning (vision)

| entry (year) | what it is | cost |
|---|---|---|
| SimCLR (2020) | contrastive learning with a projection head and big batches | 0, prose loss |
| MoCo v1–v3 (2020–21) | momentum encoder + queue of negatives | prose |
| BYOL (2020) | no negatives: online predictor chases a momentum target | 0, prose |
| SwAV (2020) | online clustering assignments as targets | prose |
| Barlow Twins / VICReg (2021) | redundancy-reduction and variance-invariance-covariance losses | prose |
| DINO / DINOv2 (2021/23) | self-distillation with a ViT; DINOv2 is the default frozen feature today | 0, prose |
| BEiT (2021) | masked image modelling with visual tokens from a dVAE | prose |
| MAE (2022) | asymmetric encoder-decoder with 75 % of patches masked; ViT pretraining that works | 0, prose |
| iBOT / data2vec (2022) | masked prediction of a teacher's latents | prose |
| SigLIP (2023) | CLIP with a sigmoid loss, no global softmax | 0, prose |
| ALIGN (2021) | CLIP at web scale with noisy alt-text | 0 |
| CoCa (2022) | contrastive + captioning in one encoder-decoder | 0 |

## Detection

| entry (year) | what it is | cost |
|---|---|---|
| R-CNN / Fast R-CNN (2014/15) | region proposals + per-region classifier; ROI pooling | prose |
| Faster R-CNN (2015) | the region proposal network; two-stage detection made end-to-end | 1: RPN head |
| SSD (2016) | multi-scale anchors on a VGG trunk, single shot | 0 |
| FPN (2017) | the feature pyramid itself, as its own entry; `fpnModule` is in the kit | 0 |
| RetinaNet (2017) | focal loss on an FPN; `useFocal` and `fpnModule` both exist | 0 |
| Cascade R-CNN (2018) | staged heads at rising IoU thresholds | prose |
| FCOS (2019) | anchor-free per-pixel boxes with centre-ness | 0 |
| CenterNet (2019) | objects as heatmap peaks; boxes regressed at the peak | 0 |
| EfficientDet (2020) | EfficientNet + BiFPN + compound scaling; the `bifpn_demo.md` plan exists | 1: BiFPN |
| YOLOX / YOLOv7 / v9 / v10 (2021–24) | anchor-free heads, reparameterised blocks, NMS-free training | 0 as YOLO variants |
| Deformable DETR (2021) | deformable attention over multi-scale features; fixes DETR's convergence | 1: deformable attention |
| DINO-DETR (2022) | denoising query training; the DETR that wins COCO | prose |
| RT-DETR (2023) | real-time DETR with an efficient hybrid encoder | 0 |
| OWL-ViT (2022) | open-vocabulary detection from a CLIP backbone | 0 |
| Grounding DINO (2023) | text-grounded detection, the SAM-era open-set detector | 0 |

## Segmentation and dense prediction

| entry (year) | what it is | cost |
|---|---|---|
| FCN (2015) | fully convolutional nets; skip fusion from pool3/pool4 | 0 |
| SegNet (2015) | encoder-decoder unpooling with saved max-pool indices | 1: index unpool |
| PSPNet (2017) | pyramid pooling module; the ASPP cousin | 1: pyramid pooling |
| UNet++ (2018) | nested, dense skips between encoder and decoder | prose |
| nnU-Net (2021) | the self-configuring UNet recipe that wins medical benchmarks with no new layers | prose |
| 3D U-Net / V-Net (2016) | volumetric convs; the BraTS 3D plan's target | 1: conv3d |
| TransUNet (2021) | ViT at the UNet bottleneck | 0 |
| UNETR / Swin-UNet (2021–22) | transformer encoders feeding a UNet decoder | 0 |
| Mask2Former (2022) | masked attention decoder, one model for semantic/instance/panoptic | 1: masked attention |
| SAM 2 (2024) | SAM with a streaming memory bank for video | 1: memory attention |
| MobileSAM / EfficientSAM (2023) | distilled SAM encoders | 0 |
| DPT (2021) | dense prediction transformer for depth | 0 |
| Depth Anything (2024) | DINOv2 features + DPT head at scale | 0 |
| RAFT (2020) | optical flow by iterative correlation-volume updates | 1: correlation volume |

## Generative: GANs, VAEs, flows, autoregressive

| entry (year) | what it is | cost |
|---|---|---|
| GAN (2014) | the original MLP generator and discriminator on MNIST | 0 |
| Conditional GAN (2014) | class label concatenated to both nets | 0 |
| WGAN / WGAN-GP (2017) | Wasserstein critic, gradient penalty; a loss, not a net | prose |
| Progressive GAN (2018) | grow resolution during training | prose |
| StyleGAN 1/2/3 (2019–21) | mapping network + modulated convs + noise inputs | 1: modulated conv |
| BigGAN (2019) | class-conditional BN, orthogonal regularisation, scale | 1: conditional BN |
| SAGAN (2019) | self-attention in the generator | 0 |
| VQ-VAE / VQ-VAE-2 (2017/19) | discrete codebook bottleneck with straight-through gradients | 1: vector quantiser |
| VQGAN (2021) | VQ-VAE + adversarial loss + transformer prior over codes | 1: vector quantiser |
| PixelCNN / PixelCNN++ (2016) | masked convolutions, autoregressive over pixels | 1: masked conv |
| NICE / RealNVP / Glow (2015–18) | normalising flows via affine coupling and invertible 1×1 | 1: coupling layer |
| β-VAE (2017) | scaled KL; disentanglement | prose |
| NVAE (2020) | deep hierarchical VAE with depthwise convs | 0 |
| MaskGIT (2022) | parallel masked token decoding over VQ codes | 0, prose |

## Generative: diffusion and flow

| entry (year) | what it is | cost |
|---|---|---|
| Sohl-Dickstein et al. (2015) | diffusion as nonequilibrium thermodynamics; the origin | prose |
| DDIM (2020) | the deterministic sampler and the η family; already in the demo | prose |
| Improved DDPM (2021) | cosine schedule, learned variance, hybrid loss | prose |
| Score-SDE (2021) | continuous-time view, reverse SDE and probability-flow ODE | prose |
| ADM / guided diffusion (2021) | the big UNet with attention at several resolutions + classifier guidance | 0 |
| Classifier-free guidance (2022) | label dropout in training, guidance scale at sampling | prose |
| Cascaded diffusion / Imagen (2022) | base + super-resolution stages, T5 text | 0 |
| DALL·E 2 / unCLIP (2022) | CLIP image-embedding prior + decoder | prose |
| DiT (2023) | ViT denoiser on latents with adaLN-zero conditioning | 1: adaLN |
| U-ViT (2023) | ViT with long skips as the denoiser | 0 |
| Consistency models (2023) | one-step generation by consistency training/distillation | prose |
| Flow matching / rectified flow (2022) | straight interpolants, velocity target; the Boltzmann plan | prose |
| Stochastic interpolants (2022) | the unifying view of diffusion and flow matching | prose |
| SD3 / MMDiT (2024) | two-stream multimodal DiT with joint attention | 1: joint attention |
| FLUX (2024) | rectified-flow MMDiT at scale | prose |
| ControlNet (2023) | trainable copy of the encoder joined by zero-initialised convs | 0, prose |
| LoRA (2021) | low-rank adapters on attention weights; fine-tuning for diffusion and LLMs alike | 1: low-rank dense |
| DreamBooth / textual inversion (2022) | subject-driven fine-tuning recipes | prose |
| Video diffusion / SVD / Sora-style (2022–24) | spatio-temporal attention over 3D patches | 1: 3D patchify |
| Instant-NGP (2022) | multiresolution hash-grid encoding for NeRF-class fields | 1: hash encoding |
| Mip-NeRF (2021) | integrated positional encoding over cones | prose |
| 3D Gaussian splatting (2023) | not a network: differentiable rasteriser of Gaussians; the NeRF successor | prose |
| DreamFusion (2022) | score-distillation sampling from a text-to-image model into a NeRF | prose |

## Language and sequence models

| entry (year) | what it is | cost |
|---|---|---|
| word2vec (2013) | skip-gram with negative sampling; an embedding table and a dot product | 0 |
| LSTM (1997) / GRU (2014) | the recurrent cells; the kit has no recurrence, the entry says why | 1: recurrent cell |
| seq2seq (2014) | encoder-decoder LSTMs for translation | 1: recurrent cell |
| Bahdanau attention (2015) | additive attention over encoder states; attention's first appearance | 1: additive attention |
| ELMo (2018) | bidirectional LSTM language model, contextual embeddings | 1: recurrent cell |
| Transformer (2017) | the original encoder-decoder; `transformerEncoder` + `transformerDecoder` compose it | 0 |
| Transformer-XL (2019) | segment recurrence + relative positions | prose |
| XLNet (2019) | permutation language modelling | prose |
| T5 (2020) | encoder-decoder, relative position bias, text-to-text framing | 0 |
| BART (2019) | denoising seq2seq pretraining | 0 |
| ELECTRA (2020) | replaced-token detection: a small generator, a discriminator | 0, prose |
| ALBERT / DistilBERT (2019) | parameter sharing; distillation to half depth | prose |
| GPT-3 (2020) | the same block at 175B; in-context learning as the finding | 0, prose |
| Chinchilla (2022) | compute-optimal scaling: tokens and parameters grow together | prose |
| PaLM (2022) | parallel attention/MLP blocks, SwiGLU, multi-query attention | 1: parallel block |
| LLaMA 1/2/3 (2023–24) | pre-RMSNorm, SwiGLU, RoPE, GQA; the open reference decoder | 1: RMSNorm (RoPE exists) |
| Mistral / Mixtral (2023) | sliding-window attention; sparse mixture of experts | 1: MoE routing |
| Switch Transformer / GShard (2021) | mixture-of-experts feed-forward with top-k routing | 1: MoE routing |
| DeepSeek-V2/V3 (2024) | multi-head latent attention + fine-grained MoE | 1: MLA |
| Longformer / BigBird (2020) | sparse attention patterns; parameter-identical to BERT, the Nyströmformer point | prose |
| Performer (2020) | random-feature attention | prose |
| FlashAttention 1/2/3 (2022–24) | IO-aware exact attention; a kernel, not a layer; a plan exists in the archive | prose |
| S4 (2022) | structured state spaces; Mamba's ancestor | 1: SSM kernel |
| Hyena (2023) | implicit long convolutions in place of attention | 1: long conv |
| RWKV (2023) | linear attention as an RNN; trains parallel, infers recurrent | 1: WKV block |
| RetNet (2023) | retention: parallel/recurrent/chunked forms | 1: retention |
| Mamba-2 (2024) | state-space duality; a `mambaBlock` variant | 0 |
| Jamba (2024) | Mamba + attention + MoE interleaved | 0 |
| ByT5 / Charformer (2021) | byte-level inputs, no tokeniser | prose |
| InstructGPT / RLHF (2022) | reward model + PPO on a language policy | 0 (see RL) |
| DPO (2023) | preference optimisation as a classification loss, no reward model | prose |
| Speculative decoding (2023) | draft model + verifier; inference only | prose |

## Speech and audio

| entry (year) | what it is | cost |
|---|---|---|
| DeepSpeech 2 (2015) | conv front end + bidirectional RNN + CTC | 1: CTC loss, recurrence |
| Tacotron 2 (2018) | seq2seq TTS with location-sensitive attention into WaveNet | 1: recurrent cell |
| Wav2Vec 2.0 (2020) | conv feature encoder + transformer + product quantiser, contrastive | 1: Gumbel VQ |
| HuBERT (2021) | masked prediction of clustered acoustic units | 0, prose |
| Conformer (2020) | conv module inside a transformer block; QANet's shape for speech | 0 |
| WaveGlow (2019) | flow-based vocoder | 1: coupling layer |
| HiFi-GAN (2020) | GAN vocoder with multi-period discriminators | 0 |
| SoundStream / EnCodec (2021–22) | neural audio codecs with residual vector quantisation | 1: RVQ |
| AudioLM / MusicLM / MusicGen (2022–23) | language models over codec tokens | 0 |
| VALL-E (2023) | TTS as codec-token language modelling | 0 |
| Jukebox (2020) | hierarchical VQ-VAE + transformers for raw music | 1: vector quantiser |

## Multimodal

| entry (year) | what it is | cost |
|---|---|---|
| ViLBERT / LXMERT (2019) | two-stream co-attention vision-language encoders | 1: co-attention |
| Perceiver / Perceiver IO (2021) | latent array cross-attends to any input; modality-agnostic | 1: latent cross-attention |
| Flamingo (2022) | perceiver resampler + gated cross-attention into a frozen LM | 1: gated cross-attention |
| BLIP / BLIP-2 (2022–23) | Q-Former bridging a frozen image encoder and a frozen LLM | 1: query transformer |
| PaLI / PaLI-X (2022–23) | ViT + encoder-decoder LM at scale | 0 |
| Kosmos-1 (2023) | interleaved image-text decoder | 0 |
| ImageBind (2023) | six modalities bound to image embeddings | 0, prose |
| Gemini / GPT-4V class (2023) | natively multimodal decoders; prose only, no public spec | prose |
| CLIPSeg / GLIP (2022) | language-conditioned segmentation and grounding | 0 |

## Graphs, molecules, proteins

| entry (year) | what it is | cost |
|---|---|---|
| GCN (2017) | spectral graph convolution; the graph primitive the kit lacks | 1: graph conv |
| GraphSAGE (2017) | sampled neighbourhood aggregation | 1: graph conv |
| GAT (2018) | attention over neighbours | 1: graph attention |
| MPNN (2017) | the message-passing framework that subsumes the three above | 1: message passing |
| SchNet (2017) | continuous-filter convolutions on atoms | 1: cf-conv |
| DimeNet (2020) | directional message passing with angles | 1: message passing |
| NequIP / MACE (2022) | E(3)-equivariant message passing; the ML-potential state of the art | 1: equivariant MP |
| Behler-Parrinello NNP (2007) / DeePMD (2018) | per-atom MLPs on local descriptors, summed to an energy | 0 |
| E(n)-equivariant GNN (2021) | equivariance from relative distances alone | 1: message passing |
| AlphaFold 2 (2021) | Evoformer + structure module; both blocks are already primitives | 0 |
| AlphaFold 3 (2024) | pairformer + a diffusion module over atom coordinates | 1: pairformer |
| ESM-2 / ESMFold (2022) | protein language model; BERT on residues | 0 |
| RFdiffusion (2023) | diffusion over protein backbones with RoseTTAFold | prose |
| FermiNet / PauliNet (2020) | permutation-equivariant streams into Slater determinants | 1: determinant |
| PointNet / PointNet++ (2017) | shared MLP + max pool over points; hierarchical grouping | 0 / 1: grouping |
| Point Transformer (2021) | attention over k-nearest points | 1: local attention |
| Deep Sets / Set Transformer (2017/19) | permutation-invariant pooling; induced set attention | 0 / 1: ISAB |

## Physics and scientific ML

| entry (year) | what it is | cost |
|---|---|---|
| Boltzmann generators (2019) | flows from N(0, I) to exp(−U/kT); the Müller-Brown plan | 1: coupling, or 0 as flow matching |
| Neural quantum states (2017) | a network as a wavefunction; the transformer-wavefunction plan | 0 |
| Transformer wavefunctions (2023) | ViT and GPT wavefunctions for frustrated spins | 0 |
| Neural ODE (2018) | an MLP velocity field with the adjoint method | 0 |
| Hamiltonian / Lagrangian NNs (2019/20) | dynamics from the input gradient of a learned energy; second-order training | prose |
| PINNs (2019) | PDE residuals as the loss, differentiating the net in its inputs | prose |
| DeepONet (2021) | branch and trunk MLPs joined by a dot product; operator learning | 0 |
| FNO (2021) | spectral convolution in Fourier space; the PDE-surrogate workhorse | 1: FFT conv |
| FourCastNet (2022) | adaptive FNO inside a ViT for global weather | 1: FFT conv |
| GraphCast (2023) | GNN on an icosahedral mesh, six-hour steps | 1: message passing |
| Pangu-Weather (2023) | 3D Swin transformer on pressure levels; `swinStage` exists | 0 |
| Aurora (2024) | a foundation model for the atmosphere, Swin-based | 0 |
| GNS (2020) | graph network simulators for particles and fluids | 1: message passing |
| Lattice-field-theory flows (2020) | normalising flows for lattice QCD sampling | 1: coupling |
| SINDy / neural symbolic regression (2016–) | sparse identification of dynamics; not a net | prose |
| AlphaTensor (2022) | RL over tensor decompositions; AlphaZero applied | prose |

## Reinforcement learning and control

| entry (year) | what it is | cost |
|---|---|---|
| DQN (2015) | three strided convs + dense on 84×84×4; the blackjack/Pong plans | 0 |
| Double / Dueling DQN, Rainbow (2016–18) | target decoupling; V + A heads; distributional C51 head | 0 as variants |
| A3C / A2C (2016) | shared trunk, policy and value heads, asynchronous actors | 0 |
| TRPO / PPO (2015/17) | trust-region and clipped surrogate losses on tiny MLPs | prose |
| DDPG / TD3 / SAC (2015–18) | actor MLP + critic MLP on state-action concat | 0 |
| IMPALA (2018) | the three-stage conv-residual trunk every later agent reused | 0 |
| R2D2 / Agent57 (2019–20) | recurrent replay, exploration bonuses | 1: recurrent cell |
| World Models (2018) | VAE + MDN-RNN + tiny controller | 1: MDN head |
| Decision Transformer (2021) | GPT over return, state and action tokens | 0 |
| DreamerV3 (2023) | conv encoder/decoder around a recurrent state-space model | encoder 0, RSSM prose |
| Diffusion Policy (2023) | a UNet denoiser over action horizons conditioned on observations | 0 |
| Gato (2022) | one transformer over tokens from every modality and task | 0 |
| RT-1 / RT-2 (2022–23) | EfficientNet + FiLM + token learner into a transformer; VLM as policy | 1: FiLM |
| VPT (2022) | behaviour cloning from video with an inverse-dynamics labeller | prose |
| RLHF / GRPO (2022/24) | reward-model PPO on an LM; GRPO drops the critic | 0, prose |
| AlphaStar / OpenAI Five (2019) | LSTM-heavy multi-agent systems; prose only | prose |

## Tabular, time series, recommendation

| entry (year) | what it is | cost |
|---|---|---|
| Wide & Deep (2016) | linear memorisation + MLP generalisation | 0 |
| DeepFM / DLRM (2017/19) | embeddings + MLP + pairwise feature interactions | 1: dot interaction |
| Two-tower retrieval (2016) | user and item towers, dot-product scoring | 0 |
| NCF (2017) | neural collaborative filtering; MLP over embeddings | 0 |
| SASRec / BERT4Rec (2018–19) | sequential recommendation with causal or masked transformers | 0 |
| TabNet (2019) | sequential attentive feature selection for tabular data | 1: attentive mask |
| TabTransformer / FT-Transformer (2020–21) | transformers over column embeddings | 0 |
| DeepAR (2017) | autoregressive RNN forecaster with a likelihood head | 1: recurrent cell |
| N-BEATS (2019) | stacked MLP blocks with backcast/forecast residuals | 0 |
| Temporal Fusion Transformer (2019) | gating, variable selection, interpretable attention | 1: variable selection |
| PatchTST (2022) | patches over time into a vanilla encoder | 0 |
| Chronos (2024) | T5 over quantised series values | 0 |

## Normalisation, efficiency, training (architecture-adjacent)

| entry (year) | what it is | cost |
|---|---|---|
| GroupNorm (2018) | batch-independent normalisation over channel groups; what the CIFAR DDPM needed | 1: GroupNorm |
| InstanceNorm / AdaIN (2016/17) | per-image, per-channel normalisation; the style-transfer normaliser | 1: InstanceNorm |
| RMSNorm (2019) | LayerNorm without the mean; every modern LLM | 1: RMSNorm |
| Weight standardisation (2019) | normalise the conv kernel, not the activation | 1: WS conv |
| Dropout (2014) | the regulariser; prose against the kit's existing flag | prose |
| Knowledge distillation (2015) | soft-target training from a teacher; DeiT's token is one form | prose |
| Deep Compression (2016) | pruning + quantisation + Huffman; the origin of the compression line | prose |
| Lottery Ticket Hypothesis (2019) | sparse subnetworks trainable from init | prose |
| BitNet / BitNet b1.58 (2023–24) | ternary weights in every dense | 1: ternary dense |
| QLoRA (2023) | 4-bit base + LoRA adapters | prose |
| Mixture-of-Depths (2024) | per-token routing past blocks | 1: token routing |
| SAM sharpness-aware minimisation (2021) | an optimiser step, not a net | prose |
| Neural Tangent Kernel (2018) | infinite-width theory; the linearisation the book's proofs sit next to | prose |
| Batch Normalization (2015) / Layer Normalization (2016) | already chapters 4 and 8; entries only to anchor the lineage | 0 |
