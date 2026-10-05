# Recovered feed-perf prototypes (2026-09-28/29)

All of it comes from one transcript, `09ff3375-5b36-40c8-8aeb-bad437be7c1d.jsonl` (session
"side quests", 2026-09-28 21:30 to 09-29 02:30 UTC). Its scratchpad
(`/tmp/claude-1000/.../09ff3375-.../scratchpad/{augbench,smoke,probe}`) no longer exists, so every
file was rebuilt by replaying the `cat > … <<'EOF'` heredocs and the `python3 - <<'EOF'` edits in
order. Each file is the last version written. None of the markers (`_prep_u8`,
`_random_erase_px`, `pilrot`, `mix_inplace2`, `warp1d_u8`, `shim_scale.sh`) appear in any other
transcript, including 08a2c944 and the six others named in the task.

Paths inside the scripts still point at the old scratchpad and at `jax/.lake/build/…`. Fix them
before re-running. `jax/.lake/build/generated_vit_s_imagenet.py` was byte-identical to
`jax/generated/generated_vit_s_imagenet.py` at `c4a8ee2c` (2026-09-27), which is also HEAD's
copy. The line numbers the transcript printed (326, 365, 1728, 1750, 1936 in the base; 365, 399,
400 and 1968 in the patched file) all match.

## probe/: the two GPU perf probes of the ViT-S JAX trainer (§5)

| file | source (line, UTC) | state |
|---|---|---|
| `patch.py` | Bash heredoc l.1014 (09-29 00:09:23), then edit l.1047 (00:10:15): `@jit` placement fix | complete, final |
| `vits_u8.py` | `patch.py <base> out u8`, regenerated here on the c4a8ee2c base. Compiles. | full file, reconstructed |
| `vits_u8_pilrot.py` | same with `u8,pilrot` | full file, reconstructed |
| `vits_u8.diff`, `vits_u8_pilrot.diff` | unified diffs against the base | for reading |
| `run.sh` | heredoc l.1018 (00:09:33): sequential 15-min `timeout -s INT 900` runs on GPUs 0–3, no checkpoints | complete |

The first launch failed (`SyntaxError`: `_prep_u8` was inserted between `@jit` and
`eval_batch`). The l.1047 fix is included.

**Measured** (4× 4060 Ti, overall avg ms/step, before the DIMM fan fix; tool results l.1065,
l.1077; plan §5):

| variant | overall ms/step | vs shipped | stall windows |
|---|---|---|---|
| shipped | 656 | — | 5 of 8 after onset |
| uint8 wire, normalize/transpose on device | 569 (avg read 557 at step 1600) | −13% | 7 of 15 |
| uint8 wire + PIL Rotate | 474 (avg read 485 at step 1800) | −28% | 4 of 17 |

The clean 100-step windows were ~29 s in both arms, i.e. 290–300 ms/step (compute). The gains
came from fewer stall pauses.

### Design of the uint8 variant (`u8` arm)

* **Producer (tf.data `_pp`, train split only):** after decode/crop/resize/flip/RandAugment, the
  float tail (`cast → (x−mean)/std → _random_erase → transpose CHW → reshape flat`) is replaced
  by `_random_erase_px(float(img))` followed by `cast(clip(round(img), 0, 255), uint8)`. The wire
  becomes **uint8 HWC `[B, 224, 224, 3]`**, a quarter of the bytes, with no transpose or flatten
  on the host.
* **Erasing in pixel space:** `_random_erase_px` is a copy of `_random_erase` with the same box
  draws (p 0.25, area U(0.02, 1/3), log-uniform aspect, first of 10 that fits). Its fill
  `N(0,1)` becomes `N(0,1)·STD_RGB + MEAN_RGB`, so after device normalization the box holds
  N(0,1) noise. The fill is then rounded and clipped to 0..255 uint8, so the noise is quantized
  (steps of 1/(0.225·255) ≈ 0.017σ) and truncated at roughly −1.8σ to +2.6σ, depending on the channel. That is why the variant is **not
  recipe-exact**.
* **Consumer (device):** a new `@jit _prep_u8(x)`:
  `(x.astype(f32) − _U8_MEAN) / _U8_STD`, then `transpose(0,3,1,2).reshape(B, −1)`. The means
  and stds are the same 0.485/0.456/0.406 ×255 and 0.229/0.224/0.225 ×255. It is called right
  after `x, y = next(train_iter)` (which `prefetch_to_device` shards as uint8), before the
  device-side `_mixup`/`_cutmix` and `train_step`. Nothing else in the step changed.
* **Validation / eval stays float:** the `else` branch keeps normalize + transpose + flatten on
  the host, so val batches still ship float32 CHW-flat. Only the train split is uint8.
* The verified-path shims (`*_shim.py`, wire v2/v4 float32 CHW) were **not** touched. No uint8
  shim wire was ever prototyped (plan §4.3 L3: "none built").

### PIL Rotate (`pilrot` arm)

It is inserted just before `_RA_INC = True`, after the bicubic block overrides `_AA_OPS`:
```python
from PIL import Image as _PILImage
def _pil_rot(x, d):
    return np.asarray(_PILImage.fromarray(x).rotate(float(d), resample=_PILImage.BICUBIC, fillcolor=(128, 128, 128)))
def _aa_rotate_pil(img, deg):
    out = tf.numpy_function(_pil_rot, [img, tf.cast(deg, tf.float32)], tf.uint8, stateful=False)
    out.set_shape(img.shape); return out
_AA_OPS['Rotate'] = (_aa_rotate_pil, _aa_rot, True)
```
Only Rotate is replaced. Shear and Translate keep the emitted 4-tap `_aa_warp1d`, because PIL is
no faster for those (2.0 ms against 2.0). The `deg` sign and magnitude come from the existing
`_aa_rot` and `_aa_apply_op`. This does not reuse the emitted `_aa_rotate` matrix; it relies on
PIL's `rotate` (centre, counter-clockwise), which is what timm calls. No exactness check against
the emitted rotate was run for this arm.

## augbench/: CPU-only harness (§4)

| file | source (line, UTC) | what it does | measured (tool result) |
|---|---|---|---|
| `bench.py` | l.596 (22:05:34) + append l.652 (22:08:37) | single-thread per-stage ms/img using the trainer's own functions | decode+crop+resize 1.90; RA Rotate 7.77, Sharpness 3.14, Shear/Translate ~2.2, Equalize 1.50, rest <1; RA(2,m9) 2.03; normalize+erase 1.29–1.48; normalize 0.71; normalize+transpose 0.78; erase 0.83 ms/img |
| `pipe.py` | l.608, edits l.615/627/634/659, final edit l.675; base text from the l.672 `sed` dump | first end-to-end tf.data harness (det/nondet/u8/src/decode) | det 2,576 / nondet 2,568 img/s. ⚠ Its in-memory edits never took effect (AutoGraph re-reads source from the file), so the u8/src/decode numbers (~2,500) are void. Superseded by pipe2. |
| `source.py` | l.680 (22:11:31) | tfds source / shuffle / flat_map / batch ceilings | raw 13,566, shuf 12,853, full 12,842 ex/s; batch 25,311 img/s |
| `pipe2.py` | l.692 (22:12:15) + l.714 (writes edited source to `mod_<md5>.py`) + l.772 (adds `pilrot`, `norot`) | tf.data knock-outs by flag: `nodecode, nora, u8, noerase, nondet, pilrot, norot`. Its `u8` simply drops the float tail (no device side). | ViT-S: as-shipped 2,580; noerase 2,756; u8 3,145; nora 4,332; nora,u8 7,056; nodecode,u8 ~23,900; pilrot 2,917; norot 3,124; pilrot,u8 3,989. R50 A2 3,808 / u8 4,754 / nora 4,827; MNv4 2,692 / 3,143 / 5,047; ConvNeXt-S 2,551 / 3,067 / 4,503 |
| `warp_proto.py` | l.751 (22:19:18) | bicubic warp prototypes: `transform_u8taps` (uint8 gathers), `transform_onegather` (one fused 16-tap gather + 2 einsums), `warp1d_u8` | see warp_bench |
| `warp_bench.py` | l.751 | timing and exactness against the emitted ops, 40 random 224² images | rotate emitted 9.65, u8 taps 8.68, one gather 13.97, TF bilinear 1.57 ms; shearX/Y emitted 2.03/1.99, u8 2.02/2.04. All three are bit-exact (max \|Δ\| 0) |
| `pil_bench.py` | l.756 (22:19:47) | PIL BICUBIC costs | rotate 2.037, shearX 2.039, np↔PIL rotate 2.237 ms/img |
| `shim_reader.py`, `shim_scale.sh` | l.813 (22:31:26) | N concurrent shims (`SHIM_SHARD=i/N`, batch 128, NCLASSES 1000), aggregate img/s | ViT-S 2/3/4/6/8 workers: 1,154/1,422/1,571/1,637/1,393; ViT-S `SHIM_MIX=off` ×4: 2,107; MNv4 ×4 2,024, ×8 1,638 |
| `mix_bench.py` | l.879 (22:35:30) | host mixup/cutmix variants, B=128 | mixup now 25.4, 1-temp 24.7, **blocked in-place (`mix_inplace2`) 17.0** ms; cutmix now 25.3, box copy 14.6 ms; both bit-identical (max \|Δ\| 0.0) |

The blocked in-place mixup idea later shipped as `_mix_rows` in f5b6853f.

## smoke/

| file | source | what |
|---|---|---|
| `run.sh` | l.914 (22:56:26) | the batch-1 15-min JAX smokes (A2, MNv4 `full`, ViT-S), the shipped-trainer baseline (656 ms/step ViT-S) the probes compare against |

## Not recovered

* The `mod_<hash>.py` files pipe2 wrote, the probe/smoke logs and the task outputs. They were
  scratch only; the numbers quoted above come from the tool results.
* No uint8 change was ever made to the verified shims or to `jax/Jax/Codegen.lean`. The u8 work
  exists only as these scratch patches of the generated JAX trainer.
