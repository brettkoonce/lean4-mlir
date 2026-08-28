# 2026-08-28 — ConvNeXt-S and -B get job configs, and `LEAN_MLIR_MEM_FRACTION` turns out not to be free

Two results, and the second corrects something committed the day before.

1. **ConvNeXt-S and ConvNeXt-B are launchable**, on the same footing ViT-S turned out to be:
   rendered, tied, stepping, and missing only a job config. `cnxs-default-4gpu` and
   `cnxb-default-4gpu` now exist.
2. ⛔⛔ **Raising the allocator fraction to 0.97 BREAKS runs that work at the default.** It was
   written into two ViT configs as free headroom. It is not. One of them is fixed.

⛔ Nothing has been TRAINED.

---

## 1. The measurements

40 steps on real ImageNet, four 4060 Ti, median of steps 9–40, eval skipped, global batch 128.

| net | fp32 | bf16 | 300 epochs (incl. 37.5 s/epoch eval + ckpt) |
|---|---|---|---|
| ConvNeXt-S | 367 ms | 304 ms | 309 → 257 h |
| ConvNeXt-B | 547 ms | 458 ms | 459 → 385 h |

⭐ **The book's ConvNeXt-S row was DERIVED, and this is its second independent confirmation.**
`sec:convnext_sb` computed 366 → 292 from a near-constant trainer/graph multiple and recorded a
later direct probe at 368 → 301. This session reads **367 → 304**. ⚠ Against B's committed
534 → 446 this session reads 547 → 458, i.e. ~2–3 % slow, which is the same drift ConvNeXt-T showed
(231 against a committed 217). ▶ So the committed rows are NOT overwritten; the derivation is
confirmed, which is what was actually in question.

⛔⛔ **The step COUNT is this net's cost and a bigger batch does not fix it.** ConvNeXt runs 10,009
steps/epoch at global 128, double every other net here. Raising the batch is the obvious move and
`bf16_renderer.md` §21.5 already measured why it fails: the block interior's cost per IMAGE is flat
from batch 8 to 128 (1.03× over the whole range), so doubling the batch halves the step count and
doubles the step. Memory was never the constraint either — bs64 extrapolates to 76 % of the default
arena.

⭐ **This is exactly where ConvNeXt and ViT diverge**, and both answers are measured. ViT-B's
per-image cost *does* improve with batch, so global 512 bought it 322 → 291 h on top of the
fidelity. ConvNeXt has no idle throughput to sell. Same question, same box, opposite answers.

## 2. ⛔⛔ The allocator fraction is not free headroom

`runs/2026-08-27-vitb-global512/` established that PJRT's CUDA plugin reserves only 75 % of a card
by default and that `LEAN_MLIR_MEM_FRACTION=0.97` lifts that to 15.11 GiB, which is what lets
ViT-B's fp32 graph run at all. The unexamined step was treating the option as **costless** — and
`vits-default-g512-4gpu` shipped setting it purely as headroom, on the argument that 10.27 GiB of
an 11.68 arena is 88 % and a fragmented pool might fail a long run.

⚠⚠ **ConvNeXt falsified that within a day.**

| run | fraction unset (11.68 GiB) | fraction 0.97 (15.11 GiB) |
|---|---|---|
| ConvNeXt-S fp32 | (not run) | ✅ 367 ms/step |
| ConvNeXt-S bf16 | ✅ 304 ms/step | ⛔ `CUDA_ERROR_OUT_OF_MEMORY` in `d2h(res)` |
| ConvNeXt-B fp32 | (not run) | ✅ 547 ms/step |
| ConvNeXt-B bf16 | ✅ 458 ms/step | ⛔ same, **and it dumps core** |

⭐ **The direction is what makes this worth recording.** The bf16 graphs are the SMALLER ones —
6.98 GiB against 7.95 for S, 9.73 against 10.93 for B — and they are the ones that die, while their
larger fp32 peers survive at the same setting. So this is not the graph outgrowing the pool. It is
the pool crowding out everything that is *not* in it: the device-to-host staging for this net's
several hundred result buffers, NCCL, workspaces. A 97 % arena leaves under a gigabyte for all of
that.

✅ **ViT-S was then probed both ways, which is what decides its config rather than an argument:**

```
LEAN_MLIR_MEM_FRACTION=0.97   528 → 319 ms/step
                     (unset)  531 → 323 ms/step
```

▶ Noise. The option bought that net nothing and carried a now-demonstrated risk, so it is **removed
from `vits-default-g512-4gpu`**.

### What each config does now, and why they differ

| job | sets the fraction? | why |
|---|---|---|
| `vitb-default-g512-4gpu` | **yes, 0.97** | its fp32 graph is 13.99 GiB and does not otherwise fit; the driver refuses without it |
| `vits-default-g512-4gpu` | no | fits at 88 %; measured no faster with it |
| `cnxs-default-4gpu` | no, and **refuses if set** | 0.97 kills its bf16 arm |
| `cnxb-default-4gpu` | no, and **refuses if set** | same, with a core dump |

⚠ **The ConvNeXt configs REFUSE rather than warn**, and the refusal was run red before being
trusted — adding the line makes `supervise.sh` print *"on this net that is not headroom — it is a
crash"* and stop. That is the mirror image of ViT-B's refusal, which fires when the option is
ABSENT. Two nets, opposite requirements, and neither is discoverable from the graph's peak alone.

⭐ **The general lesson, stated so it is not re-learned a third time.** A compile-time
`peak_memory_in_bytes` says whether the GRAPH fits the pool. It says nothing about what the process
needs outside the pool, and that is not small for a multi-replica run with many result buffers.
▶ So: raise the fraction when a graph does not otherwise fit, and leave it alone when it does.
"There is room, so take it" is the reasoning that produced this bug.

## Reproduce

```
# the four ConvNeXt probes (two of them are expected to fail at 0.97)
LEAN_MLIR_VARIANT=adamdpwxclipdrop  LEAN_MLIR_BATCH=32 … convnext-s-imagenet-verified
```

The exact invocations are in each `probe_*.log`'s header line; all use
`LEAN_MLIR_MAX_STEPS=40 LEAN_MLIR_SKIP_EVAL=1` on `CUDA_VISIBLE_DEVICES=0,2,3,4`.
