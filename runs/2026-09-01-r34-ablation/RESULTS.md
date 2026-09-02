# ResNet-34 / Imagenette recipe ablation — fp32, XLA/PJRT, 80 ep, bs 32

Post-`f4e4172`, one vintage, 2026-09-01. Final top-1 over the 3,925-image val split.
Recovered from the runner's completion lines; see the note at the bottom.

| arm | top-1 | Δ vs full | what it removes |
|---|---|---|---|
| full    | 90.420382 | —      | nothing (AdamW, cosine+3ep warmup, wd, label smoothing, aug) |
| nowd    | 89.783439 | −0.64  | weight decay (render `resnet34_adamwd00`) |
| nowarm  | 89.375796 | −1.04  | the 3-epoch warmup |
| nols    | 88.917197 | −1.50  | label smoothing (render `resnet34_adamls0`) |
| noadam  | (running) |        | AdamW → Nesterov momentum (render `resnet34_mom`) |
| nocos   | 85.936306 | −4.48  | the cosine schedule (constant lr) |
| noaug   | 83.414013 | −7.01  | random crop + hflip (`LEAN_MLIR_NO_AUG`) |
| bare    | 84.433121 | −5.99  | momentum, no warmup, no schedule |

⛔ **"The contributions are essentially additive" DOES NOT HOLD on these numbers.** The
leave-one-out deltas sum to ≈ −15.9 (excl. noadam) against a bare delta of −5.99. The old
(pre-2026-04-24, IREE) table argued −17.38 vs −17.47 and concluded no interaction; at one
vintage on XLA the ingredients overlap heavily instead. `bare` alone moved +11.6 points
(72.82 → 84.43) against that table.

⚠ **`noadam` is not "vanilla SGD".** `R34Opt` has no plain-SGD case; the arm swaps AdamW for
Nesterov momentum with the rest of the recipe intact. The old table's "vanilla SGD, lr 0.01"
row was never that.

⚠ **nowarm / nocos / noaug lost their per-epoch curves**, not their results: a bf16 sweep was
started against this directory before `run_r34_ablation.sh` had its overwrite guard, and the
redirect truncated three finished logs. The final numbers above come from the runner's own
completion lines and are intact. Re-run those three arms if the chapter needs their curves.
