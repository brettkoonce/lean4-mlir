# 2026-09-01 — VisDrone decode sweep + the box-aware affine A/B

Two results from one session, both on the `cfoc2` arm
(`FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_EPOCHS=12`, R34 backbone, tower 0).
Full analysis in `planning/visdrone_detector.md` §13a / §13b; this file is the workings.

## Provenance

Logits regenerated from the committed checkpoint — the previous dumps had been
cleaned off disk:

```
LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=2 \
  FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=cfoc2 \
  .lake/build/bin/yolov1-visdrone-fpn infer data/visdrone_fpn runs/2026-09-01-cfoc2-rescore
```

548 val records, 185,220-wide output, 406,002,240 bytes. Binary
`.lake/build/bin/yolov1-visdrone-fpn` dated 2026-08-29 01:46 — built ~2 min
before `b25d7ce` (the class-focal commit), i.e. the binary that produced the
0.1961 on record. PJRT shim `ffi/libpjrt_ffi.so` unchanged since the branch point.

Each row below is one invocation of:

```
python3 scripts/yolo_map_visdrone.py runs/2026-09-01-cfoc2-rescore/logits.bin \
  data/visdrone448/val.bin --fpn data/visdrone --grid 14 --multilabel \
  --topk <T> --ml-k <K> --ml-floor <F>
```

Run 10-way parallel on 32 cores, ~51 s each solo. `rare4` is the unweighted mean
AP over bicycle / awning-tri / tricycle / truck.

## Faithfulness checks (do these before believing any row)

| check | expected (§13) | got |
|---|---|---|
| topk 3000, k 3, floor 0.05 | 0.1961 / 0.4414 / 0.7485 | **0.1961 / 0.4414 / 0.7485** |
| ml-k 1 (= argmax) | 0.1762 | **0.1762** |

## The 60 points

```
 topk ml-k floor     mAP  ca-AP recall     dets  rare4
 8000   10   0.0  0.1974 0.4417 0.7598  3559109 0.0767
 8000   10  0.01  0.1974 0.4417 0.7598  3559112 0.0767
 8000   10  0.05  0.1974 0.4417 0.7595  3568532 0.0766
 8000    5   0.0  0.1974 0.4417 0.7598  3562804 0.0766
 8000    5  0.01  0.1974 0.4417 0.7598  3562807 0.0766
 8000    5  0.05  0.1973 0.4417 0.7595  3570758 0.0765
 8000    3   0.0  0.1967 0.4417 0.7600  3609556 0.0757
 8000    3  0.01  0.1967 0.4417 0.7600  3609557 0.0757
 8000    3  0.05  0.1967 0.4417 0.7597  3612758 0.0757
 3000   10   0.0  0.1966 0.4414 0.7483  1273950 0.0755
 3000   10  0.01  0.1966 0.4414 0.7483  1273950 0.0755
 3000   10  0.05  0.1966 0.4414 0.7483  1274201 0.0754
 3000    5   0.0  0.1966 0.4414 0.7483  1274120 0.0754
 3000    5  0.01  0.1966 0.4414 0.7483  1274120 0.0754
 3000    5  0.05  0.1966 0.4414 0.7483  1274335 0.0754
 8000   10  0.15  0.1962 0.4416 0.7560  3662699 0.0747
 8000    5  0.15  0.1962 0.4416 0.7560  3662699 0.0747
 3000    3   0.0  0.1961 0.4414 0.7486  1280809 0.0749
 3000    3  0.01  0.1961 0.4414 0.7486  1280809 0.0749
 3000    3  0.05  0.1961 0.4414 0.7485  1280890 0.0749
 8000    3  0.15  0.1960 0.4416 0.7560  3667623 0.0746
 3000   10  0.15  0.1958 0.4414 0.7475  1291835 0.0742
 3000    5  0.15  0.1958 0.4414 0.7475  1291835 0.0742
 3000    3  0.15  0.1956 0.4414 0.7476  1293186 0.0741
 8000    2   0.0  0.1947 0.4417 0.7556  3688685 0.0731
 8000    2  0.01  0.1947 0.4417 0.7556  3688685 0.0731
 8000    2  0.05  0.1947 0.4417 0.7556  3688780 0.0731
 3000    2   0.0  0.1944 0.4415 0.7477  1300705 0.0726
 3000    2  0.01  0.1944 0.4415 0.7477  1300705 0.0726
 3000    2  0.05  0.1944 0.4415 0.7477  1300708 0.0726
 8000    2  0.15  0.1944 0.4416 0.7546  3707466 0.0727
 3000    2  0.15  0.1942 0.4415 0.7475  1306417 0.0723
 1000   10   0.0  0.1921 0.4390 0.7113   403832 0.0703
 1000   10  0.01  0.1921 0.4390 0.7113   403832 0.0703
 1000   10  0.05  0.1921 0.4390 0.7113   403837 0.0703
 1000    5   0.0  0.1921 0.4390 0.7113   403836 0.0703
 1000    5  0.01  0.1921 0.4390 0.7113   403836 0.0703
 1000    5  0.05  0.1921 0.4390 0.7113   403841 0.0703
 1000   10  0.15  0.1919 0.4390 0.7114   405003 0.0701
 1000    3   0.0  0.1919 0.4390 0.7115   404303 0.0702
 1000    3  0.01  0.1919 0.4390 0.7115   404303 0.0702
 1000    3  0.05  0.1919 0.4390 0.7115   404306 0.0702
 1000    5  0.15  0.1919 0.4390 0.7114   405003 0.0701
 1000    3  0.15  0.1918 0.4391 0.7116   405146 0.0700
 1000    2   0.0  0.1909 0.4391 0.7117   406798 0.0690
 1000    2  0.01  0.1909 0.4391 0.7117   406798 0.0690
 1000    2  0.05  0.1909 0.4391 0.7117   406798 0.0690
 1000    2  0.15  0.1909 0.4391 0.7118   407227 0.0689
 8000    1   0.0  0.1780 0.4426 0.7353  3873623 0.0633
 8000    1  0.01  0.1780 0.4426 0.7353  3873623 0.0633
 8000    1  0.05  0.1780 0.4426 0.7353  3873623 0.0633
 8000    1  0.15  0.1780 0.4426 0.7353  3873625 0.0633
 3000    1   0.0  0.1779 0.4425 0.7330  1377361 0.0632
 3000    1  0.01  0.1779 0.4425 0.7330  1377361 0.0632
 3000    1  0.05  0.1779 0.4425 0.7330  1377361 0.0632
 3000    1  0.15  0.1779 0.4425 0.7330  1377361 0.0632
 1000    1   0.0  0.1762 0.4409 0.7075   423616 0.0618
 1000    1  0.01  0.1762 0.4409 0.7075   423616 0.0618
 1000    1  0.05  0.1762 0.4409 0.7075   423616 0.0618
 1000    1  0.15  0.1762 0.4409 0.7075   423616 0.0618

60 points
```

## Verdict

⛔ **Keep `--topk 3000 --ml-k 3 --ml-floor 0.05`.** Best of 60 is 0.1974
(topk 8000, ml-k 5 or 10) — **+0.7%** for **2.8× the detections**. The decode
lever is closed.

⚠⚠ **`--topk` is part of the recipe and the plan did not say so.** Default is
1000; at the default this checkpoint scores **0.1919**, not 0.1961. The default
was deliberately left alone — raising it would restate the historical argmax rows
too (0.1762 → 0.1779).

## The affine A/B

`FPN_AFFINE=50 FPN_TAG=aff50 FPN_EPOCHS=12`, 49 min on one 4060 Ti, scored at the
canonical decode. `runs/2026-09-01-aff50/` (train), `-aff50-score` / `-aff50-e10`
/ `-aff50-e8` (logits).

| arm | mAP@0.5 | ca-AP | recall |
|---|---|---|---|
| `aff50` e8 | 0.1473 | 0.3858 | 0.7342 |
| `aff50` e10 | 0.1609 | 0.3904 | 0.7479 |
| `aff50` e12 | 0.1763 | 0.4154 | 0.7498 |
| `cfoc2` e12 (control) | **0.1961** | 0.4414 | 0.7485 |

−10% at the matched budget, but **still rising at e12** where the control peaks —
underfitting, not a failed lever. Epoch checkpoints were scored via the
copy-and-rename dance into `aff50e8` / `aff50e10` tags (`_fwd_eval.mlir` must be
copied alongside `_params.bin` + `_bn_stats.bin` or `infer` hard-fails).

Follow-ons launched the same day: `aff30` (FPN_AFFINE=50, 30 epochs, fresh cosine)
and `aff25` (FPN_AFFINE=25, 12 epochs).
