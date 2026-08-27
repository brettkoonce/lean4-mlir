# MobileNetV4 collective ties — 2026-08-27

The DP tie MNv4 never had. Until today `verified_mlir/mnv4in_adamdp64_train_step.mlir` and
its bf16 peer were on disk and **deliberately unquotable**: `MobileNetV4RenderB.lean`'s own
`#eval` block said they existed to answer "how long would a 4× run take", carried
*"do not train off these"*, and named what would lift the caveat — *"a DP tie for MNv4's
collectives, the way R34/R50/MNv2/ConvNeXt have one"*. This is that tie.

⭐ **The point is not that the artifacts were suspect.** It is that an untied collective
artifact looks *exactly* as trustworthy as a tied one — same bytes, same directory, same
naming — so the only thing that could separate them was a gate.

## Results

| gate | what it proves | result |
|---|---|---|
| `mnv4-dp-check` (fp32) | `all_reduce(add)/4` is the identity on a duplicated batch | ✅ `bnstat` **bit-exact 67,904/67,904**, gradient norm-rel **8.45e-7** |
| `mnv4-dp-check` (bf16) | same, for the bf16 pair | ✅ **bit-exact on all 9,715,512 floats** in θ, m, v *and* bnstat |
| `shard-check mnv4in` | the replicas saw **different** rows | ✅ TEST **1.10e-6**, CONTROL **2.00** — 1.8e6× apart |

Both gates green over all 29,214,443 returned floats, 233 params / 9,715,512 scalars,
77 BN layers, 4× RTX 4060 Ti, XLA/PJRT 0.114.

⭐ **The bf16 arm is *stricter* than fp32, which reads backwards and is not.** fp32 leaves
~1e-6 on `m` because the DP module is a different HLO program and XLA may order the backward
reductions differently; bf16 rounds those orderings back onto the same bf16 values, so the
difference disappears. It is a coarser mantissa hiding a real reordering, not a better path.

## The controls

A gate nobody has seen go red is an assertion. Both were run against a deliberately broken
render — every `all_reduce` divisor `4.0 → 1.0`, i.e. **sum instead of mean**, so every
gradient is 4×:

| gate | passing | broken | |
|---|---|---|---|
| `mnv4-dp-check` gradient norm-rel | 8.45e-7 | **2.72** | 6 orders |
| `shard-check` TEST | 1.10e-6 | **3.000000** | 6 orders |

⭐ **`3.000000` is a predicted number, not merely a large one.** Sum-not-mean at 4 replicas
returns `4g` where `g` is correct, so `|4g − g| / |g| = 3` exactly. The gate landing on the
arithmetic the failure mode implies is stronger evidence than a big number would be.

⭐ **`bnstat` stays bit-exact in the broken run**, which is the other half of the control: the
divisor is on the gradient path only, so a forward that moved would have meant the harness was
comparing the wrong regions.

## Reproducing

    runs/2026-08-27-mnv4-dp-shard-gates/run.sh

⚠⚠ **FOUR GPUs, and that is forced.** MNv4 renders `adamdp64` at 4 replicas only — there is no
2-replica peer the way the Imagenette-scale nets have one — so `PJRT_REPLICAS=2` hits the shim's
replica-count guard rather than degrading to a 2-way run. This is also the 1000-class 224² net;
there is no cheaper Imagenette-scale MNv4 DP render to gate against.

⚠ Uses devices `0,2,3,4`, the AER-clean four (idx 1 and 5 throw BadTLP under load —
`reference_ares_pcie_aer`).

## What this unblocks

`scripts/jobs/mnv4-default-4gpu.conf`, and with it MNv4's Track-4 status moving from
*single-device only* to a job name — roughly 63.5 h on one card to a quarter of that on four.
