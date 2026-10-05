# 2026-10-05 — the shim's uint8 wire

Plan: `planning/side_quest_runs.md` §4.5. Code: `jax/Jax/Codegen.lean` (`emitDataLoading … (u8Wire)`,
the shim's `_U8_WIRE` / `_U8_ACTIVE`, wire v5/v6), `ffi/f32_helpers.c` (`lean_mlir_u8_norm`),
`LeanMlir/Verified/Train.lean` (`u8Norm`, `ShimProc.norm`, `LEAN_MLIR_SHIM_U8`).

| file | what |
|---|---|
| (not kept) | the first gate run was **VACUOUS** — the trace-time test wanted a uint8 tensor and RandAugment returns `tf.cast(<uint8>, tf.float32)`, so no producer granted the wire and the gate refused to pass; the re-run overwrote its `tie.out` |
| `tie.out`, `tie2.out` | `tests/u8_wire_tie.sh` PASS: control 0 bytes, float vs uint8 0 bytes, 4/4 producers on the uint8 wire — scalar kernel, then the AVX2 build |
| `normbench*.c` | the C normalize alone: channel-major 65 ms, pixel-major 57, AVX2 15 per 512 at 224²; generic and AVX2 bit-identical over every byte value in every channel |
| `probe.scalar.out`, `v_mnv4-full-4gpu.scalar.log` | MNv4 verified §3a smoke, scalar kernel: mean 293 / median 294 / floor 244 |
| `probe.out`, `v_mnv4-full-4gpu.log` | the same with the AVX2 kernel: **mean 290** / median 290 / floor 233 (318 before, 329 before the shim threading fix) |
| `identity_float.txt` | the float wire unchanged: old vs new SHIM_HASH and streamed bytes, five shims |
| `recovered/` | the 09-28 prototypes rebuilt from session `09ff3375`'s transcript (`RECOVERED.md`): the JAX-trainer uint8 + device normalize probe (−13% under the stall), + PIL Rotate (−28%), the augbench harness, the blocked-mixup prototype. JAX side only; the verified-path wire here is new |

MNv4 stays producer-CPU-bound (57 ms of a 290 ms step starved): what remains is TF's per-image
augmentation, the §4.3 L4/L5 levers (PIL Rotate — unchecked against the generated op — and the
warps).
