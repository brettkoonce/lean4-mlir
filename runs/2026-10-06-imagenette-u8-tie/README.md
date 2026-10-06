# Imagenette raw-uint8 and streamed loaders: the bit-identity gate, first run

`tests/imagenette_u8_tie.sh`, as landed with the Orin's loaders (`LEAN_MLIR_IMAGENETTE_U8`,
`LEAN_MLIR_IMAGENETTE_STREAM`; deploy/ORIN.md §4). ResNet-34 `adam`, batch 32, one RTX 4060 Ti,
the deterministic shim (`scripts/det_shim.sh`), 2 epochs × 40 steps with eval on every epoch
(3,925 val images, so the last val batch is the padded one). Four runs: f32 twice (control),
raw-uint8 resident, streamed from `train.bin` by shuffled index. Compared: the final [θ|m|v]
blob (`LEAN_MLIR_DUMP_PARAMS`) and the two eval lines, verbatim.

```
── runs ──
   A1  U8=0 STREAM=0  255477624 bytes  loader: f32  eval lines: 2
   A2  U8=0 STREAM=0  255477624 bytes  loader: f32  eval lines: 2
   B   U8=1 STREAM=0  255477624 bytes  loader: u8  eval lines: 2
   C   U8=0 STREAM=1  255477624 bytes  loader: stream  eval lines: 2

── verdict ──
   control  A1 vs A2 (f32, twice)      : 0 differing bytes, eval lines same
   verdict  A1 vs B  (f32 vs uint8)    : 0 differing bytes, eval lines same
   verdict  A1 vs C  (f32 vs streamed) : 0 differing bytes, eval lines same
   loaders  A1=f32  B=u8  C=stream

✓ PASS — control clean; raw-uint8 and streamed loaders bit-identical to f32 over 2 x 40 steps, eval lines equal.
```

Each run is about three minutes under the deterministic shim; the gate is not a speed measurement.
