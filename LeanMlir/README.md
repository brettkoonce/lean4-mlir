# `LeanMlir/` — the library

Everything the trainers and the proofs import. The proofs live in [`Proofs/`](Proofs/); start
with [its README](Proofs/README.md), whose "Start here" section reads the linear classifier
end to end in about 650 lines.

| modules | what they are |
|---|---|
| [`Proofs/`](Proofs/) | every theorem: per-layer VJPs, whole-net backward passes, and the ties from each committed render to the certified math |
| `Types`, `Spec`, `SpecHelpers` | the `NetSpec` DSL: layers, specs, `TrainConfig`, parameter counting |
| `VerifiedSpec`, `VerifiedNetsCore`, `VerifiedNets`, `VerifiedTrain`, `ParamLayouts`, `ViTRender` | the verified path: the nets the proofs are about, and the driver that trains them on the committed renders in [`verified_mlir/`](../verified_mlir/) |
| `MlirCodegen`, `Train`, `ReferenceNets` | the reference path: `NetSpec` → StableHLO at run time, unverified; the BraTS UNet and the ablations run on it |
| `IreeRuntime`, `F32Array`, `LEBytes`, `MnistData` | the runtime: bindings to [`ffi/`](../ffi/) (XLA/PJRT by default, IREE optionally) and host-side buffers |
| `SyncBnCheck`, `GradcheckHelpers`, `VjpOracleNets` | support for the gates in [`tests/`](../tests/) |
| `VerifiedAttack`, `VerifiedPgdGen`, `VerifiedSmoothing`, `E4M3Quant` | the robustness and fp8 studies |
| `Blackjack`, `Cam`, `Ddpm` | support for the Chapter 10 demos |

```bash
lake build LeanMlir    # the library
lake build Certs       # every certificate CI checks
```
