# Trainium first session — trn1.2xlarge, 2026-10-29/30 (Neuron SDK 2.33.0)

`i-07ab710f38e5acdcf`, us-east-2c, Deep Learning AMI Neuron JAX 0.10 (Ubuntu 24.04), compile
cache warmed on atlas (22 graphs, deploy/TRAINIUM.md §3). Logs: `../2026-10-29-trainium-setup/`
(check → suite), `setup.log` here (lean → mnist → cifar), the suite in
`../platform/2026-10-29-ip-172-31-36-145-neuron/`.

| stage | result |
|---|---|
| check, venv, jax | pass |
| smoke | failed first: the plugin answers ToHostBuffer's dst=NULL size query with 0. `host_size` in `ffi/pjrt_ffi.c` falls back to dims × element size; then pass |
| suite | 19 pass, 7 fail, 6 skip (the SKIPs are the compiler's instruction cap, TRAINIUM.md §2) |
| mnist | linear 92.10, MLP 97.82, CNN 98.73 at e12 (RTX 4060 Ti, `runs/2026-09-01-*-verified-xla-cuda/`: 92.10 / 97.81 / 98.77); CNN 7.5 s/epoch |
| cifar | cifar8w SGD: loss flat at 2.302, epoch 1 = 10.00 % (Arc B50 2026-10-22: 30.88 %); stopped there |
| imagenette ViT-Tiny | one epoch: 34.62 % val, CI 33.15–36.13 (4060 Ti 2026-08-12: 35.18 %); **1,281 ms/step** steady state, 1,270 of it in the invoke, 0 waiting on the feed (`../2026-10-30-trainium-imagenette/vit-probe.tsv`) — the book's 4060 Ti row is 61 ms/step, so ~21× |
| imagenette ResNet-34 | `adambf16` (the f32 render fails `neuronx-cc`, TRAINIUM.md §2) compiles, then fails to load: `Descriptor limit reached … 16777200` on ring `qSPSpillReload0_2`, `create_data_refill_rings() error` — the batch-32 step spills past the runtime's DMA descriptor cap. Wants a smaller per-call batch, which is a new render |

## The fault: Neuron's `select_and_scatter`

`neuron_bisect.py` runs each op family on the neuron device and on JAX's CPU, through JAX's own
client — no shim — then the whole `cifar8_train_step` the same way:

```
OK  conv fwd                     rel err 2.5e-07
OK  conv grad (dx, dw)           rel err 2.7e-07 1.3e-06
OK  maxpool fwd                  rel err 0.0e+00
BAD maxpool grad (sel&scat)      rel err 1.2e+00
OK  relu grad                    rel err 0.0e+00
BAD conv+pool grad               rel err 1.0e+00 6.1e-01
OK  reduce sum / transpose / reverse
cifar8_train_step: outputs 0–15 (the convs, upstream of a pool) rel err 1.7–53;
                   outputs 16–21 (the dense head) 2–4e-07
```

`sas_small.py`: at 8×8×8×8 every window's gradient lands on its top-left element, not its max
(the total is right); at 128×16×32×32 the gradient lands on 1,193,084 positions for 524,288
windows and the total is wrong; at 1×1×4×4 the compiler fails (`INTERNAL`). The MNIST CNN's one
pool trains to the 4060 Ti's number, so the fault is shape-dependent. The suite's cifar8 and BN
failures are this and the second fault below; the ViT bf16 recipes (median 3.3e-2 against 1e-2) are a separate miss, and
`resnet34_adambf16_train_step` failed in the runtime (`create_data_refill_rings() error`).

## The second fault: an upsample feeding a transposed conv (the one to report upstream)

A `select_and_scatter`-free max-pool backward (`sas_rewrite.py`: four strided slices, a row-major
first-max mask, four interior pads summed — `maxPool3s2BackText`'s recipe at 2×2) is bit-identical
to `select_and_scatter` on XLA:CPU for cifar8, cifar8-BN-Adam and the MNIST CNN, and correct on
Neuron in isolation. cifar8 on Neuron is still wrong with it. Bisecting the rewritten graph
(`cut.py`, `multi.py`, `slice.py`: return one value, return several, extract a value's backward
slice with real arguments) narrowed it to conv 8's weight gradient; reducing the 59-op slice by
hand (`slices/`) left six ops. **`neuron_conv_upsample_repro.py`** is the plain-JAX repro — random
data, no shim, no verified graph:

```
x: (N, C, H, W)   g: (N, C, H/2, W/2)
u = repeat(repeat(g, 2, axis=2), 2, axis=3)                 # nearest-neighbour 2× upsample
w = conv(transpose(x, (1,0,2,3)), transpose(u, (1,0,2,3)), pad 1)   # (C, C, 3, 3): the weight-gradient form

WRONG N=128 C=32 H=W= 4  rel err 7.10e-01
WRONG N=128 C=32 H=W= 8  rel err 7.32e-01
WRONG N=128 C=16 H=W=16  rel err 5.64e-01
WRONG N= 32 C=32 H=W= 4  rel err 7.13e-01
WRONG N=  8 C= 8 H=W= 4  rel err 7.06e-01
ok    N=128 C=32 H=W= 2  rel err 1.47e-07
control (the same u computed on the host and passed in): rel err 3.71e-07
```

`alt.py`, `barrier.py`: an `optimization_barrier` after the upsample or after the transpose does
not help, nor does letting the conv's dimension numbers do the transposes (`CNHW`/`IOHW`); the
upsample as four interior pads summed is wrong at 4×4 and right at 16×16; **the upsample as a
`dot_general` with a 0/1 matrix is right at both**. A rewrite built that way (v2 in
`sas_rewrite.py`) is still bit-identical on XLA:CPU, but `neuronx-cc` did not finish compiling
cifar8 with it in 42 min (the original graph: 17 s), so it is not a workaround yet.
