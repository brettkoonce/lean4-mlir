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
| mnist | linear 92.10, MLP 97.82, CNN 98.73 at e12 (CUDA 2026-10-22: 92.10 / 97.83 / 98.66); CNN 7.5 s/epoch |
| cifar | cifar8w SGD: loss flat at 2.302, epoch 1 = 10.00 % (CUDA 30.88 %); stopped there |

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
pool trains to the CUDA number, so the fault is shape-dependent. The suite's cifar8 and BN
failures are this; the ViT bf16 recipes (median 3.3e-2 against 1e-2) are a separate miss, and
`resnet34_adambf16_train_step` failed in the runtime (`create_data_refill_rings() error`).
