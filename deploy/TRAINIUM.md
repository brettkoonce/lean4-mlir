# AWS Trainium — the verified trainers on a NeuronCore

**Status: in preparation (2026-10-26). Nothing has run on a Trainium device yet; the account's
Trn quota (8 vCPU, one trn1.2xlarge, us-east-2) is with AWS.** What is on record is everything
that needs no device. On an ordinary x86 box Neuron's compiler, `neuronx-cc`, compiles every
graph the MNIST and CIFAR tiers run for trn1 (§2), and under Neuron's fake device our shim creates
a real Neuron client, at PJRT C-API 0.90, and the plugin compiles our graphs through its own path
into its compile cache (§3) — so the instance can start with every graph compiled. Four of the
suite's ten Imagenette train steps compile at batch 32; the depthwise nets and ConvNeXt are over
the compiler's instruction cap (§2).

The trainers reach an accelerator through a PJRT plugin that the shim (`ffi/pjrt_ffi.c`)
`dlopen`s, so Trainium needs Neuron's plugin and nothing else from this repo's side: Neuron
SDK 2.33.0's `libneuronxla` 3.0.10135, whose `libneuronpjrt.so` exports `GetPjrtApi`, with its
compiler `neuronx-cc` 2.28 as a pip package and its runtime `libnrt.so.1` from apt. Neuron's own
JAX tops out at 0.10.0 (the other platforms are on 0.11); only the `jax` rung uses it.

The box: trn1.2xlarge — one Trainium1 chip (two NeuronCore-v2, 32 GB HBM), 8 vCPU, 32 GB RAM,
~$1.34/h on demand. In this account EC2 runs only in us-east-2 (Ohio), and trn1 only in its zone
us-east-2c.

## Requirements

| | |
|---|---|
| instance | trn1.2xlarge in us-east-2c, 100 GB root volume (the image's default 35 GB is too small for Mathlib's cache and the build) |
| image | **Deep Learning AMI Neuron JAX 0.10 (Ubuntu 24.04), SDK 2.33.0** — `ami-001bf438e94bac85d` (2026-10-20), or the SSM parameter `/aws/service/neuron/dlami/jax-0.10/ubuntu-24.04/latest/image_id`. It has the driver, the runtime and the tools; a plain Ubuntu 24.04 needs §1's apt packages |
| runtime | `aws-neuronx-dkms`, `aws-neuronx-runtime-lib` (`libnrt.so.1`), `aws-neuronx-collectives`, `aws-neuronx-tools` (`neuron-ls`) |
| tools | gcc, Python 3.11+, elan |
| disk | ~25 GB: the Neuron venv (~3 GB), Mathlib's prebuilt cache, the build, the compile cache |

## Launch

From a box with the account's CLI login (`aws login`), default region us-east-2. The key pair
`atlas-ed25519` and the security group `ssh-atlas` (port 22 from atlas only) already exist.

```bash
aws ec2 run-instances --instance-type trn1.2xlarge --image-id ami-001bf438e94bac85d \
  --placement AvailabilityZone=us-east-2c --key-name atlas-ed25519 \
  --security-group-ids sg-02bec1f565d318877 \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=trn1-lean4-mlir}]'
ssh ubuntu@<public ip>
```

**Stop it when idle** (`aws ec2 stop-instances --instance-ids <id>`): stopped, it costs only its
volume (~$8/month at 100 GB) and keeps the venv, the build and the compile cache. The account's
budget (`lean4-mlir-monthly`, $30/month) only emails; it stops nothing.

## Run it

```bash
git clone https://github.com/brettkoonce/lean4-mlir && cd lean4-mlir
bash deploy/trainium_setup.sh                 # every rung, in order
bash deploy/trainium_setup.sh smoke           # resume from a rung after fixing it
bash deploy/trainium_setup.sh venv aot        # only these rungs — both run on any x86 box
```

| rung | proves | when it fails, the problem is in |
|---|---|---|
| `check` | `/dev/neuron0` under the driver, `neuron-ls`, `libnrt.so.1`, gcc, Python, elan, disk; how many compiled graphs `/var/tmp/neuron-compile-cache` holds (§3) | the box: the image, Neuron's apt packages |
| `venv` | `.venv-neuron` with `jax-neuronx` 0.10.0.1.0.14198, `libneuronxla` 3.0.10135, `neuronx-cc` 2.28.7631, `jax` 0.10.0 — never the pinned `.venv` | pip / Neuron's index |
| `aot` | by name only, on any x86 box: `neuronx-cc` compiles the suite's 16 tier-2 graphs for trn1 on the CPU (§2) | the compiler's op coverage and limits, before any device |
| `jax` | JAX computes on the `neuron` device, with `JAX_PLATFORMS=neuron` so it cannot fall back to the CPU quietly | the plugin itself, independent of this repo |
| `smoke` | `ffi/test_pjrt_smoke.c` through the real shim on one NeuronCore: create the client, run `verified_mlir/linear_fwd.mlir` against the host's answer, compile the other five MNIST renders | the shim ↔ plugin seam |
| `suite` | `scripts/platform/check.sh --backend neuron`, tiers 0–2, both NeuronCores visible so tier 1's two-device tests run | the plugin's numerics, op by op |
| `data` | MNIST into `data/mnist/`, CIFAR-10 into `data/` | the network |
| `lean` | elan installs the toolchain, `lake exe cache get` fetches Mathlib, and the five MNIST and CIFAR trainers build — only those, on 8 vCPU | the Lean build |
| `mnist` | `lake run mnist`: linear, MLP and CNN on a NeuronCore | the trainers on Neuron |
| `cifar` | `lake run cifar`: the six optimizer arms | sustained conv training |

Everything is teed to `runs/<date>-trainium-setup/setup.log`, with the instance type, Neuron
package versions and plugin checksum in `manifest.txt` beside it. The `aot` rung writes
`runs/platform/<date>-<host>-neuron-aot-trn1/`. Imagenette is not a rung (§2).

## What is Neuron-specific

- **`scripts/platform/env.sh`**'s `neuron` arm is picked by `/dev/neuron0` on the box, or by
  `PLATFORM_BACKEND=neuron` anywhere. It points `PJRT_PLUGIN` at
  `.venv-neuron/…/libneuronxla/libneuronpjrt.so` (else a DLAMI's `/opt/aws_neuronx_venv_jax*`),
  puts that venv's `bin/` on `PATH` and `/opt/aws/neuron/lib` on `LD_LIBRARY_PATH`, and exports:
  - **`NEURON_INTERNAL_PJRT_C_API_VERSION=0.90`.** The plugin serves PJRT C-API 0.70 to 0.104 and
    picks one from this variable (libneuronxla's `version.py` maps each jaxlib to one); 0.90 is
    `ffi/pjrt_c_api.h`'s. Without it the plugin warns and serves 0.104. The `jax` rung unsets it,
    so jaxlib 0.10.0 gets its own 0.104.
  - **`NEURON_CC_FLAGS=--auto-cast=none`.** By default `neuronx-cc` casts f32 matmuls to bf16; the
    suite's f32 tolerances would report that as a wrong answer, and a trainer would silently run
    a different precision from every other platform.
- **The plugin compiles by running `neuronx-cc` from `PATH`**, as a separate process per graph,
  and caches the result under `/var/tmp/neuron-compile-cache` (`NEURON_COMPILE_CACHE_URL`). The
  first run of a trainer pays every compile; later runs read the cache.
- **`run.sh`** accepts the `neuron` backend; the GPU index becomes `NEURON_RT_VISIBLE_CORES`, a
  NeuronCore. A trn1 chip is two of them.
- **`ffi/pjrt_ffi.c` is unchanged.** It recognises no Neuron platform name and applies none of
  the Arc's serial-copy workarounds; whether the plugin needs any is for the `suite` rung to say.

## §1 The box

The DLAMI above has everything. On a plain Ubuntu 24.04 (Neuron's manual setup page):

```bash
. /etc/os-release
sudo install -m 0755 -d /etc/apt/keyrings
echo "deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com $VERSION_CODENAME main" \
  | sudo tee /etc/apt/sources.list.d/neuron.list
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
  | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/neuron.gpg
sudo apt-get update
sudo apt-get install -y linux-headers-$(uname -r) aws-neuronx-dkms aws-neuronx-runtime-lib \
  aws-neuronx-collectives aws-neuronx-tools
```

## §2 Before any Trainium hour: the compiler on the CPU

`neuronx-cc` runs on any Linux x86_64 box. Neuron's plugin hands it the program as
`model.mlir` (`libneuronxla/neuron_cc_wrapper.py`), so `scripts/platform/neuron_aot.sh` does the
same with each render — the shim's rename of the entry to `@main`, `--auto-cast=none` — and
records whether it compiled, how long it took and the NEFF's size:

```bash
bash deploy/trainium_setup.sh venv aot                          # the suite's 16 graphs
scripts/platform/neuron_aot.sh cnn_train_step cifar8w_bn_adam_train_step   # any renders
NEURON_AOT_TARGET=trn2 scripts/platform/neuron_aot.sh mobilenetv2_adam_train_step
```

A pass says the compiler's op coverage and lowering accept the graph for that target. It does
not say that the plugin's own passes accept it, or that the NEFF computes the right numbers:
that is the `smoke` and `suite` rungs.

Measured on atlas (2× Xeon E5-2699 v4, 88 threads, 20 compiles at a time; Neuron SDK 2.33.0;
2026-10-26): **27 of 33 renders compile for trn1**
([`runs/platform/2026-10-26-atlas-neuron-aot-trn1/`](../runs/platform/2026-10-26-atlas-neuron-aot-trn1/results.tsv)),
the same 27 as a first run the day before.

| set | renders | compile for trn1 | each |
|---|---|---|---|
| MNIST: linear, MLP, CNN — forward and train step | 6 | **all 6** | 3–17 s |
| CIFAR: `cifar8w`'s six arms and two forwards, the suite's four `cifar8` graphs | 12 | **all 12** | 1–3 min |
| Imagenette forwards: ResNet-34, MobileNetV2, EfficientNet-B0, ViT-Tiny, ConvNeXt-T | 5 | **all 5** | 1–3 min; ConvNeXt 12 min |
| Imagenette train steps, batch 32 at 224² (the suite's tier 2) | 10 | **4**: ResNet-34 bf16 (21 min), ViT-Tiny f32 and both bf16 recipes (33–39 min) | — |

So every op the nets use lowers — every forward compiles — and what fails is the size of a
whole train step (forward, backward and the Adam update in one program) at batch 32:

| train step | error | |
|---|---|---|
| MobileNetV2 f32, bf16 (`mobilenetv2in_adam64bf16`) | `NCC_IXTP002` | 5,078,840 and 5,360,207 instructions, against a cap of 5,000,000 |
| EfficientNet-B0 f32, bf16 | `NCC_IXTP002` | 5,123,079 and 5,123,072 |
| ConvNeXt-T f32 | `NCC_EBVF030` | 9,173,277, after an hour |
| ResNet-34 f32 | `NCC_IBIR243` | the compiler's own IR verifier fails on a `maximum` (the max-pool) after 19 min; the bf16 render of the same net compiles |

Neither knob tried moves the cap: `--model-type=generic` gives MobileNetV2 the same 5,078,840,
`-O1` makes it 14,108,624 (EfficientNet 16.7M), and trn2 is over too (MobileNetV2 6.0M,
EfficientNet 6.4M; [`…-trn2/`](../runs/platform/2026-10-26-atlas-neuron-aot-trn2/results.tsv)).
The compiler's own advice is "smaller batches". The renderers already have the shape for that —
gradient accumulation, `acc<k>x<B>` (k micro-batches of B per step, `ResNet50RenderB.lean`) — so an
`acc2x16` render keeps batch 32's update with half the program per call; its BN statistics are
over 16 images, not 32. Not rendered yet: new renders are new `#eval` lines in the proven
renderers. The six are `SKIP` in `scripts/platform/expected/neuron-trn1-2xlarge.txt`, so the
suite does not spend hours compiling them to fail.

## §3 Warming the compile cache before the instance exists

Neuron's runtime has a fake device: with `NEURON_RT_FAKE_INSTANCE_TYPE=trn1.32xlarge` (the
runtime does not fake a 2xlarge; both are `--target=trn1`) the plugin creates a client on a box
with no Neuron hardware, compiles each graph into its cache, and fails only when it loads the NEFF
onto the fake NeuronCore (`NRT_INVALID … lnc_idx=4294967295`). `scripts/platform/neuron_warm.sh`
drives that through the shim's `iree_ffi_session_create` — the trainers' and the suite's own entry
point, under `env.sh`'s profile — so each entry's key, `MODULE_<graph hash>+<flags hash>`, is the
one the instance computes; the flags are part of it (`linear_fwd` without `--auto-cast=none` is a
different entry). It fetches the runtime and collectives libraries from Neuron's apt repository,
extracted under `~/.cache/neuron-warm/`, never installed.

```bash
scripts/platform/neuron_warm.sh              # the MNIST and CIFAR graphs + tier 2, less the SKIPs
scp /var/tmp/neuron-compile-cache.tar.gz ubuntu@<ip>:
ssh ubuntu@<ip> tar -C /var/tmp -xzf neuron-compile-cache.tar.gz
```

More cores barely help the compiler: pinned to 4 cores (8 threads, the instance's shape) on atlas,
MNIST's CNN train step took 18 s against 17 s with 88 threads, CIFAR's BN-Adam 171 s against 178 s,
ResNet-34 bf16 26 min against 21. So on the instance's 8 vCPU the CIFAR tier's first run would
spend ~20 min compiling, and the suite's ResNet-34 and three ViT graphs half an hour or more each.
Warmed, the plugin logs `Using a cached neff` instead: that line on the instance is the check
that the keys matched.

**Measured (atlas, 2026-10-26): all 22 graphs compiled through the plugin into the cache** —
the MNIST and CIFAR tiers' 14 and the suite's tier 2 less its six SKIPs, the longest ViT's bf16
recipe at 38 min — so the plugin's own passes accept every graph `neuronx-cc` accepted in §2.
The cache packs to 195 MB
([`runs/platform/2026-10-26-atlas-neuron-warm/`](../runs/platform/2026-10-26-atlas-neuron-warm/results.tsv)).

## What to expect

- **Compile time is the first session's cost** unless the cache was warmed (§3). Stop the
  instance rather than terminate it, so `/var/tmp/neuron-compile-cache` survives.
- **Neuron's JAX is beta.** AWS's native path is now PyTorch (TorchNeuron); the JAX plugin we use
  is supported on trn1 in 2.33.0 but is not where the new work is.

## After it runs

Send back `runs/<date>-trainium-setup/` (the log and manifest), the suite's
`runs/platform/<date>-<host>-neuron/` and the `runs/<date>-mnist-*/` and `runs/<date>-cifar8w-*/`
logs. A green `cifar` rung earns the platform a row in `scripts/platform/support.tsv`, and with
it `PLATFORMS.md`; the suite's failures, if any, go in
`scripts/platform/expected/neuron-trn1-2xlarge.txt`.
