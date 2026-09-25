# SIGSEGV in `RocmCommandBuffer::LaunchGraph` during steady-state training (gfx1100)

**Status: RESOLVED 2026-08-13 — not a bug in JAX, XLA or ROCm. The system ROCm
userspace was four point releases behind the wheel.** Upgrading `/opt/rocm`
7.2.0 → **7.2.4** fixes it outright. Nothing was filed upstream, and nothing
should be: see "What to file instead" for the one part that is still worth
reporting.

Keep `repro.py`. It is now a regression check: if this ever comes back, it
tells you in 25 seconds.

## The one-line version

A newer `jax-rocm7-plugin` against an older `/opt/rocm` corrupts memory in
steady state. Same soname, different build. Nothing fails at load; it trains
for thousands of dispatches and then dies somewhere unrelated.

## What settles it

Same box, same `repro.py`, same afternoon:

| jax stack | ROCm **7.2.0** | ROCm **7.2.4** |
|---|---|---|
| 0.10.0 + plugin 0.9.1.post4 | 6 of 6 complete | 6 of 6 |
| 0.10.2 + plugin 0.10.2 | **0 of 6** (died epochs 1–11) | **6 of 6** |
| 0.11.0 + plugin 0.11.0 | **1 of 6** | **6 of 6** |

The middle row is the control that closes it. That is the *same venv*,
untouched — not reinstalled, not repinned. It died 11 of 11 before and is
6 of 6 after, with only `/opt/rocm` changed underneath it. One variable.

The verified trainers agree: `mnist-mlp-verified` 6/6 and `mnist-cnn-verified`
3/3 on 7.2.4, both plugins, at CUDA-identical accuracy.

## Where the original diagnosis went wrong

The first pass concluded "confirmed ROCm-specific" from a clean CUDA control
at the identical JAX version. That control was sound and its conclusion was
still wrong, because the *stacks underneath* the two backends were not
comparable: the CUDA box's userspace was current and this one's was not. The
comparison read as "ROCm vs CUDA" while actually being "stale vs fresh."

**The variable nobody moved was the one under `/opt`.** Both plugins resolved
every NEEDED library — `ldd` shows zero "not found" — so nothing pointed at
the userspace. Resolution success is not version compatibility.

Cheap check worth running first, next time: compare `cat /opt/rocm/.info/version`
against the wheel's release date. Ours was seven months apart.

## Reproducing the failure (if you need to)

Requires a ROCm older than the wheel — e.g. `apt/7.2` with plugin ≥ 0.10.2.

```
HIP_VISIBLE_DEVICES=0 python repro.py
```

784-512-512-10 MLP, SGD, batch 128, 468 steps/epoch, 12 epochs, jitted eval
between epochs, random data. Prints the epoch it reached. Two signatures, both
memory corruption surfacing wherever the damaged region is next touched:

* SIGSEGV (139) inside `libhsa-runtime64` via `RocmCommandBuffer::LaunchGraph`
  → `GpuCommandBuffer::Submit` → `CommandBufferThunk::ExecuteOnStream`
* glibc `free(): invalid next size (normal)` abort (134)

`librocprofiler-sdk` appears between HIP and XLA in the backtrace and is a
passenger — `HSA_TOOLS_LIB=` changes nothing. Of every knob tried, only
`--xla_gpu_enable_command_buffer=` moved the survival point, and it did not
fix it.

## The fix

```
sudo sed -i 's|rocm/apt/7\.2 |rocm/apt/7.2.4 |' /etc/apt/sources.list.d/rocm.list
sudo apt update && sudo apt install rocm
```

No reboot: with no `amdgpu-dkms` installed the kernel driver is untouched.
⚠ **7.2.4 is the last release of the classic apt line.** There is no 7.3+;
the numbering jumps to TheRock (7.9.0 preview → 7.14.0), which is a different
distribution, not an upgrade. Going past 7.2.4 means adopting TheRock's whole
userspace.

## What to file instead

One real defect survives this: **the `jax-rocm7-*` wheels declare no minimum
ROCm version and check nothing at load.** They install cleanly, resolve every
library by soname, train for thousands of dispatches, and then corrupt memory.
A metadata constraint or a startup version check would have turned this into
one error message.

That report is hardware-agnostic — it is about which ROCm installations the
wheels support, not about which GPU you own, so "that's an RDNA thing" is not
a responsive answer to it.

## Not covered by this fix

**ViT-Tiny still cannot train on ROCm** and its crash wears one of the two
signatures above, so do not read this page as clearing it. It is a separate,
older fault: see
[`../2026-08-vit-imagenet-rocm-first-step-abort/`](../2026-08-vit-imagenet-rocm-first-step-abort/).

## Environment

- 2× AMD Radeon RX 7900 XTX (gfx1100, RDNA 3), Linux 7.0.0-28-generic, Python 3.12.3
- Broken: ROCm 7.2.0 (7.2.26015) + jax/jaxlib 0.10.2 or 0.11.0 with matching plugin
- Fixed: ROCm 7.2.4 (7.2.53211, MIOpen 3.5.1.70204, rocBLAS 5.2.0.70204) + the same wheels
- CUDA control (clean throughout): RTX 4060 Ti, driver 575.57.08, CUDA 12.9, `jax-cuda12-pjrt` 0.10.2
