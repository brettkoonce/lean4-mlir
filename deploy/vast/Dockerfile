# deploy/vast/Dockerfile — the bench box as an image: Ubuntu 24.04 plus this tree BUILT
# (the pinned CUDA 12 venv, the PJRT shim, elan, Mathlib's cache, the verified ImageNet trainers,
# the JAX trainers synced into .lake/build). On vast, the instance's on-start command runs
# deploy/vast/onstart.sh, which starts deploy/vast/bench.sh once. No ImageNet in it: the boot run is the
# device-only half (stack check, peak memory and bare-card step per net); with imagenet2012 under
# $TFDS_DATA_DIR it goes on to the loader and the fed rows.
#
#     deploy/vast/build.sh                       # docker build from the repo root; prints the push line
#
# The build needs no GPU. CUDA comes from the venv's nvidia-*-cu12 wheels (12.9), which run on any
# newer driver, and the driver comes from the host through the NVIDIA container toolkit, so the base
# is plain Ubuntu 24.04 (python 3.12). The two NVIDIA_* variables are what the toolkit reads to mount
# the driver and nvidia-smi into a container whose base is not NVIDIA's.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ENV LEAN4MLIR=/opt/lean4-mlir \
    ELAN_HOME=/opt/elan \
    PATH=/opt/elan/bin:${PATH} \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates git curl rsync openssh-server python3 python3.12 python3.12-venv \
      bsdextrautils util-linux \
    && rm -rf /var/lib/apt/lists/*

# ── the venv, from the lock alone, so a source edit does not reinstall 8 GB of wheels.
#    The iree rc in the lock is not on PyPI; iree.dev's index has it.
WORKDIR ${LEAN4MLIR}
COPY jax/requirements-cuda-lock.txt jax/requirements-cuda-lock.txt
RUN python3.12 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -q --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -q --find-links https://iree.dev/pip-release-links.html \
         -r jax/requirements-cuda-lock.txt \
    && test -f .venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so

# ── elan and the toolchain lean-toolchain names
COPY lean-toolchain lean-toolchain
RUN curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
      | sh -s -- -y --no-modify-path --default-toolchain "$(cat lean-toolchain)" \
    && lake --version

# ── the tree (deploy/vast/Dockerfile.dockerignore keeps .git, .lake, .venv, data and runs out)
COPY . .

# ── the shim and its smoke test; the trainers; the JAX side's generated files
ARG TRAINERS="resnet50-imagenet-verified vit-s-imagenet-verified vit-b-imagenet-verified convnext-s-imagenet-verified convnext-b-imagenet-verified vit-ema-drop-render"
RUN gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so \
    && gcc -O2 ffi/test_pjrt_smoke.c -ldl -lm -o ffi/test_pjrt_smoke
RUN ulimit -n 65536 && lake exe cache get \
    && for t in ${TRAINERS}; do echo "▸ lake build $t"; lake build "$t" || exit 1; done \
    && bash scripts/regen_jax_generated.sh sync

ARG GIT_REV=unknown
RUN echo "${GIT_REV}" > .image_rev

WORKDIR ${LEAN4MLIR}
CMD ["bash"]
