# ═══════════════════════════════════════════════════════════════════
# lean4-mlir RunPod probe image — JAX benchmark, boot-and-go
#
# Build (from repo root):
#   docker build -f docker/runpod-probe.Dockerfile -t lean4-mlir-probe .
# Test locally (needs nvidia-container-toolkit):
#   docker run --rm --gpus device=0 lean4-mlir-probe bench --eff-steps 3
# Push (your Docker Hub account):
#   docker tag lean4-mlir-probe <user>/lean4-mlir-probe:v1
#   docker push <user>/lean4-mlir-probe:v1
# RunPod: template → Container Image = <user>/lean4-mlir-probe:v1, then
#   from the pod shell:  bench            (R50-A3 rsb-faithful)
#                        bench --net vit-tiny
#                        XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 bench --per-dev 2048
#
# Everything lives under /root (NOT /workspace — RunPod mounts the
# network volume there, which would shadow baked files). The JAX CUDA
# wheels bundle their own CUDA libs, so the slim non-CUDA base is
# enough — only the host driver (injected by RunPod) matters.
# `runpod/base` keeps its own ENTRYPOINT (sshd via PUBLIC_KEY, etc.).
# ═══════════════════════════════════════════════════════════════════

FROM runpod/base:1.0.7-ubuntu2404

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3-venv && \
    rm -rf /var/lib/apt/lists/*

# ── 1. python deps (cache-stable layer): CUDA lock minus the iree pins ──
# (the iree-*rc pins live on iree.dev, not PyPI, and the JAX probes don't
# need them — same trim as the probe README runbook)
COPY jax/requirements-cuda-lock.txt /tmp/lock.txt
RUN python3 -m venv /root/venv && \
    grep -v '^iree' /tmp/lock.txt > /tmp/req.txt && \
    /root/venv/bin/pip install --no-cache-dir -r /tmp/req.txt Pillow && \
    rm /tmp/lock.txt /tmp/req.txt

# ── 2. imagenette baked as preprocessed bins (skips ~1.5 GB download +
#      13k-image JPEG decode on every pod boot) ──
COPY download_imagenette.sh preprocess_imagenette.py /tmp/bake/
RUN cd /tmp/bake && PATH=/root/venv/bin:$PATH ./download_imagenette.sh && \
    mkdir -p /root/imagenette && mv data/imagenette/*.bin /root/imagenette/ && \
    cd / && rm -rf /tmp/bake

# ── 3. the repo (changes often → last layer; bump REPO_SHA to bust cache) ──
ARG REPO_REF=main
ARG REPO_SHA=unset
RUN git clone -b ${REPO_REF} --depth 1 \
      https://github.com/brettkoonce/lean4-mlir.git /root/lean4-mlir && \
    mkdir -p /root/lean4-mlir/data/imagenette && \
    ln -s /root/imagenette/train.bin /root/lean4-mlir/data/imagenette/train.bin && \
    ln -s /root/imagenette/val.bin   /root/lean4-mlir/data/imagenette/val.bin

# ── 4. `bench` from anywhere; venv on PATH for interactive shells ──
RUN printf '#!/bin/bash\ncd /root/lean4-mlir && exec /root/venv/bin/python jax/probe/benchmark.py "$@"\n' \
      > /usr/local/bin/bench && chmod +x /usr/local/bin/bench && \
    echo 'export PATH=/root/venv/bin:$PATH' >> /root/.bashrc
