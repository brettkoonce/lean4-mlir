# ═══════════════════════════════════════════════════════════════════
# lean4-mlir RunPod ROPE image — test the RoPE / long-context config
#
# Sibling of docker/runpod-probe.Dockerfile (JAX benchmark). This one is
# for the verified Lean→MLIR→IREE→CUDA codegen path — the one that carries
# FlashAttention + RoPE + no-absolute-position (the 8K story). Same template:
# lean image, heavy toolchain bootstrapped ON the pod (where the GPU + full
# CUDA env live), everything under /root.
#
# Build (from repo root):
#   docker build -f docker/runpod-rope.Dockerfile -t lean4-mlir-rope .
# Push:
#   docker tag lean4-mlir-rope <user>/lean4-mlir-rope:v1
#   docker push <user>/lean4-mlir-rope:v1
# RunPod: template → Container Image = <user>/lean4-mlir-rope:v1, then
#   from the pod shell:
#     rope-test          char-level nano-rope-nopos: train + length extrapolation
#                        (train T=64, eval T=128/256) vs the length-locked absolute
#                        model. First run bootstraps the Lean/IREE pipeline (~10-20
#                        min: elan + iree + runtime build + mathlib oleans); later
#                        runs skip to the test. `rope-test 200` for a quicker pass.
#     ./run_tinystories_8k.sh   the full 8K model (flash+rope+no-pos, ~29M params),
#                        once data/tinystories/{train,val}.bin exist (BPE prep).
#
# Provision: >=50 GB container disk (mathlib oleans 6.9G + IREE build ~2G +
# project build ~3G). A100 = sm_80 (the default IREE_CHIP the test sets).
# runpod/base ships the CUDA toolkit the on-pod IREE-CUDA runtime build needs.
# NOTE: the image clones github.com/brettkoonce/lean4-mlir@${REPO_REF} — that
# branch must carry the flash/rope/posEmb commits (this session's work).
# ═══════════════════════════════════════════════════════════════════

FROM runpod/base:1.0.7-ubuntu2404

# ── 1. build deps for the on-pod Lean/IREE bootstrap (elan + IREE runtime
#      from source: cmake/ninja/gcc; the JAX image needed none of these) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3-venv build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

# ── 2. the repo (changes often → last-ish layer; bump REPO_SHA to bust cache) ──
ARG REPO_REF=main
ARG REPO_SHA=unset
RUN git clone -b ${REPO_REF} --depth 1 \
      https://github.com/brettkoonce/lean4-mlir.git /root/lean4-mlir

# ── 3. bake tinyshakespeare (~1 MB) so `rope-test` is self-contained + offline ──
RUN cd /root/lean4-mlir && bash download_shakespeare.sh && \
    python3 preprocess_shakespeare.py

# ── 4. `rope-test` from anywhere; elan on PATH for interactive shells ──
RUN printf '#!/bin/bash\ncd /root/lean4-mlir && exec bash scripts/rope_test.sh "$@"\n' \
      > /usr/local/bin/rope-test && chmod +x /usr/local/bin/rope-test && \
    printf 'export PATH=/root/.elan/bin:/root/lean4-mlir/.venv/bin:$PATH\nexport IREE_CHIP=sm_80\n' \
      >> /root/.bashrc
