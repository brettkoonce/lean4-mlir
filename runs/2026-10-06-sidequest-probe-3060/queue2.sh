#!/bin/bash
# Follow-up after queue.sh: ConvNeXt-B JAX OOMed at JAX's default 0.75 preallocation (a 5.98 GiB
# request on a 12 GB card); retry the same 12-min window at XLA_PYTHON_CLIENT_MEM_FRACTION=0.95.
set -u
D=$(cd "$(dirname "$0")" && pwd)
until grep -q '^done' "$D/queue.out" 2>/dev/null; do sleep 30; done
# reuse queue.sh's functions without running its body
eval "$(sed -n '/^set -u/,/^echo "start/p' "$D/queue.sh" | sed '$d')"
echo "start $(date -u)"
echo "jax cnxb_f95 $(date -u)"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 jaxrun cnxb_f95 .lake/build/generated_convnext_b_imagenet.py
echo "done $(date -u)"
