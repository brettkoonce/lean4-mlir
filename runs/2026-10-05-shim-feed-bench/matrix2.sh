#!/bin/bash
# After the in-place mix + writer thread (Jax/Codegen.lean): the job shapes again -> results2.txt
D=$(cd "$(dirname "$0")" && pwd); cd "$D/../.."
B="python3 $D/bench.py"
run() { $B "$@" 2>&1 | grep -v '^$' | tail -1 | tee -a "$D/results2.txt"; sleep 10; }
echo "start $(date)" > "$D/results2.txt"
run --shim generated_vit_tiny_imagenet_shim.py --nclasses 1000 --batch 512 --n 4 --label "NEW ViT  512 rr (the job)"
run --shim generated_vit_tiny_imagenet_shim.py --nclasses 1000 --batch 512 --n 4 --mode free --label "NEW ViT  512 free"
run --shim generated_mobilenet_v4_imagenet_full_shim.py --batch 512 --n 4 --label "NEW MNv4 512 rr (the job)"
echo "done $(date)" >> "$D/results2.txt"
