#!/bin/bash
# The producer bench at the jobs' shape (batch 512 per producer, round-robin) against §4.2's
# batch-128 numbers. CPU only. One line per config -> results.txt.
D=$(cd "$(dirname "$0")" && pwd); cd "$D/../.."
B="python3 $D/bench.py"
V="--shim generated_vit_tiny_imagenet_shim.py --nclasses 1000"
M="--shim generated_mobilenet_v4_imagenet_full_shim.py"
run() { $B "$@" 2>&1 | grep -v '^$' | tail -1 | tee -a "$D/results.txt"; sleep 10; }
echo "start $(date)" > "$D/results.txt"
run $V --batch 128 --n 4 --label "ViT  128 rr (09-28 shape)"
run $V --batch 512 --n 4 --label "ViT  512 rr (the job)"
run $V --batch 512 --n 4 --mode free --label "ViT  512 free"
run $V --batch 512 --n 4 --mix off --label "ViT  512 rr mix off"
run $V --batch 128 --n 4 --mix off --label "ViT  128 rr mix off"
run $V --batch 512 --n 6 --label "ViT  512 rr n6"
run $M --batch 128 --n 4 --label "MNv4 128 rr"
run $M --batch 512 --n 4 --label "MNv4 512 rr (the job)"
run $M --batch 512 --n 4 --mode free --label "MNv4 512 free"
echo "done $(date)" >> "$D/results.txt"
