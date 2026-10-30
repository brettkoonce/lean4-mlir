#!/usr/bin/env bash
# Warm one compile cache per neuronx-cc flag set on atlas (fake device), in parallel.
# Each line: tag | NEURON_CC_FLAGS (exact string: it is hashed into the cache key) | renders
cd /home/skoonce/lean/lean4-mlir
S=$(dirname "$0"); : > "$S/flag_matrix.tsv"
while IFS='|' read -r tag flags renders; do
  [ -z "$tag" ] && continue
  tag=$(echo $tag); flags=$(echo $flags)
  ( NEURON_CC_FLAGS="$flags" NEURON_WARM_CACHE=/var/tmp/neuron-flags/$tag NEURON_WARM_JOBS=4 \
      scripts/platform/neuron_warm.sh $renders > "$S/warm-$tag.log" 2>&1 ) &
  sleep 20   # neuron_warm picks its output dir by date; stagger so each gets its own
  echo -e "$tag\t$flags\t$(sed -n 's/.*→ //p' "$S/warm-$tag.log")" >> "$S/flag_matrix.tsv"
done <<'M'
O1      | --auto-cast=none -O1                                           | vit_adam_train_step vit_fwd resnet34_adam16_train_step resnet34_fwd_eval resnet34_adam_train_step
O3      | --auto-cast=none -O3                                           | vit_adam_train_step vit_fwd resnet34_adam16_train_step resnet34_fwd_eval
tf32    | --auto-cast=matmult --auto-cast-type=tf32                      | vit_adam_train_step vit_fwd resnet34_adam16_train_step resnet34_fwd_eval
bf16O3  | --auto-cast=matmult --auto-cast-type=bf16 -O3                  | vit_adam_train_step vit_fwd resnet34_adam16_train_step resnet34_fwd_eval
xf      | --auto-cast=none --model-type=transformer                      | vit_adam_train_step vit_fwd
xfbf16  | --auto-cast=matmult --auto-cast-type=bf16 --model-type=transformer | vit_adam_train_step vit_fwd
dp      | --auto-cast=none                                               | vit_adamdp_train_step
dpbf16  | --auto-cast=matmult --auto-cast-type=bf16                      | vit_adamdp_train_step
M
wait
column -t -s $'\t' "$S/flag_matrix.tsv"
for t in $(cut -f1 "$S/flag_matrix.tsv"); do echo "== $t"; tail -3 "$S/warm-$t.log"; done
