#!/bin/bash
# neff_id.sh CACHEDIR... — one line per cache entry: args, bf16?, MAC, traffic, engine stream sizes
for c in "$@"; do for m in $c/neuronxcc-*/MODULE_*; do
  pb=$m/model.hlo_module.pb; [ -f $m/model.neff ] || continue
  args=$(grep -m1 -o 'func.func @main([^{]*' $pb | grep -o 'tensor<' | wc -l)
  bf=$(grep -c 'xbf16>' $pb); conv=$(grep -c 'stablehlo.convolution' $pb); dots=$(grep -c 'dot_general' $pb); sas=$(grep -c 'select_and_scatter' $pb); ar=$(grep -c 'all_reduce' $pb)
  x=$(grep -m1 -o 'func.func @main(%arg0: tensor<[0-9x]*' $pb | sed 's/.*tensor<//')
  st=$(dd if=$m/model.neff bs=1024 skip=1 2>/dev/null | tar xzOf - hlo_stats.json 2>/dev/null | python3 -I -c "import json,sys; s=json.load(sys.stdin); print(f\"MAC={s['HloMacCount']/1e9:.1f}G traffic={s['Traffic']/1e6:.0f}MB\")")
  sz=$(dd if=$m/model.neff bs=1024 skip=1 2>/dev/null | tar tzvf - 2>/dev/null | awk '{n=$6; s=$3} n~/PE0.bin/{pe=s} n~/DVE0.bin/{dve=s} n~/Activation0.bin/{act=s} n~/Pool0.bin/{pool=s} n~/SP0.bin/{sp=s} n~/SP0.json/{spj=s} n~/backend_DMA/{dma=s} END{printf "PE=%.1f DVE=%.1f Act=%.1f Pool=%.1f SP=%.1f SPjson=%.1f DMAdbg=%.1f MB", pe/1e6,dve/1e6,act/1e6,pool/1e6,sp/1e6,spj/1e6,dma/1e6}')
  echo "$(basename $c)/$(basename $m | cut -c1-28) args=$args arg0=$x bf16lines=$bf conv=$conv dots=$dots sas=$sas allreduce=$ar $st $sz"
done; done
