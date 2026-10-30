"""neff_attr.py NEFFDIR [NEFFDIR...] — static profile of an extracted NEFF: per HLO op type, how many
DMA blocks and how many instructions on each engine (PE, DVE, Pool, Activation, SP) it generated.
Chain: asm_<E>.dbg → backend_<E>.dbg → penguin.dbg → hlo.dbg (parent_ids); DMA blocks → penguin → hlo."""
import sys, re, collections, os
from neuronxcc.proto import ir_debug_info_pb2 as p

def load(f):
    m = p.ir_debug_info(); m.ParseFromString(open(f, 'rb').read()); return m

def profile(d):
    hlo = load(f'{d}/debug_info/debug_info_hlo.dbg_sg000000')
    # id -> (op type, name, shape?) from "%name.N = optype(args)"
    hlo_type = {}; hlo_name = {}
    for ins in hlo.instructions:
        m = re.match(r'%([\w.\-]+) = (\w+)\(', ins.name)
        if m: hlo_type[ins.id] = m.group(2); hlo_name[ins.id] = m.group(1)
        else: hlo_type[ins.id] = '<' + ins.name.split()[0][:20] + '>'; hlo_name[ins.id] = ins.name[:30]
    peng = load(f'{d}/debug_info/debug_info_penguin.dbg_sg000000')
    peng_hlo = {ins.id: list(ins.parent_ids) for ins in peng.instructions}   # penguin id -> hlo ids
    def hlo_of_peng(pid):
        hs = peng_hlo.get(pid, [])
        return hs[0] if hs else None
    tab = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    # DMA blocks
    dma = load(f'{d}/sg00/debug_info_backend_DMA.dbg')
    for ins in dma.instructions:
        h = None
        for pid in ins.parent_ids:
            h = hlo_of_peng(pid)
            if h is not None: break
        t = hlo_type.get(h, '?') if h is not None else '?'
        if t == '?':
            mm = re.match(r'([a-z_]+)\.\d+', ins.name)
            t = '?' + mm.group(1) if mm else '?'
        tab[t]['dma'] += 1; tot['dma'] += 1
        q = ins.attrs.get('dma_queue', '')
        if 'Spill' in q or 'spill' in q: tab[t]['dma_spill'] += 1; tot['dma_spill'] += 1
    # engines: asm -> backend -> penguin -> hlo
    for eng in ['PE', 'DVE', 'Pool', 'Activation', 'SP']:
        bf = f'{d}/sg00/debug_info_backend_{eng}.dbg'; af = f'{d}/sg00/debug_info_asm_{eng}.dbg'
        if not (os.path.exists(bf) and os.path.exists(af)): continue
        back = load(bf)
        back_peng = {ins.id: list(ins.parent_ids) for ins in back.instructions}
        asm = load(af)
        for ins in asm.instructions:
            h = None
            for bid in ins.parent_ids:
                for pid in back_peng.get(bid, []):
                    h = hlo_of_peng(pid)
                    if h is not None: break
                if h is not None: break
            t = hlo_type.get(h, '?') if h is not None else '?'
            tab[t][eng] += 1; tot[eng] += 1
    return tab, tot

cols = ['dma', 'dma_spill', 'PE', 'DVE', 'Pool', 'Activation', 'SP']
for d in sys.argv[1:]:
    tab, tot = profile(d)
    print(f'\n## {d}')
    print(f'{"hlo op":<22}' + ''.join(f'{c:>11}' for c in cols))
    print(f'{"TOTAL":<22}' + ''.join(f'{tot[c]:>11}' for c in cols))
    for t, c in sorted(tab.items(), key=lambda kv: -(kv[1]['dma'] * 50 + sum(kv[1][e] for e in cols[2:])))[:22]:
        print(f'{t:<22}' + ''.join(f'{c[k]:>11}' for k in cols))
