"""What each affine knob COSTS in GT on VisDrone — the measurement the defaults
come from, so they are not inherited from a dataset with the opposite size
distribution.

    .venv/bin/python3 scripts/fpn_affine_knob_cost.py

Reports, per setting, the share of encoded positives that survive the transform
and how many of the survivors are under 2 px on a side. Both matter: a box that
survives at 1.5 px is a target the detector cannot possibly hit, so "survival"
alone would make an over-aggressive scale look cheap.

⚠ One structural caveat this cannot measure away. The augmenter works on the
448-px RECORD, not the source image, so translation moves content off the frame
and fills the vacated strip with the dataset mean — it cannot pull in real pixels
from outside the crop the way an affine on the original photo would. Boxes that
leave are therefore pure loss with nothing gained at the opposite edge. That is
what makes translate expensive here relative to its usual reputation.
"""
import ctypes, os, subprocess, sys, tempfile, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from preprocess_visdrone import FPN_GRIDS, FPN_T_LO, FPN_T_HI
ANCHORS = [
    [(0.006935, 0.014941), (0.015750, 0.028005), (0.033728, 0.035028)],
    [(0.023961, 0.070528), (0.055662, 0.068706), (0.093187, 0.094324)],
    [(0.060280, 0.168604), (0.107559, 0.204684), (0.181239, 0.149031)]]
PX, NTOT = 448, 185220
GEO = np.array([[g, 3] for g in FPN_GRIDS], dtype=np.int32).ravel()
ANC = np.array([a for lvl in ANCHORS for a in lvl], dtype=np.float32).ravel()
inc = subprocess.run(['lean','--print-prefix'],capture_output=True,text=True,
                     check=True).stdout.strip()+'/include'
so = os.path.join(tempfile.gettempdir(),'fpn_affine_gate.so')
subprocess.run(['gcc','-O2','-fPIC','-shared','-I',inc,REPO+'/ffi/f32_helpers.c',
                '-o',so,'-lm'],check=True)
lib = ctypes.CDLL(so, mode=os.RTLD_LAZY)
lib.fpn_affine_one.restype=None
lib.fpn_affine_one.argtypes=[ctypes.POINTER(ctypes.c_float)]*3+[ctypes.c_void_p]+\
    [ctypes.c_size_t]*3+[ctypes.POINTER(ctypes.c_int32),ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_float)]+[ctypes.c_double]*7
BX=(ctypes.c_char*(12348*40))()
REC=3*PX*PX+NTOT*4
def load(i):
    with open(REPO+'/data/visdrone_fpn/val.bin','rb') as f:
        f.seek(4+i*REC); img=np.frombuffer(f.read(3*PX*PX),dtype=np.uint8)
        tgt=np.frombuffer(f.read(NTOT*4),dtype=np.float32)
    return (img.reshape(3,PX,PX).astype(np.float32)/255.).ravel().copy(), tgt.copy()
def run(img,tgt,s,tx,ty):
    img=np.ascontiguousarray(img,np.float32).copy(); tgt=np.ascontiguousarray(tgt,np.float32).copy()
    sc=np.zeros_like(img)
    lib.fpn_affine_one(img.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        tgt.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),ctypes.cast(BX,ctypes.c_void_p),
        3,PX,PX,GEO.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),3,
        ANC.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),s,tx,ty,FPN_T_LO,FPN_T_HI,1.0,0.1)
    return tgt
def occupancy(t):
    out,off=[],0
    for si,g in enumerate(FPN_GRIDS):
        gg=g*g; blk=t[off:off+3*15*gg].reshape(3,15,g,g)
        out.append(int((blk[:,4]>0.5).sum())); off+=3*15*gg
    return out

recs=[load(i) for i in range(16)]
base=[occupancy(t) for _,t in recs]
tot=sum(sum(x) for x in base)

def sweep(label, sgain, tgain):
    rng=np.random.default_rng(1); kept=0; small=0; n=4
    for img,tgt in recs:
        for _ in range(n):
            s=1.0+float(rng.uniform(-1,1))*sgain
            t=run(img,tgt,s,float(rng.uniform(-1,1))*tgain,float(rng.uniform(-1,1))*tgain)
            off=0
            for si,g in enumerate(FPN_GRIDS):
                gg=g*g; blk=t[off:off+3*15*gg].reshape(3,15,g,g)
                m=blk[:,4]>0.5
                kept+=int(m.sum())
                wh=np.minimum(blk[:,2][m],blk[:,3][m])*PX
                small+=int((wh<2.0).sum())
                off+=3*15*gg
    print(f'  {label:28s} survive {100*kept/n/tot:5.1f}%   of those, '
          f'{100*small/max(kept,1):5.1f}% have a side < 2 px')

print(f'baseline positives over 16 val records: {tot}')
print('\nthe two knobs, separated:')
sweep('nothing (control)', 0.0, 0.0)
for tg in (0.05, 0.10, 0.20):
    sweep(f'translate only  +-{tg:.2f}', 0.0, tg)
for sg in (0.10, 0.25, 0.50):
    sweep(f'scale only      +-{sg:.2f}', sg, 0.0)
sweep('scale .25 + translate .10', 0.25, 0.10)

