#!/usr/bin/env python3
"""Generate tests/comparator/{Challenge,Solution}Tier.lean + config-tier.json.

WHY GENERATED. The tier pair states the whole-net, tie, faithfulness, data-parallel,
float and certificate theorems, whose signatures run to hundreds of lines apiece
(`cnx_net_tiedGB` alone is 186). Hand-copying them into two files that must stay
BIT-IDENTICAL is a losing game; comparator would catch the drift, but only after a
35-minute CI run. So both files are printed from one string per theorem, taken from
Lean's own pretty printer at `#check @<decl>` — the statement is the declaration's
type by construction, and the solution proves it with the bare constant.

THE OPTIONS ARE NOT COSMETIC. Default `#check` output does not re-elaborate: it elides
proof arguments, inferable implicits and deep subterms (`⋯`). Measured 2026-09-20 over
these 21 declarations: bare pp fails on 10, `pp.analyze` gets that to 9, adding
`pp.proofs` to 3, and adding `pp.maxSteps`/`pp.deepTerms` to 1. The survivor is the ViT
step tie, whose `vitBlockCotInAtMHV` applications drop dimension implicits that nothing
downstream pins; it needs `pp.explicit`, which costs 422 lines instead of 158. Hence
EXPLICIT below.

USAGE:  python3 scripts/gates/gen_comparator_tier.py            # regenerate + verify
        python3 scripts/gates/gen_comparator_tier.py --check    # fail if the files would change

VERIFY. The default run elaborates the generated solution inside the PARENT package (where
every olean already exists) rather than the nested comparator package, so the check costs
seconds and needs no second Mathlib tree. It does not replace the comparator run
(`tests/comparator/run.sh`), which needs the landrun / lean4export / comparator toolchain and
its own Mathlib tree.
"""
import json, re, subprocess, sys, os, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT  = os.path.join(ROOT, 'tests', 'comparator')

# The audited set from formalization.yaml, minus what config-arch.json already covers.
# Names are FULLY qualified: three of the yaml's own strings are not (they are written
# relative to an implicit `open Proofs`), which is why this list is not read from it.
DECLS = [
    "Proofs.bn_input_grad_correct",
    "Proofs.resnet50ForwardB_full_has_vjp_at_correct",
    "Proofs.vitTiny_has_vjp_correct",
    "Proofs.StableHLO.mnv4FwdGraphB_full_faithful",
    "Proofs.Mnv2FullBSeal.sealX_nonconstant",
    "Proofs.Mnv4FullBSeal.sealX_backward_nontrivial",
    "Proofs.ResNet34PoCB.convStridedWGradB_den",
    "Proofs.smoothedCE_grad",
    "Proofs.ResNet50TieB.r50_net_tiedB",
    "Proofs.ViTTiePoC.vit_net_tied_certified",
    "Proofs.CnxTiePoCGB.cnx_net_tiedGB",
    "Proofs.dpMeanGrad_ne_globalBatchGrad",
    "Proofs.dpSyncGrad_eq_globalBatchGrad",
    "Proofs.den_bnSyncBack_allReduce",
    "Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global",
    "Proofs.StableHLO.resnet34FwdGraphSync_full_shard",
    "Proofs.ResNet34SyncTieB.r34_net_syncTiedB",
    "Proofs.StableHLO.mobilenetv2FwdGraphSync_full_shard",
    "Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB",
    "Proofs.StableHLO.efficientnetFwdGraphSync_full_shard",
    "Proofs.EnetSyncTieG.efficientnet_net_syncTiedG",
    "Proofs.StableHLO.resnet50FwdGraphSync_full_shard",
    "Proofs.ResNet50SyncTieB.r50_net_syncTiedB",
    "Proofs.StableHLO.mnv4FwdGraphSync_full_shard",
    "Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB",
    "Proofs.adamW_at_allReduceMeanF",
    "Proofs.r34InputGradB_eq_r34B_full_vjp",
    "Proofs.efficientnetInputGradB_full_correct",
    "Proofs.convnextImagenetInputGradB_eq_vjp",
    "Proofs.FloatModel.linear_e4m3_argmax_preserved",
    "Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends",
    "Proofs.lipschitz_margin_certified_radius",
    "Proofs.LipschitzCertDemo.scorecard_sdp",
    "Proofs.smoothing_certified_radius_classifier",
    "Proofs.MuonGeometry.shampoo_eq_muon",
]
MODULES = [
    "LeanMlir.Proofs.Architectures.BatchNorm",
    "LeanMlir.Proofs.Certificates.LipschitzCert",
    "LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDP",
    "LeanMlir.Proofs.Certificates.SmoothingGaussian",
    "LeanMlir.Proofs.Float.FloatBridge",
    "LeanMlir.Proofs.Foundation.DataParallel",
    "LeanMlir.Proofs.Foundation.DataParallelNode",
    "LeanMlir.Proofs.Foundation.DataParallelSync",
    "LeanMlir.Proofs.Foundation.DataParallelSyncBf16",
    "LeanMlir.Proofs.Foundation.MuonGeometry",
    "LeanMlir.Proofs.Foundation.SmoothedLossCot",
    "LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB",
    "LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTieB",
    "LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullWholeBackCertifiedTie",
    "LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBSeal",
    "LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBSeal",
    "LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB",
    "LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB",
    "LeanMlir.Proofs.Foundation.GradNodesB",
    "LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP",
    "LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB",
    "LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB",
    "LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB",
    "LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncStepTieB",
    "LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncStepTieG",
    "LeanMlir.Proofs.Nets.ResNet.ResNet50SyncStepTieB",
    "LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncStepTieB",
    "LeanMlir.Proofs.Nets.ViT.ViTDepthK",
    "LeanMlir.Proofs.Nets.ViT.ViTStepTie",
    "LeanMlir.Proofs.Training.TrainedLinearDescent",
]
BASE_OPTS = ["format.width 96", "pp.analyze true", "pp.funBinderTypes true",
             "pp.numericTypes true", "pp.proofs true", "pp.maxSteps 100000000",
             "pp.deepTerms true"]
EXPLICIT = {"Proofs.ViTTiePoC.vit_net_tied_certified"}   # see the module docstring


def lean(src: str) -> str:
    fd, path = tempfile.mkstemp(suffix='.lean', dir=os.path.join(ROOT, 'tests'))
    os.write(fd, src.encode()); os.close(fd)
    try:
        r = subprocess.run(['lake', 'env', 'lean', os.path.relpath(path, ROOT)],
                           cwd=ROOT, capture_output=True, text=True, timeout=3600)
        return r.stdout + r.stderr
    finally:
        os.unlink(path)


def print_types(decls, opts):
    src = '\n'.join(f'import {m}' for m in MODULES) + '\n\nset_option maxHeartbeats 4000000\n'
    src += ''.join(f'set_option {o}\n' for o in opts) + '\n'
    src += ''.join(f'#check @{d}\n' for d in decls)
    log = lean(src)
    if re.search(r': error', log):
        sys.exit('pretty-print probe failed:\n' + log[:4000])
    types, cur, name = {}, None, None
    for ln in log.split('\n'):
        m = re.match(r'^@?([A-Za-z_][A-Za-z0-9_.\']*) : (.*)$', ln)
        if m and m.group(1) in decls:
            if name: types[name] = '\n'.join(cur)
            name, cur = m.group(1), [m.group(2)]
        elif cur is not None and ln.strip():
            cur.append(ln)
    if name: types[name] = '\n'.join(cur)
    missing = [d for d in decls if d not in types]
    if missing: sys.exit(f'no type printed for: {missing}')
    return types


HEADER = """{imports}

universe u_1

open Proofs
open scoped Real

set_option maxHeartbeats 8000000
-- The statements are pretty-printer output, and the printer names binders the declarations
-- never use (`fun (x : Fin n) => (0 : ℝ)`, `[inst : ...]` under `pp.explicit`).
set_option linter.unusedVariables false

/-! # {title}

{blurb}

⚠ **MACHINE-GENERATED — do not hand-edit.** Every statement is the project declaration's
own type as Lean prints it (`scripts/gates/gen_comparator_tier.py`), so this file and its
{peer} carry the same text by construction rather than by review. Regenerate after any
statement change; the generator verifies that what it wrote still elaborates.
-/
"""
BLURB_C = """The **tier** half of the comparator suite. `Challenge.lean` carries the
architecture-free calculus floor and `ChallengeArch.lean` the per-layer and whole-net
Jacobians; this file carries the layer above them — the step ties, the codegen
faithfulness results, the whole-net back-chains, the data-parallel results, the float
bridge, the descent result and the three certificate theorems — plus the three nets the
other two files never mention (ResNet-34, ResNet-50, MobileNetV4).

Its contents are exactly the declarations `formalization.yaml` advertises as the audited
set, minus the four `config-arch.json` already covers. So "every declaration this project
puts forward is independently kernel-rechecked" is a claim a reader can now check by
diffing this theorem list against that yaml."""
BLURB_S = """Solution to `ChallengeTier.lean`. Each proof is the project theorem itself:
the statement IS that theorem's type, so the delegation is a bare constant and nothing can
be weakened between the two files without failing to elaborate."""


def render(types, solution: bool):
    body = []
    for d in DECLS:
        leaf = d.split('.')[-1]
        ty = '\n'.join('    ' + l if l.strip() else l for l in types[d].split('\n'))
        tail = f' :=\n  {d}' if solution else ' := by sorry'
        body.append(f'/-- `{d}` -/\ntheorem chk_{leaf} :\n{ty}{tail}\n')
    return HEADER.format(imports='\n'.join(f'import {m}' for m in MODULES),
                         title='Solution to the tier challenge' if solution
                               else 'Challenge file for `leanprover/comparator` — the tie, faithfulness and certificate tier',
                         blurb=BLURB_S if solution else BLURB_C,
                         peer='`SolutionTier.lean`' if not solution else '`ChallengeTier.lean`') \
        + '\n' + '\n'.join(body)


def check_yaml():
    """Every `main_results` row must name a comparator config that actually holds it.

    The field sat at `comparator_config: ""` on all 25 rows until 2026-09-20 — schema
    present, information zero, while four of the rows WERE being checked. Filling it in is
    only worth anything if something keeps it honest, so: parse the yaml, open the config
    each row names, and require `chk_<leaf>` to be in its theorem_names.
    """
    y = open(os.path.join(ROOT, 'formalization.yaml')).read()
    rows = re.findall(r'declaration: "([^"]+)".*?comparator_config: "([^"]*)"', y, re.S)
    if not rows: sys.exit('formalization.yaml: no main_results rows parsed')
    bad = []
    for decl, cfg in rows:
        if not cfg:
            bad.append(f'{decl}: comparator_config is empty'); continue
        path = os.path.join(ROOT, cfg)
        if not os.path.exists(path):
            bad.append(f'{decl}: no such config {cfg}'); continue
        if f"chk_{decl.split('.')[-1]}" not in json.load(open(path))['theorem_names']:
            bad.append(f'{decl}: not in {cfg}')
    if bad: sys.exit('formalization.yaml <-> comparator config mismatch:\n  ' + '\n  '.join(bad))


def main():
    check = '--check' in sys.argv
    plain = [d for d in DECLS if d not in EXPLICIT]
    types = print_types(plain, BASE_OPTS)
    if EXPLICIT:
        types.update(print_types(sorted(EXPLICIT), BASE_OPTS + ['pp.explicit true']))
    files = {'ChallengeTier.lean': render(types, False),
             'SolutionTier.lean':  render(types, True)}
    cfg = json.dumps({"challenge_module": "ChallengeTier", "solution_module": "SolutionTier",
                      "theorem_names": [f"chk_{d.split('.')[-1]}" for d in DECLS],
                      "permitted_axioms": ["propext", "Quot.sound", "Classical.choice"],
                      "enable_nanoda": False}, indent=4) + '\n'
    files['config-tier.json'] = cfg

    if check:
        bad = [n for n, c in files.items()
               if open(os.path.join(OUT, n)).read() != c]
        if bad: sys.exit('stale (regenerate with scripts/gates/gen_comparator_tier.py): ' + ', '.join(bad))
        check_yaml()
        print(f'✓ comparator tier files match the generator ({len(DECLS)} theorems); '
              f'every formalization.yaml row names a config that contains it')
        return

    # verify the solution elaborates in the parent package before writing anything
    log = lean(files['SolutionTier.lean'])
    errs = [l for l in log.split('\n') if ': error' in l]
    if errs: sys.exit('generated SolutionTier does not elaborate:\n' + '\n'.join(errs[:20]))
    for n, c in files.items():
        open(os.path.join(OUT, n), 'w').write(c)
    print(f'✓ wrote {len(DECLS)} theorems to ChallengeTier/SolutionTier + config-tier.json '
          f'({len(files["SolutionTier.lean"].splitlines())} lines each)')


if __name__ == '__main__':
    main()
