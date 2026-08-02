#!/usr/bin/env python3
"""perturb_clip.py — candidate renders for `clip-tie` (planning/grad_clip.md §7).

    python3 scripts/perturb_clip.py <in.mlir> <out.mlir> <mode>

Every mode edits the COMMITTED clip render in place, touching only operands and constants:
**no SSA name is renamed, no line is added or removed, no type changes, and the entry name is
untouched.** So arity, op counts, the writer audit and the prefix audit are all unchanged, and the
only thing that can move is the number the graph computes — which is what makes a red run mean
something. Same construction as `perturb_wd_mask.py` and `misplace_drop_sites.py`.

The clip block this rewrites, as emitted by `SHlo.gradSumSqAccF` / `SHlo.clipScaleF`:

    %a = stablehlo.constant dense<0.0> : tensor<f32>          # per parameter, N times
    %b = stablehlo.multiply <grad>, <grad> : <T>
    %c = stablehlo.reduce(%b init: %a) applies stablehlo.add across dimensions = [...] : (<T>, tensor<f32>) -> tensor<f32>
    %d = stablehlo.add <acc>, %c : tensor<f32>                # the fold, left-nested from %zero
    ...
    %e = stablehlo.sqrt <total> : tensor<f32>                 # per parameter, N times
    %f = stablehlo.constant dense<1e-06> : tensor<f32>
    %g = stablehlo.add %e, %f : tensor<f32>
    %h = stablehlo.constant dense<1.0> : tensor<f32>          # the CLIP THRESHOLD
    %i = stablehlo.divide %h, %g : tensor<f32>
    %j = stablehlo.constant dense<1.0> : tensor<f32>          # the min's 1.0
    %k = stablehlo.minimum %j, %i : tensor<f32>
    %l = stablehlo.broadcast_in_dim %k, dims = [] : (tensor<f32>) -> <T>
    %m = stablehlo.multiply %l, <grad> : <T>

MODES
  hi        the threshold 1.0 -> 1e9 at every site. **Not a control — this is GATE 3's vehicle.**
            Above the norm, `min(1, c/(gn+eps))` is exactly 1.0 and `x * 1.0` is exact in binary32,
            so the result must be BYTE-IDENTICAL to the unclipped render. It is generated rather
            than committed because an artifact baking a threshold no config sets is a
            silent-hyperparameter artifact (handoff §2a-quater).

  perparam  ⚠⚠ THE CONTROL THAT MATTERS. Point each site's `sqrt` at THAT PARAMETER's own partial
            fold instead of the shared total, turning the global clip into a per-parameter one.
            Everything else in the file is untouched. A per-parameter clip scales, never amplifies,
            and is the identity below the threshold — it satisfies every property of the feature
            except the one that defines it, so a gate that checks "did it get smaller" passes this
            and only the ratio-CONSTANCY check fires.

  nosqrt    drop the root: clip on ||g||^2 rather than ||g||. Trains and descends.
  epsout    `c/gn + 1e-6` instead of `c/(gn + 1e-6)` — the rms-tie eps-placement lesson, one op
            over. Expected to be SMALL at these norms; report it, do not oversell it.

Every mode refuses if it changed nothing, because a control that silently no-ops reports as a
passing gate (the `vit-dp-check` hazard).
"""
import re
import sys

CLIP_DEFAULT = "1.0"
HI = "1000000000.0"


def load(path):
    with open(path) as f:
        return f.readlines()


def clip_sites(lines):
    """Every `minimum` line of a clip site, with the block's other SSA names resolved by walking
    back from it. Returns dicts with the line indices this script needs."""
    out = []
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(%\S+) = stablehlo\.minimum (%\S+), (%\S+) : tensor<f32>", ln)
        if not m:
            continue
        # the eight lines above a `minimum` are the fixed clipScaleF prologue
        blk = lines[i - 6:i]
        if len(blk) != 6:
            continue
        sq = re.match(r"\s*(%\S+) = stablehlo\.sqrt (%\S+) : tensor<f32>", blk[0])
        ep = re.match(r"\s*(%\S+) = stablehlo\.constant dense<([^>]+)> : tensor<f32>", blk[1])
        ad = re.match(r"\s*(%\S+) = stablehlo\.add (%\S+), (%\S+) : tensor<f32>", blk[2])
        cc = re.match(r"\s*(%\S+) = stablehlo\.constant dense<([^>]+)> : tensor<f32>", blk[3])
        dv = re.match(r"\s*(%\S+) = stablehlo\.divide (%\S+), (%\S+) : tensor<f32>", blk[4])
        if not (sq and ep and ad and cc and dv):
            continue
        out.append({
            "i_sqrt": i - 6, "sqrt_out": sq.group(1), "total": sq.group(2),
            "i_eps": i - 5, "i_add": i - 4, "i_clip": i - 3, "clip_val": cc.group(2),
            "i_div": i - 2, "div_out": dv.group(1), "min_out": m.group(1),
        })
    return out


def fold_partials(lines):
    """`acc_out -> partial` for every fold step: the running total AFTER adding this parameter's
    own `reduce`, and that parameter's own `reduce` alone."""
    out = {}
    for i, ln in enumerate(lines):
        m = re.match(r"\s*(%\S+) = stablehlo\.add (%\S+), (%\S+) : tensor<f32>", ln)
        if not m:
            continue
        red = re.match(r"\s*(%\S+) = stablehlo\.reduce\(", lines[i - 1] if i else "")
        if red and red.group(1) == m.group(3):
            out[m.group(1)] = m.group(3)
    return out


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    lines = load(src)
    sites = clip_sites(lines)
    if not sites:
        sys.exit(f"REFUSING: no clip sites found in {src} — is it a `*clip*` render?")
    changed = 0

    if mode == "hi":
        for s in sites:
            if s["clip_val"] != CLIP_DEFAULT:
                sys.exit(f"REFUSING: threshold is {s['clip_val']}, expected {CLIP_DEFAULT} — "
                         f"this mode assumes the reference render")
            lines[s["i_clip"]] = lines[s["i_clip"]].replace(
                f"dense<{CLIP_DEFAULT}>", f"dense<{HI}>", 1)
            changed += 1

    elif mode == "perparam":
        # ⚠⚠ THE CONTROL THAT MATTERS. Point each site's `sqrt` at that parameter's OWN partial
        # fold instead of the shared total. The fold and the clip sites are both emitted in
        # parameter order, so the k-th of each pair up; the pairing is asserted, not assumed.
        partial = fold_partials(lines)
        acc_names = list(partial.keys())
        if len(acc_names) != len(sites):
            sys.exit(f"REFUSING: {len(acc_names)} fold steps vs {len(sites)} clip sites — "
                     f"the pairing this mode assumes does not hold")
        for k, s in enumerate(sites):
            own = partial[acc_names[k]]
            lines[s["i_sqrt"]] = lines[s["i_sqrt"]].replace(s["total"], own, 1)
            changed += 1

    elif mode == "nosqrt":
        # clip on ||g||^2 rather than ||g||. `%zero` is a rank-0 constant both nets' preambles
        # already declare, so this stays a one-for-one line swap.
        for s in sites:
            lines[s["i_sqrt"]] = lines[s["i_sqrt"]].replace(
                f"stablehlo.sqrt {s['total']}", f"stablehlo.add {s['total']}, %zero", 1)
            changed += 1

    elif mode == "epsout":
        # `c/gn + 1e-6` instead of `c/(gn + 1e-6)`, by SWAPPING the two constants and re-reading
        # the three ops between them. Same seven lines, same SSA names, same definition order:
        #     f = <thresh>   g = divide f, e   h = <eps>   i = add g, h   k = minimum j, i
        for s in sites:
            eps_ln, clip_ln = lines[s["i_eps"]], lines[s["i_clip"]]
            eps_nm = re.match(r"\s*(%\S+) = ", eps_ln).group(1)
            clip_nm = re.match(r"\s*(%\S+) = ", clip_ln).group(1)
            eps_val = re.match(r".*dense<([^>]+)>", eps_ln).group(1)
            ind = re.match(r"(\s*)", eps_ln).group(1)
            add_nm = re.match(r"\s*(%\S+) = ", lines[s["i_add"]]).group(1)
            lines[s["i_eps"]] = f"{ind}{eps_nm} = stablehlo.constant dense<{s['clip_val']}> : tensor<f32>\n"
            lines[s["i_add"]] = f"{ind}{add_nm} = stablehlo.divide {eps_nm}, {s['sqrt_out']} : tensor<f32>\n"
            lines[s["i_clip"]] = f"{ind}{clip_nm} = stablehlo.constant dense<{eps_val}> : tensor<f32>\n"
            lines[s["i_div"]] = f"{ind}{s['div_out']} = stablehlo.add {add_nm}, {clip_nm} : tensor<f32>\n"
            changed += 1

    else:
        sys.exit(f"unknown mode '{mode}' — expected hi | perparam | nosqrt | epsout")

    if changed == 0:
        sys.exit("REFUSING: nothing changed — a control that no-ops reports as a passing gate")
    with open(dst, "w") as f:
        f.writelines(lines)
    print(f"{mode}: rewrote {changed} of {len(sites)} clip sites -> {dst}")


if __name__ == "__main__":
    main()
