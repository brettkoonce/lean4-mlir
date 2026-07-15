#!/usr/bin/env python
"""What does each seg loss's gradient do as a class collapses? (brats_demo.md)

The generalization of `seg_dice_vanishing_grad_probe.py`, which asked this of
Dice alone and got the answer that reframed the whole demo. This asks it of
every loss we emit, because the answer is what decides which one can save a
thin class -- and it is measurable *before* spending a GPU-week finding out.

The framing, which took a wrong turn to arrive at: a loss does not rescue a
rare class by having a large gradient on it. CE's gradient on a collapsed class
is already the largest of any loss here, and CE collapses anyway. What matters
is the gradient on the rare class *relative to the one it competes with* -- the
enormous, easy background that owns 97% of the pixels. There are exactly two
ways to win that fight, and each loss picks one, both, or neither:

    (A) amplify the rare class          -- weighted CE
    (B) defund the easy majority        -- focal

Dice attempts (A) and fails, because its gradient carries a p_i factor from the
softmax Jacobian and goes to zero exactly where the class has collapsed. This
script measures all of it against the REAL EMITTED MLIR, not a numpy model.

Run:  <jax-venv>/bin/python scripts/seg_grad_scorecard.py
Needs iree.runtime + iree.compiler. CPU only.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import iree.runtime as rt  # noqa: E402

PROBE = ".lake/build/bin/seg-loss-probe"
IREE_COMPILE = ".venv/bin/iree-compile"

# Inverse-frequency weights, from scripts/brats_class_weights.py. Same vector
# demos/MainUnetBratsTrain.lean uses; passed on the CLI rather than duplicated
# into the Lean probe.
BRATS_W = "w=1.0:60.9033:220.0868:195.5835"

# Where RetinaNet prior-bias init (TrainConfig.headPriorBias) puts the net at
# step 0: bias_c = log(pi_c), so z0 - z3 = log(0.9746/0.0050) = 5.27. Verified
# against the emitted checkpoint -- softmax(head bias) == prior to 2e-09. This
# row is the whole argument for pairing prior-bias init with focal, so the sweep
# lands on it exactly rather than straddling it.
PRIOR_BIAS_Z0 = float(np.log(0.9746 / 0.0050))

# (label, probe argv tail). `ce` is the yardstick every ratio is taken against.
LOSSES = [
    ("ce",     ["ce"]),
    ("dice",   ["dice"]),
    ("dicece", ["dicece"]),
    ("wce",    ["wce", BRATS_W]),
    ("focal",  ["focal", "g=2.0"]),
]


def run(argv_tail, B, NC, H, W, z, y):
    with tempfile.TemporaryDirectory() as td:
        mlir = os.path.join(td, "g.mlir")
        subprocess.run([PROBE] + argv_tail + [str(B), str(NC), str(H), str(W), mlir],
                       capture_output=True, check=True)
        vmfb = os.path.join(td, "g.vmfb")
        r = subprocess.run([IREE_COMPILE, mlir, "--iree-hal-target-backends=llvm-cpu",
                            "-o", vmfb], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"iree-compile failed for {argv_tail}:\n{r.stderr[:2000]}")
        ctx = rt.SystemContext(config=rt.Config("local-task"))
        with open(vmfb, "rb") as f:
            ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
        out = ctx.modules.seg_loss_probe["main"](z.astype(np.float32), y.astype(np.int32))
        return np.asarray(out[0]).item(), np.asarray(out[1]).astype(np.float64)


def brats_like_labels(B, H, W, rng):
    """A label field with BraTS's measured class balance: 97.46 / 1.60 / 0.44 /
    0.50 percent. The imbalance IS the experiment, so a uniform random field
    would measure nothing."""
    y = np.zeros((B, H, W), dtype=int)
    n = B * H * W
    idx = rng.permutation(n)
    counts = [int(round(n * f)) for f in (0.0160, 0.0044, 0.0050)]
    at = 0
    flat = y.reshape(-1)
    for cls, c in zip((1, 2, 3), counts):
        flat[idx[at:at + c]] = cls
        at += c
    return flat.reshape(B, H, W)


def main():
    if not os.path.exists(PROBE):
        sys.exit(f"{PROBE} missing -- run: lake build seg-loss-probe")
    B, NC, H, W = 4, 4, 32, 32
    rng = np.random.RandomState(0)
    y = brats_like_labels(B, H, W, rng)
    n = y.size
    counts = [int((y == c).sum()) for c in range(NC)]
    print("seg-loss gradient scorecard -- measured against the emitted MLIR\n")
    print(f"label field {B}x{H}x{W} = {n} px, BraTS-like balance: " +
          "  ".join(f"c{c}={counts[c]} ({100*counts[c]/n:.2f}%)" for c in range(NC)))
    print("""
Sweep: raise the BACKGROUND logit, which is what the collapse actually is.

An earlier version of this script pushed class 3's logit down and left the
background at random logits. That measured (A) fine and measured (B) as a flat
line -- which is wrong, and wrong in a way that would have let us ship a false
claim about focal. With random logits the background sits at p_0 ~ 0.25: it is
not CONFIDENT, so focal's (1-p_t)^gamma factor has nothing to bite on and focal
looks identical to CE.

But the trained net is confident. That is the whole content of "CE lands below
the class-prior floor" (brats_demo.md Workstream A): it gets there by being very
sure about trivially-separable background. So the state to probe is z_0 -> large,
which simultaneously makes p_0 -> 1 at background pixels and squeezes p_3 -> 0 at
tumour pixels. One knob, both halves of the collapse.
""")

    tum = (y == 3)
    bg = (y == 0)

    hdr = (f"  {'z0':>5} {'p3@tum':>9} {'p0@bg':>8} |" +
           "".join(f"{name:>11}" for name, _ in LOSSES))
    print("(A) RESCUE SIGNAL   mean |dz| on ch3 at true-class-3 px")
    print("    (the gradient that has to push the collapsed class back up)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for bias in (0.0, 2.0, 4.0, PRIOR_BIAS_Z0, 6.0, 8.0, 10.0):
        z = rng.randn(B, NC, H, W) * 0.5
        z[:, 0, :, :] += bias
        e = np.exp(z - z.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        p3 = p[:, 3, :, :][tum].mean()
        p0 = p[:, 0, :, :][bg].mean()
        vals = {}
        for name, tail in LOSSES:
            _, dz = run(tail, B, NC, H, W, z, y)
            vals[name] = {
                "rescue": np.abs(dz[:, 3, :, :][tum]).mean(),
                "major": np.abs(dz[:, 0, :, :][bg]).mean(),
                "rescue_tot": np.abs(dz[:, 3, :, :][tum]).sum(),
                "major_tot": np.abs(dz[:, 0, :, :][bg]).sum(),
            }
        rows.append((bias, p3, p0, vals))
        tag = "   <- prior-bias init starts the net HERE" if bias == PRIOR_BIAS_Z0 else ""
        print(f"  {bias:5.1f} {p3:9.2e} {p0:8.4f} |" +
              "".join(f"{vals[nm]['rescue']:11.3e}" for nm, _ in LOSSES) + tag)

    print("\n(B) MAJORITY SIGNAL   mean |dz| on ch0 at true-background px")
    print("    (the easy 97% the rescue signal has to out-shout)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for bias, p3, p0, vals in rows:
        tag = "   <- prior-bias init starts the net HERE" if bias == PRIOR_BIAS_Z0 else ""
        print(f"  {bias:5.1f} {p3:9.2e} {p0:8.4f} |" +
              "".join(f"{vals[nm]['major']:11.3e}" for nm, _ in LOSSES) + tag)

    print("\n(C) THE RATIO THAT DECIDES IT   sum|dz| over class-3 px / sum|dz| over bg px")
    print("    (totals, not means -- 0.5% of pixels vs 97% is the whole fight)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for bias, p3, p0, vals in rows:
        tag = "   <- prior-bias init starts the net HERE" if bias == PRIOR_BIAS_Z0 else ""
        print(f"  {bias:5.1f} {p3:9.2e} {p0:8.4f} |" +
              "".join(f"{vals[nm]['rescue_tot']/max(vals[nm]['major_tot'],1e-30):11.3e}"
                      for nm, _ in LOSSES) + tag)

    print("""
Reading the three tables (measured values as of 2026-07-15):

  (A) is where Dice dies. Its column falls ~60x as p_3 goes 2.4e-01 -> 6.0e-05
      -- the p_i factor in dz_i = p_i(g_i - sum_j g_j p_j) -- so the signal that
      should rescue the class evaporates precisely as the class collapses. CE's
      column is FLAT (1.9e-04 -> 2.4e-04): (p-y)/N is -1/N at p=0, wholly
      indifferent. That is why dicece == ce in the matched run -- Dice brings
      nothing to the argument.

  (A) also shows wce and focal are NOT variations on one idea. wce multiplies
      the rescue signal by ~50x here (its class-3 weight over the Sw
      normalizer). focal leaves it at CE's, to three digits. **focal does not
      amplify the rare class at all.**

  (B) is focal's entire mechanism, and it is invisible in (A). As the background
      becomes confident (p_0: 0.25 -> 0.9998), CE's own majority gradient decays
      ~4300x, but focal's decays 2e10x -- another ~4e6x beyond CE. The
      (1-p_t)^gamma factor defunds the easy majority. focal wins by silencing
      the crowd, not by handing the rare class a megaphone.

  (C) is the number that predicts the outcome, and it separates the three
      mechanisms cleanly:

      * dice     0.028 -> 0.086, and DECLINING past z0=6. It cannot win the
                 fight at any collapse depth. Note it starts ~5x better balanced
                 than CE and ends ~335x worse: Dice's correction weakens as the
                 collapse deepens, which is positive feedback into the very state
                 it was meant to prevent. "Dice can only help if the collapse
                 never happens", quantified.
      * wce      a constant ~196x over CE at every depth -- exactly w_3/w_0, as
                 it must be. State-independent, and therefore working from step 0.
      * focal    5.1e-03 at init, indistinguishable from CE (1.01x), then 1.2e+08
                 once collapsed. Negative feedback: the more the net collapses,
                 the harder focal fights back.

  The sharp prediction, and the reason to run focal rather than assume it:
  **focal is a no-op at initialization.** At a uniform softmax there is no
  confidence to suppress, so it IS CE. Its protection only materializes as the
  net becomes confident -- which is the collapse itself. The collapse is decided
  in the first ~100 steps (brats_demo.md Workstream A), so the open question is
  whether focal's feedback engages fast enough to catch it, or arrives, like
  Dice, to a decision already made. wce has no such timing risk and no such
  self-limiting property; it is a blunt constant. Those are different bets, and
  the matched-budget run is what settles them.

None of this substitutes for that run. It is the cheaper question asked first:
which of these CAN work, before paying to find out which DOES.""")


if __name__ == "__main__":
    sys.exit(main())
