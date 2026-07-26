"""crown_ibp.md gotcha 5: unstable-neuron fraction + a float CROWN-IBP payoff probe.

Weights come from the COMMITTED LipschitzCertScorecardFullNets.lean (the exact
/256 integers the certificates are proved against), not from retraining.
Images are the first 100 MNIST test images at exact k/255, as the scorecards use.

Three counts per (net, eps):
  ibp_box    -- what the committed tier does: propagate the output BOX and
                require denseHi j < denseLo y.  Reproduces 92/88/69/24 (SF).
  ibp_margin -- same interval box on the hidden layer, but sign-split the
                MARGIN row v = W2[y] - W2[j] instead of the two logits.
                Free tightening, no CROWN.
  crown      -- CROWN-IBP: relax each unstable ReLU linearly, back-substitute
                v through W1 to one row A, concretize once: <A,x0> - eps*||A||_1.
"""
import re, struct
from pathlib import Path
import numpy as np


REPO = Path(__file__).resolve().parent.parent
NETS = REPO / "LeanMlir/Proofs/Certificates/LipschitzCertScorecardFullNets.lean"
DATA = REPO / "data"
H, K, DIM, DEN, PIX, N_IMG = 16, 10, 784, 256, 255, 100
EPS_GRID = [1, 2, 4, 8]


def parse_int_list(text):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok.startswith("Int.negSucc"):
            out.append(-(int(tok.split()[1]) + 1))
        else:
            out.append(int(tok))
    return out


def load_nets():
    src = NETS.read_text()
    nets = {}
    for tag in ("SF", "TF"):
        W1 = np.array([parse_int_list(
            src.split(f"def w1z{tag}{k} : List ℤ := [")[1].split("]")[0])
            for k in range(H)], dtype=np.int64)
        # W2 is emitted as a rational matrix, not a List ℤ -- pull the /256 numerators.
        blk = src.split(f"noncomputable def W2{tag} : Fin {K} → Fin {H} → ℝ")[1]
        rows = []
        for ln in blk.splitlines():
            nums = re.findall(rf"\((-?\d+) : ℝ\)/{DEN}", ln)
            if len(nums) == H:
                rows.append([int(v) for v in nums])
            if len(rows) == K:
                break
        W2 = np.array(rows, dtype=np.int64)
        assert W1.shape == (H, DIM) and W2.shape == (K, H), (W1.shape, W2.shape)
        nets[tag] = (W1, W2)
    return nets


def load_mnist():
    with open(DATA / "t10k-images-idx3-ubyte", "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        X = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)
    with open(DATA / "t10k-labels-idx1-ubyte", "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)
    return X[:N_IMG].astype(np.int64), y[:N_IMG].astype(int)


def main():
    nets = load_nets()
    Xraw, y = load_mnist()

    for tag in ("SF", "TF"):
        W1q, W2q = nets[tag]
        A1 = np.abs(W1q).sum(1)                       # ||W1_k||_1, /DEN
        pre = Xraw @ W1q.T                            # <W1_k, x0>, /(DEN*PIX)
        W1f, W2f = W1q / DEN, W2q / DEN
        Xf = Xraw / PIX

        print(f"\n{'='*72}\n{tag}  (weights from committed Nets.lean)\n{'='*72}")
        print(f"{'eps':>7} {'unstable/16':>12} {'dead':>6} {'active':>7} "
              f"{'ibp_box':>8} {'ibp_margin':>11} {'crown':>7}")

        for num in EPS_GRID:
            lo1 = pre - num * A1                      # /(DEN*PIX), exact ints
            hi1 = pre + num * A1
            unstable = (lo1 < 0) & (hi1 > 0)
            dead = hi1 <= 0
            active = lo1 >= 0

            # --- exact integer IBP box separation (the committed tier) -----
            rl = np.maximum(lo1, 0)
            rh = np.maximum(hi1, 0)
            pos = W2q > 0
            outLo = (np.where(pos[None], rl[:, None, :], rh[:, None, :]) * W2q[None]).sum(2)
            outHi = (np.where(pos[None], rh[:, None, :], rl[:, None, :]) * W2q[None]).sum(2)
            box_ok = np.array([
                all(outHi[i, j] < outLo[i, y[i]] for j in range(K) if j != y[i])
                for i in range(N_IMG)])

            # --- margin-direct interval bound (still IBP, tighter split) ---
            # m_j = v.h with v = W2[y]-W2[j]; lower bound = sign-split v on [rl, rh]
            V = W2q[y][:, None, :] - W2q[None, :, :]            # (N, K, H)
            mlo = (np.where(V > 0, rl[:, None, :], rh[:, None, :]) * V).sum(2)
            mask = np.ones((N_IMG, K), bool)
            mask[np.arange(N_IMG), y] = False
            marg_ok = np.where(mask, mlo > 0, True).all(1)

            # --- CROWN-IBP (float) ------------------------------------------
            l = lo1 / (DEN * PIX)
            u = hi1 / (DEN * PIX)
            width = np.where(unstable, u - l, 1.0)
            s = np.where(unstable, u / width, 0.0)             # upper slope
            alpha = np.where(u > -l, 1.0, 0.0)                 # standard heuristic
            Vf = W2f[y][:, None, :] - W2f[None, :, :]          # (N, K, H)
            up = Vf > 0
            # per-neuron coefficient on z and constant, for a LOWER bound on v.h
            coef = np.where(unstable[:, None, :],
                            np.where(up, Vf * alpha[:, None, :], Vf * s[:, None, :]),
                            np.where(active[:, None, :], Vf, 0.0))
            const = np.where(unstable[:, None, :] & ~up,
                             -Vf * s[:, None, :] * l[:, None, :], 0.0).sum(2)
            A = coef @ W1f                                      # (N, K, DIM)
            eps = num / PIX
            lb = (A * Xf[:, None, :]).sum(2) - eps * np.abs(A).sum(2) + const
            crown_ok = np.where(mask, lb > 0, True).all(1)

            print(f"{num:>5}/255 {unstable.sum(1).mean():>12.2f} "
                  f"{dead.sum(1).mean():>6.2f} {active.sum(1).mean():>7.2f} "
                  f"{box_ok.sum():>8} {marg_ok.sum():>11} {crown_ok.sum():>7}")

            # gotcha 1: round the upper-envelope slope UP to a /2^k grid (sound:
            # relu is convex, so any s' >= u/(u-l) still dominates the chord on
            # [l,u]).  How many bits before the count degrades?
            grid = []
            for kbits in (2, 4, 6, 8, 10, 12):
                sr = np.where(unstable, np.ceil(s * 2**kbits) / 2**kbits, 0.0)
                cf = np.where(unstable[:, None, :],
                              np.where(up, Vf * alpha[:, None, :], Vf * sr[:, None, :]),
                              np.where(active[:, None, :], Vf, 0.0))
                cs = np.where(unstable[:, None, :] & ~up,
                              -Vf * sr[:, None, :] * l[:, None, :], 0.0).sum(2)
                Ar = cf @ W1f
                lbr = ((Ar * Xf[:, None, :]).sum(2) - (num / PIX) * np.abs(Ar).sum(2) + cs)
                grid.append((kbits, int(np.where(mask, lbr > 0, True).all(1).sum())))
            print("        /2^k rounded slope: " +
                  "  ".join(f"k={kb}:{c}" for kb, c in grid))

            # how close does any CERTIFYING image come to lb = 0?  (float risk)
            worst = np.where(mask, lb, np.inf).min(1)
            near = worst[crown_ok]
            scale = np.abs(np.where(mask, lb, 0.0)).max()
            print(f"        crown float slack: min certifying margin "
                  f"{near.min():.3e} (dyn. range {scale:.3e}, "
                  f"ratio {near.min()/scale:.2e}); "
                  f"pairs with |lb| < 1e-9: "
                  f"{int((np.abs(np.where(mask, lb, np.inf)) < 1e-9).sum())}")

            if num == 8:
                gained = int((crown_ok & ~box_ok).sum())
                lost = int((box_ok & ~crown_ok).sum())
                hist = np.bincount(unstable.sum(1), minlength=H + 1)
                print(f"        unstable-count histogram (0..16 neurons): "
                      f"{list(hist)}")
                print(f"        crown gains {gained} image(s) over ibp_box, "
                      f"loses {lost}")


main()
