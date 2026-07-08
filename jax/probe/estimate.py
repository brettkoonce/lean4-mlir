"""Estimate the JAX program runs' wall-clock + cost from measured probe epochs —
the `lake run benchmark` pattern for the JAX path.

Feed it steady-state seconds/epoch from the imagenette probes (epoch 2+, NOT epoch 1
which is XLA compile) and it scales the rental program's reference workloads
(planning/mi300x_rental_program.md) by the measured hardware factors:

  python jax/probe/estimate.py --r50-sec-epoch 12.4 --vit-sec-epoch 5.1 --rate 1.39

Scaling model (assumptions printed with the table):
  * conv family anchor: R50 probe img/s @224 bs192; A3 rows scaled by (224/160)^2 for
    the train-res split; bs2048 utilization >= bs192 so estimates are conservative.
  * attn family anchor: ViT-Ti probe img/s; ViT-S/B scaled by fwd-FLOPs ratio
    (1.26 / 4.61 / 17.58 GF @224/16) — assumes GEMM-bound, which is the ViT regime.
  * COMPUTE-BOUND estimates: assumes the memmap/pre-decoded loader (or imagenette
    bins). The stock tfds JPEG pipeline may cap throughput first — see
    pipeline_bench.py; if its img/s is lower than a row's implied rate, the pipeline
    number wins and the row takes longer.
"""
import argparse

IMAGENETTE_EPOCH_IMGS = 49 * 192          # 9,408 (drop_remainder batches of 192)
IMAGENET_EPOCH_IMGS = 1_281_167
VIT_GF = {"Ti": 1.26, "S": 4.61, "B": 17.58}   # fwd GFLOPs @224, patch 16


def rows(r50_ips, vit_ips):
    """(name, family, total images, img/s at THIS run's config)"""
    out = []
    if r50_ips:
        out += [
            ("R50 A3  true-2048   100ep @160", "conv", 100 * IMAGENET_EPOCH_IMGS,
             r50_ips * (224 / 160) ** 2),
            ("R50 A2  true-2048   300ep @224", "conv", 300 * IMAGENET_EPOCH_IMGS,
             r50_ips),
            ("R50 A2  accum x2 (A100 80GB)   ", "conv", 300 * IMAGENET_EPOCH_IMGS,
             r50_ips),
        ]
    if vit_ips:
        out += [
            (f"ViT-S/16 DeiT       300ep @224", "attn", 300 * IMAGENET_EPOCH_IMGS,
             vit_ips * VIT_GF["Ti"] / VIT_GF["S"]),
            (f"ViT-B/16 DeiT       300ep @224", "attn", 300 * IMAGENET_EPOCH_IMGS,
             vit_ips * VIT_GF["Ti"] / VIT_GF["B"]),
        ]
    return out


def fmt_h(sec):
    return f"{sec / 3600:5.1f} h" if sec >= 3600 else f"{sec / 60:5.1f} m"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--r50-sec-epoch", type=float, default=0,
                    help="steady sec/epoch, probe_resnet50_imagenette (epoch 2+)")
    ap.add_argument("--vit-sec-epoch", type=float, default=0,
                    help="steady sec/epoch, probe_vit_tiny_imagenette (epoch 2+)")
    ap.add_argument("--rate", type=float, default=1.39, help="$/GPU/h (A100 community)")
    ap.add_argument("--pipeline-img-s", type=float, default=0,
                    help="measured pipeline_bench img/s cap; 0 = assume compute-bound")
    a = ap.parse_args()
    if not (a.r50_sec_epoch or a.vit_sec_epoch):
        ap.error("need --r50-sec-epoch and/or --vit-sec-epoch")

    r50_ips = IMAGENETTE_EPOCH_IMGS / a.r50_sec_epoch if a.r50_sec_epoch else 0
    vit_ips = IMAGENETTE_EPOCH_IMGS / a.vit_sec_epoch if a.vit_sec_epoch else 0

    print("━━━ jax/probe/estimate.py ━━━ program wall-clock from measured probe factors")
    if r50_ips:
        print(f"  conv anchor: R50-imagenette {r50_ips:7,.0f} img/s "
              f"({a.r50_sec_epoch:.1f}s/epoch @224 bs192)")
    if vit_ips:
        print(f"  attn anchor: ViT-Ti        {vit_ips:7,.0f} img/s "
              f"({a.vit_sec_epoch:.1f}s/epoch @224 bs192)")
    if a.pipeline_img_s:
        print(f"  pipeline cap applied: {a.pipeline_img_s:,.0f} img/s")
    print(f"  {'run':34}  {'wall':>7}  {'@ ${:.2f}/h'.format(a.rate):>9}")
    for name, fam, imgs, ips in rows(r50_ips, vit_ips):
        if a.pipeline_img_s:
            ips = min(ips, a.pipeline_img_s)
        sec = imgs / ips
        print(f"  {name:34}  {fmt_h(sec):>7}  ${sec / 3600 * a.rate:7.2f}")
    print("  assumptions: compute-bound (memmap loader), bs2048 util >= bs192 probe")
    print("  (conservative), ViT scaled by fwd-FLOPs ratio, A3 by (224/160)^2.")


if __name__ == "__main__":
    main()
