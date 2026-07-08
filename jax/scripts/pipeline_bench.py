"""Host input-pipeline throughput bench — is the trainer input-bound, and does the
pipeline scale with CPU?

Every cloud-rental cost estimate (planning/mi300x_rental_program.md) rides on the
assumption that the tfds/tf.data host pipeline sustains ~4-6k img/s. This measures it:
imports `build_imagenet_iter` from a generated trainer (nothing else runs — jax is
pinned to CPU so no GPU is touched), drains batches, and reports img/s plus how much
of the machine's CPU the pipeline actually used (the CPU-bound test: img/s should
scale with --threads until cores or disk run out).

Usage (TFDS_DATA_DIR must point at a prepared imagenet2012):
  python jax/scripts/pipeline_bench.py                       # train pipeline, defaults
  python jax/scripts/pipeline_bench.py --batch 2048          # A2-sized batches
  python jax/scripts/pipeline_bench.py --eval                # decode+center-crop only
                                                             # (delta vs train = aug+RA cost)
  python jax/scripts/pipeline_bench.py --sweep               # thread-scaling curve
                                                             # (re-execs itself; TF threadpools
                                                             #  are fixed at first use)
  python jax/scripts/pipeline_bench.py --gpu-step-ms 250     # verdict vs a measured step time

Interpreting: `cpu-sat` is pipeline CPU-seconds per wall-second / total cores. Near 1.0
and img/s still short of the GPU's appetite -> genuinely CPU-bound (buy the pipeline fix:
pre-decoded arrays + prefetch). Low cpu-sat and flat scaling -> disk/decode-locality bound.
"""
import argparse, importlib.util, os, subprocess, sys, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never grab the GPU for a host bench

GEN_DIR = os.path.join(os.path.dirname(__file__), "..", ".lake", "build")
DEFAULT_GENS = ["generated_resnet50_imagenet_a2true2048.py",
                "generated_resnet50_imagenet_rsbfaithful.py",
                "generated_resnet50_imagenet_short.py"]


def load_generated(name):
    path = name if os.path.sep in name else os.path.join(GEN_DIR, name)
    spec = importlib.util.spec_from_file_location("gen_trainer", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, path


def bench(args):
    import tensorflow as tf                      # threadpools are frozen at first op:
    if args.threads > 0:                         # set BEFORE the generated module import
        tf.config.threading.set_intra_op_parallelism_threads(args.threads)
        tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.set_visible_devices([], "GPU")     # tf must not reserve the GPU either

    gen = args.gen or next(g for g in DEFAULT_GENS
                           if os.path.exists(os.path.join(GEN_DIR, g)))
    m, path = load_generated(gen)
    training = not args.eval
    it = iter(m.build_imagenet_iter("train" if training else "validation",
                                    args.batch, training=training, augment=training))

    for _ in range(args.warmup):                 # autotune ramp + shuffle-buffer fill
        next(it)
    t0w, t0c = time.monotonic(), sum(os.times()[:2])
    for _ in range(args.batches):
        next(it)
    wall = time.monotonic() - t0w
    cpu = sum(os.times()[:2]) - t0c              # process CPU-s (tf threads are in-process)
    ncpu = os.cpu_count() or 1
    imgs = args.batches * args.batch
    ips = imgs / wall
    sat = cpu / wall / ncpu

    mode = "eval(decode+ccrop)" if args.eval else "train(decode+aug+RA3)"
    thr = args.threads if args.threads > 0 else "auto"
    print(f"[pipeline] {os.path.basename(path)} | {mode} | batch {args.batch} | "
          f"threads {thr}/{ncpu}")
    print(f"[pipeline] {imgs} imgs in {wall:.1f}s -> {ips:,.0f} img/s | "
          f"cpu-sat {sat:.2f} ({cpu:.0f} cpu-s)")
    if args.gpu_step_ms > 0:
        need = args.batch / (args.gpu_step_ms / 1000.0)
        verdict = "INPUT-BOUND" if ips < need else "gpu-bound (pipeline keeps up)"
        print(f"[pipeline] gpu wants {need:,.0f} img/s @ {args.gpu_step_ms}ms/step "
              f"-> {verdict} (pipeline/gpu = {ips / need:.2f}x)")
    return ips


def sweep(args):
    """Thread-scaling curve via re-exec (TF threadpools can't be resized in-process)."""
    base = [sys.executable, os.path.abspath(__file__), "--batch", str(args.batch),
            "--batches", str(args.batches), "--warmup", str(args.warmup)]
    if args.eval:
        base.append("--eval")
    if args.gen:
        base += ["--gen", args.gen]
    ncpu = os.cpu_count() or 1
    points = sorted({4, 8, 16, 32, ncpu} | {0}) if ncpu >= 4 else [0]
    print(f"[sweep] threads x {points} on {ncpu} cores (0 = tf default/AUTOTUNE)")
    for t in points:
        if 0 < t > ncpu:
            continue
        subprocess.run(base + ["--threads", str(t)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gen", default="", help="generated trainer .py (name in .lake/build or path)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--batches", type=int, default=50, help="timed batches (after warmup)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="tf intra+inter op threads (0=default)")
    ap.add_argument("--eval", action="store_true", help="bench the eval pipeline (no aug/RA)")
    ap.add_argument("--sweep", action="store_true", help="re-exec across thread counts")
    ap.add_argument("--gpu-step-ms", type=float, default=0.0,
                    help="measured GPU ms/step at this batch -> input-bound verdict")
    a = ap.parse_args()
    sweep(a) if a.sweep else bench(a)
