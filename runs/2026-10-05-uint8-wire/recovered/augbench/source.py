import os, sys, time, resource
os.environ["CUDA_VISIBLE_DEVICES"] = ""; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, tensorflow as tf, tensorflow_datasets as tfds
stage = sys.argv[1]; N = 60 * 512
ds = tfds.load('imagenet2012', split='train', decoders={'image': tfds.decode.SkipDecoding()},
               data_dir='/home/skoonce/tensorflow_datasets')
if stage in ("shuf", "full", "batch"):
    ds = ds.shuffle(8192, seed=42, reshuffle_each_iteration=True).repeat()
if stage in ("full", "batch"):
    ds = ds.flat_map(lambda ex: tf.data.Dataset.from_tensors(ex).repeat(3))
    ds = ds.shuffle(8192, seed=43, reshuffle_each_iteration=True)
if stage == "batch":
    ds = ds.map(lambda ex: (tf.zeros([224, 224, 3], tf.uint8), ex['label']), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(512).prefetch(tf.data.AUTOTUNE); N = 60
it = iter(ds)
for _ in range(9000 if stage != "batch" else 8): next(it)
c0 = resource.getrusage(resource.RUSAGE_SELF); t = time.perf_counter()
for _ in range(N): next(it)
dt = time.perf_counter() - t; c1 = resource.getrusage(resource.RUSAGE_SELF)
cpu = (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime)
n_img = N * (512 if stage == "batch" else 1)
print(f"{stage:6s}: {n_img/dt:8.0f} examples/s   {cpu/dt:5.1f} cores busy")
