import time, numpy as np
from PIL import Image
rng = np.random.default_rng(0)
imgs = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(200)]
kw = dict(resample=Image.BICUBIC, fillcolor=(128, 128, 128))
def t(fn):
    for x in imgs[:5]: fn(x)
    s = time.perf_counter()
    for x in imgs: fn(x)
    return (time.perf_counter() - s) / len(imgs) * 1e3
print(f"PIL rotate 23 BICUBIC   {t(lambda x: x.rotate(23.0, **kw)):.3f} ms/img")
print(f"PIL shearX 0.2 BICUBIC  {t(lambda x: x.transform(x.size, Image.AFFINE, (1, 0.2, 0, 0, 1, 0), **kw)):.3f} ms/img")
print(f"np->PIL->np round trip  {t(lambda x: np.asarray(Image.fromarray(np.asarray(x)).rotate(23.0, **kw))):.3f} ms/img")
