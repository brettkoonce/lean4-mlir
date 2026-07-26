"""Reads the SAME `data/visdrone_fpn/{train,val}.bin` the Lean trainer eats.

Consuming the identical encoded bytes removes the data pipeline as a variable:
the twin and the Lean arm see byte-identical images and byte-identical targets,
so any difference in outcome is the model or the training, never the data.
(`check_bin_alignment.py` already verified this encoding round-trips to 2.1e-09.)

Record layout, from `process_split_fpn` and `lean_f32_load_voc_fpn`
(ffi/f32_helpers.c:529):

    <I count>                     4-byte little-endian record count
    then per record:
      uint8  [3, 448, 448]        CHW, raw bytes         602,112
      float32[NTOT]               flat target [P3|P4|P5]  740,880
                                                        = 1,342,992 bytes

Normalization applied by the C loader, reproduced exactly:
    x = (byte/255 - mean) / std,  mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225)
"""
import struct

import numpy as np
import torch
from torch.utils.data import Dataset

IMG = 448
NTOT = 185220
PIX = 3 * IMG * IMG
REC = PIX + NTOT * 4

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class FpnBinDataset(Dataset):
    def __init__(self, path, limit=None, normalize=True):
        self.path = str(path)
        self.normalize = normalize
        with open(self.path, "rb") as f:
            self.count = struct.unpack("<I", f.read(4))[0]
        expect = 4 + self.count * REC
        import os
        actual = os.path.getsize(self.path)
        assert actual == expect, (
            f"{self.path}: header says {self.count} records = {expect} bytes, "
            f"file is {actual}. Record size or NTOT is wrong.")
        if limit is not None:
            self.count = min(self.count, limit)
        self._mm = None

    def _mm_get(self):
        # open lazily so DataLoader workers each get their own handle
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=4)
        return self._mm

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        mm = self._mm_get()
        rec = mm[i * REC:(i + 1) * REC]
        img = np.asarray(rec[:PIX], dtype=np.float32).reshape(3, IMG, IMG) / 255.0
        if self.normalize:
            img = (img - MEAN) / STD
        tgt = np.frombuffer(rec[PIX:].tobytes(), dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(tgt.copy())
