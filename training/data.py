import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from model import MEAN, STD


def load_split(root, split):
    """GAICD v1 annotations: one crop per line, `y1 x1 y2 x2 MOS` in original pixels.

    Kept as numpy so DataLoader workers get copies instead of shared-memory file descriptors.
    """
    root = Path(root)
    items = []
    for img in sorted((root / "images" / split).glob("*.jpg")):
        rows = [line.split() for line in (root / "annotations" / f"{img.stem}.txt").read_text().splitlines() if line.strip()]
        boxes = np.array([[float(r[1]), float(r[0]), float(r[3]), float(r[2])] for r in rows], np.float32)
        mos = np.array([float(r[4]) for r in rows], np.float32)
        items.append((img, boxes, mos))
    return items


class GAICD(Dataset):
    def __init__(self, items, mos_mean, mos_std, train, short_side=256, crops_per_image=64):
        self.items = items
        self.mos_mean, self.mos_std = mos_mean, mos_std
        self.train = train
        self.short_side = short_side
        self.crops = crops_per_image
        self.jitter = v2.ColorJitter(0.4, 0.4, 0.4, 0.05)
        self.norm = v2.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes, mos = self.items[i]
        boxes, mos = torch.from_numpy(boxes), torch.from_numpy(mos)
        img = Image.open(path).convert("RGB")
        w, h = img.size
        short = random.choice((224, 256, 288)) if self.train else self.short_side
        scale = short / min(w, h)
        # Snap sides to multiples of 32 so MPS/cuDNN see few distinct shapes.
        nw, nh = max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32)
        x = v2.functional.to_image(img.resize((nw, nh), Image.BILINEAR))
        x = v2.functional.to_dtype(x, torch.float32, scale=True)
        boxes = boxes * torch.tensor([nw / w, nh / h, nw / w, nh / h])
        if self.train:
            x = self.jitter(x)
            if random.random() < 0.5:
                x = x.flip(-1)
                boxes = torch.stack([nw - boxes[:, 2], boxes[:, 1], nw - boxes[:, 0], boxes[:, 3]], dim=1)
            if len(boxes) > self.crops:
                keep = torch.randperm(len(boxes))[: self.crops]
                boxes, mos = boxes[keep], mos[keep]
        return self.norm(x)[None], boxes, (mos - self.mos_mean) / self.mos_std
