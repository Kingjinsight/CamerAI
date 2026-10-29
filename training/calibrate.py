"""Measures the model's preference for larger/smaller framings in the app's setting and stores it
in gaic.json as `size_prior`, which the app partly subtracts so zoom hints follow content, not size.

Mirrors the app: portrait 3:4 frames, viewfinder = frame / 1.5, candidates from
Geometry.candidates() in android/.../Geometry.kt (keep the two in sync).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from model import MEAN, STD

W, H = 240, 320
CONTEXT = 1.5


def candidates(view, scales=7, max_positions=6):
    out = [(0.5, 0.5, view)]
    lo = max(0.42, view * 0.6)
    for i in range(scales):
        s = lo * (1 / lo) ** (i / (scales - 1))
        n = min(max(math.ceil((1 - s) / 0.05) + 1, 1), max_positions)
        for yi in range(n):
            for xi in range(n):
                x = (1 - s) / 2 if n == 1 else xi * (1 - s) / (n - 1)
                y = (1 - s) / 2 if n == 1 else yi * (1 - s) / (n - 1)
                out.append((x + s / 2, y + s / 2, s))
    return np.array(out, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="android/app/src/main/assets/gaic.onnx")
    ap.add_argument("--data", default="data/GAIC")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    view = 1 / CONTEXT
    cand = candidates(view)
    boxes = np.stack([(cand[:, 0] - cand[:, 2] / 2) * W, (cand[:, 1] - cand[:, 2] / 2) * H,
                      (cand[:, 0] + cand[:, 2] / 2) * W, (cand[:, 1] + cand[:, 2] / 2) * H], 1)
    rng = np.random.default_rng(0)
    rel = []
    for path in sorted(Path(args.data, "images", "test").glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        cw = min(w, h * 3 / 4)
        ch = cw * 4 / 3
        x0 = (w - cw) / 2 + rng.uniform(-0.3, 0.3) * (w - cw)
        frame = img.crop((int(x0), int((h - ch) / 2), int(x0 + cw), int((h + ch) / 2))).resize((W, H), Image.BILINEAR)
        x = ((np.asarray(frame, np.float32) / 255 - np.array(MEAN, np.float32)) / np.array(STD, np.float32))
        scores = sess.run(None, {"image": x.transpose(2, 0, 1)[None], "boxes": boxes.astype(np.float32)})[0]
        rel.append(scores - scores[0])
    rel = np.stack(rel)

    # Mean score per relative size over all positions, shifted so a pure pan (size 1) costs nothing.
    sizes = np.round(cand[1:, 2] / view, 3)
    r = np.unique(sizes)
    b = np.array([rel[:, 1:][:, sizes == u].mean() for u in r])
    prior = zip(r.tolist(), (b - np.interp(1.0, r, b)).tolist())
    meta_path = Path(args.model).with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["size_prior"] = [[round(u, 3), round(v, 3)] for u, v in prior]
    meta_path.write_text(json.dumps(meta, indent=2))
    print("size_prior (relative size, score bias):", meta["size_prior"])


if __name__ == "__main__":
    main()
