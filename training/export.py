import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model import GAIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="android/app/src/main/assets/gaic.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = GAIC(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image = torch.randn(1, 3, 320, 240)
    boxes = torch.tensor([[0.0, 0.0, 240.0, 320.0], [20.0, 30.0, 200.0, 270.0], [60.0, 40.0, 220.0, 253.0]])
    # Image sides must be multiples of 16 (feature stride), between 64 and 1024 px.
    n, h, w = torch.export.Dim("n", min=1, max=2048), torch.export.Dim("h", min=4, max=64), \
        torch.export.Dim("w", min=4, max=64)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, (image, boxes), str(out), dynamo=True, external_data=False,
                      input_names=["image", "boxes"], output_names=["scores"],
                      dynamic_shapes={"image": {2: 16 * h, 3: 16 * w}, "boxes": {0: n}})

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    for hh, ww, nn in [(320, 240, 3), (256, 352, 150)]:
        img = torch.randn(1, 3, hh, ww)
        xy = rng.uniform(0, 0.4, (nn, 2)) * [ww, hh]
        wh = rng.uniform(0.5, 0.6, (nn, 2)) * [ww, hh]
        bx = torch.tensor(np.concatenate([xy, xy + wh], 1), dtype=torch.float32)
        ref = model(img, bx).detach().numpy()
        got = sess.run(None, {"image": img.numpy(), "boxes": bx.numpy()})[0]
        print(f"{hh}x{ww} n={nn} max|torch-onnx|={np.abs(ref - got).max():.2e}")
    meta = {"mos_mean": ckpt["mos_mean"], "mos_std": ckpt["mos_std"], "epoch": ckpt["epoch"]}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", meta)


if __name__ == "__main__":
    main()
