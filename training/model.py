import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDE = 16
ALIGN = 9
SAMPLING = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _sample_grid(x1, y1, x2, y2, fh, fw):
    """Bilinear sampling grid (align_corners=False) for boxes given in feature-map units."""
    n = x1.shape[0]
    s = ALIGN * SAMPLING
    t = (torch.arange(s, device=x1.device, dtype=x1.dtype) + 0.5) / s
    xs = x1[:, None] + (x2 - x1)[:, None] * t[None]
    ys = y1[:, None] + (y2 - y1)[:, None] * t[None]
    gx = (xs * 2 / fw - 1)[:, None, :].expand(n, s, s)
    gy = (ys * 2 / fh - 1)[:, :, None].expand(n, s, s)
    return torch.stack([gx, gy], dim=-1)


class GAIC(nn.Module):
    """Grid-Anchor-based Image Cropping (Zeng et al., TPAMI 2020) with a MobileNetV2 backbone.

    forward(image [1,3,H,W] normalized, boxes [N,4] x1,y1,x2,y2 in input pixels) -> scores [N]
    """

    def __init__(self, reddim=8, pretrained=True):
        super().__init__()
        weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        mb = torchvision.models.mobilenet_v2(weights=weights).features
        self.f8, self.f16, self.f32 = mb[:7], mb[7:14], mb[14:18]
        self.reduce = nn.Conv2d(32 + 96 + 320, reddim, 1)
        self.head = nn.Sequential(
            nn.Conv2d(reddim * 2, 768, ALIGN), nn.ReLU(inplace=True),
            nn.Conv2d(768, 128, 1), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 1, 1),
        )

    def features(self, image):
        a = self.f8(image)
        b = self.f16(a)
        c = self.f32(b)
        size = b.shape[-2:]
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
        c = F.interpolate(c, size=size, mode="bilinear", align_corners=False)
        return self.reduce(torch.cat([a, b, c], dim=1))

    def score(self, feat, boxes):
        _, ch, fh, fw = feat.shape
        n = boxes.shape[0]
        b = boxes / STRIDE
        x1, y1, x2, y2 = b.unbind(dim=1)
        fmap = feat.expand(n, ch, fh, fw)

        roi = F.grid_sample(fmap, _sample_grid(x1, y1, x2, y2, fh, fw), align_corners=False,
                            padding_mode="border")

        # RoD: zero feature cells whose centers fall inside the crop, then align the whole map.
        cx = torch.arange(fw, device=feat.device, dtype=feat.dtype) + 0.5
        cy = torch.arange(fh, device=feat.device, dtype=feat.dtype) + 0.5
        in_x = (cx[None] >= x1[:, None]) & (cx[None] <= x2[:, None])
        in_y = (cy[None] >= y1[:, None]) & (cy[None] <= y2[:, None])
        keep = 1.0 - (in_y[:, :, None] & in_x[:, None, :]).to(feat.dtype)
        zero, full_w, full_h = torch.zeros_like(x1), torch.full_like(x1, fw), torch.full_like(x1, fh)
        rod = F.grid_sample(fmap * keep[:, None], _sample_grid(zero, zero, full_w, full_h, fh, fw),
                            align_corners=False, padding_mode="border")

        aligned = F.avg_pool2d(torch.cat([roi, rod], dim=1), SAMPLING)
        return self.head(aligned).flatten()

    def forward(self, image, boxes):
        return self.score(self.features(image), boxes)
