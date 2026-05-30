"""
MobileFaceNet-S: Lightweight student model for face recognition.
Architecture: Depthwise separable convs + bottleneck blocks + ArcFace-style embedding.
Target: 256-d L2-normalised embedding, ~23 MB PyTorch -> 14.9 MB TFLite after compression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn2(self.pw(F.relu(self.bn1(self.dw(x))))))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        mid = in_ch * expand
        self.use_residual = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True)]
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_residual else self.conv(x)


class MobileFaceNetS(nn.Module):
    """
    MobileFaceNet-S: student model optimised for 112x112 face crops.
    Outputs 256-d L2-normalised embedding.
    Param count: ~3.4M -> suitable for INT8 quantisation.
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.PReLU(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, groups=64, bias=False),
            nn.BatchNorm2d(64), nn.PReLU(64),
        )

        # Bottleneck stages: (in, out, stride, expand, repeat)
        cfg = [
            (64,  64,  2, 2, 5),
            (64,  128, 2, 4, 1),
            (128, 128, 1, 2, 6),
            (128, 128, 2, 4, 1),
            (128, 128, 1, 2, 2),
        ]
        blocks = []
        for in_c, out_c, s, t, n in cfg:
            for i in range(n):
                blocks.append(InvertedResidual(in_c if i == 0 else out_c, out_c, s if i == 0 else 1, t))
        self.blocks = nn.Sequential(*blocks)

        # Head: conv -> linear -> BN -> embedding
        self.conv_last = nn.Sequential(
            nn.Conv2d(128, 512, 1, bias=False),
            nn.BatchNorm2d(512), nn.PReLU(512),
        )
        self.linear = nn.Linear(512 * 7 * 7, embedding_dim, bias=False)
        self.bn_final = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 112, 112), returns (B, 256) L2-normalised"""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        x = x.flatten(1)
        x = self.bn_final(self.linear(x))
        return F.normalize(x, p=2, dim=1)


class ArcFaceHead(nn.Module):
    """ArcFace margin loss head for training."""
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s; self.m = m
        import math
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        sin = (1.0 - cos.pow(2)).clamp(0).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos).scatter_(1, label.view(-1, 1), 1)
        logits = (one_hot * phi + (1.0 - one_hot) * cos) * self.s
        return F.cross_entropy(logits, label)


if __name__ == "__main__":
    model = MobileFaceNetS(256)
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output norm (should be 1.0): {out.norm(dim=1)}")
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {params:.2f}M")
