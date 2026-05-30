"""
MobileFaceNet recognizer — the doc's model, rebuilt correctly.

The original blueprint flattened a 7x7x512 feature map straight into a
Dense(25088 -> 256) layer (6.4M params, ~13MB in fp16). That is the entire
reason the old export was ~15MB. Canonical MobileFaceNet (Chen et al., 2018)
instead collapses the spatial dims with a *linear global depthwise conv*
(GDConv) before the embedding projection, which keeps the whole model at
~1.0M params. Same input (112x112), same 512-d ArcFace-style embedding.

Train with the ArcFaceHead below, or load pretrained weights (see README).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, groups=1, use_prelu=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c) if use_prelu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """MobileNetV2-style inverted residual with a linear projection."""
    def __init__(self, in_c, out_c, stride, expansion):
        super().__init__()
        mid = in_c * expansion
        self.use_res = stride == 1 and in_c == out_c
        self.conv = nn.Sequential(
            ConvBlock(in_c, mid, 1, 1, 0),                       # expand (pointwise)
            ConvBlock(mid, mid, 3, stride, 1, groups=mid),       # depthwise
            ConvBlock(mid, out_c, 1, 1, 0, use_prelu=False),     # project (linear)
        )

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out


class MobileFaceNet(nn.Module):
    """
    Canonical MobileFaceNet. ~1.0M params for embedding_dim=512.

    Args:
        embedding_dim: output embedding size (512 standard; 256 also fine).
        normalize_output: if True, L2-normalise the embedding in-graph. Set
            True for export/inference so the app gets unit vectors directly;
            set False for training (ArcFaceHead normalises internally).
    """
    def __init__(self, embedding_dim: int = 512, normalize_output: bool = False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize_output = normalize_output

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)                  # 112 -> 56
        self.dw_conv = ConvBlock(64, 64, 3, 1, 1, groups=64)    # 56 -> 56

        # (in, out, stride, expansion, repeats)
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
                blocks.append(Bottleneck(in_c if i == 0 else out_c,
                                         out_c, s if i == 0 else 1, t))
        self.blocks = nn.Sequential(*blocks)                    # -> 7x7x128

        self.conv_sep = ConvBlock(128, 512, 1, 1, 0)            # 7x7x512
        # Linear GDConv: depthwise 7x7 over a 7x7 map -> 1x1x512, NO activation.
        self.gdconv = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 0, groups=512, bias=False),
            nn.BatchNorm2d(512),
        )
        self.conv_out = nn.Conv2d(512, embedding_dim, 1, 1, 0, bias=False)
        self.bn_out = nn.BatchNorm2d(embedding_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.blocks(x)
        x = self.conv_sep(x)
        x = self.gdconv(x)
        x = self.bn_out(self.conv_out(x))
        x = x.view(x.size(0), -1)
        if self.normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x


class ArcFaceHead(nn.Module):
    """Additive angular margin head (Deng et al., 2019) for training only."""
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s, self.m = s, m
        self.cos_m, self.sin_m = math.cos(m), math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, label):
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))
        sin = (1.0 - cos.pow(2)).clamp(0, 1).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        one_hot = torch.zeros_like(cos).scatter_(1, label.view(-1, 1), 1.0)
        logits = (one_hot * phi + (1.0 - one_hot) * cos) * self.s
        return F.cross_entropy(logits, label)


def build_recognizer(embedding_dim: int = 512,
                     weights: str = None,
                     normalize_output: bool = True,
                     device: str = "cpu") -> MobileFaceNet:
    """Construct the recognizer and optionally load a checkpoint."""
    model = MobileFaceNet(embedding_dim=embedding_dim, normalize_output=normalize_output)
    if weights:
        state = torch.load(weights, map_location="cpu")
        state = state.get("state_dict", state) if isinstance(state, dict) else state
        # tolerate "module." prefixes from DataParallel checkpoints
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[recognizer] loaded with missing={len(missing)} unexpected={len(unexpected)} keys")
    return model.to(device).eval()


if __name__ == "__main__":
    m = MobileFaceNet(512)
    x = torch.randn(2, 3, 112, 112)
    y = m(x)
    print("output:", tuple(y.shape))
    print(f"params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f}M")
