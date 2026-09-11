"""Action-recognition analyzer (paper's primary task, Kinetics-400).

Uses a frozen torchvision video model pretrained on Kinetics-400 (default
``r3d_18``). The analyzer is never trained -- it only scores clips:

  * training : accuracy loss L_Acc = cross-entropy(logits, label).
  * eval     : top-k predictions for top-1..top-5 accuracy.

The pretrained weights carry the canonical 400-class ordering
(``weights.meta['categories']``). ``kinetics_category_index`` exposes a
name -> class-index map so the data layer can convert dataset folder names
(e.g. ``abseiling``) into the exact label index this frozen model expects --
which is what makes zero-shot accuracy on the frozen analyzer meaningful.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
import json as _json
from pathlib import Path

import torch.nn.functional as F
from torchvision.models.video import (
    mc3_18,
    MC3_18_Weights,
    r2plus1d_18,
    R2Plus1D_18_Weights,
    r3d_18,
    R3D_18_Weights,
)

from .base import TaskAnalyzer

_BACKBONES = {
    "r3d_18": (r3d_18, R3D_18_Weights.KINETICS400_V1),
    "mc3_18": (mc3_18, MC3_18_Weights.KINETICS400_V1),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights.KINETICS400_V1),
}

# Zhao (arXiv:2512.15331) Table 1 backbones, via pytorchvideo (lazy import):
# slowonly (the paper's best/claim-max backbone) and slowfast (its weakest).
_PTV_BACKBONES = ("slowonly", "slowfast")


def _build_ptv(name: str, clip_size: int):
    """torchvision-style (net, categories) for pytorchvideo models.

    pytorchvideo hub models emit logits in PYTORCHVIDEO's Kinetics-400 class
    order, which differs from torchvision's. We fetch the official
    kinetics_classnames.json (kernel has internet) and build a permutation
    ptv_idx -> torchvision_idx so labels from the canonical index (built with
    torchvision category order) line up. If the fetch fails the permutation
    stays identity and the anchor sanity check will scream (~0.3% acc).
    """
    import json as _json
    import urllib.request as _url
    try:
        # NOTE: pytorchvideo's hub exports SlowOnly R50 as `slow_r50`
        # (there is no slowonly_r50 symbol — the first push died on this).
        from pytorchvideo.models.hub import slow_r50, slowfast_r50
    except ImportError as e:
        raise ImportError(
            "Zhao backbones need pytorchvideo (pip install pytorchvideo)") from e
    if name == "slowonly":
        net = slow_r50(pretrained=True)
    elif name == "slowfast":
        net = slowfast_r50(pretrained=True)
    else:
        raise ValueError(name)
    net = net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    # ptv ships head AvgPool3d((8,7,7)) sized for 8x224x224 inputs; our
    # 16x112x112 clips reduce feature maps to T16 H4 W4 -> the fixed kernel
    # no longer fits (the Net is blocks-based, the pool lives at
    # blocks[5].pool — an earlier .head-based fix silently missed it).
    # Replace RECURSIVELY; global average is the same op, pools carry no
    # learned parameters. Verified locally: forward 16x112 -> [B,400].
    import torch.nn as _nn

    def _fix_pool(mod):
        for _name, _ch in mod.named_children():
            if isinstance(_ch, _nn.AvgPool3d) and any(k > 1 for k in _ch.kernel_size):
                setattr(mod, _name, _nn.AdaptiveAvgPool3d(1))
            else:
                _fix_pool(_ch)

    _fix_pool(net)

    # Permutation: measured empirically by ops/ptv_perm_probe.py (512 raw
    # train clips, Hungarian assignment on slow-vs-r3d_18 argmax agreement)
    # and stored in data/ptv_perm.json. The tutorial's kinetics_classnames
    # json is gone from the internet — the file IS the source of truth now.
    perm = torch.arange(400)
    perm_path = Path(__file__).resolve().parents[2] / "data" / "ptv_perm.json"
    try:
        perm = torch.tensor(_json.loads(perm_path.read_text()), dtype=torch.long)
        if perm.shape != (400,) or (perm.sort().values != torch.arange(400)).any():
            raise ValueError("not a valid permutation of 0..399")
        print(f"[ptv] permutation loaded from data/ptv_perm.json")
    except FileNotFoundError:
        print("[ptv] FATAL data/ptv_perm.json missing — run ops/ptv_perm_probe.py")
        raise
    return net, perm

# Kinetics normalisation used by torchvision video weights.
_MEAN = (0.43216, 0.394666, 0.37645)
_STD = (0.22803, 0.22145, 0.216989)


def _canon(name: str) -> str:
    """Normalise a class name for matching (lowercase, alnum-joined)."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


@lru_cache(maxsize=None)
def kinetics_category_index(backbone: str = "r3d_18") -> Dict[str, int]:
    """Map canonicalised Kinetics-400 class name -> label index.

    Always TORCHVISION order (the canonical index.json was built with it);
    ptv backbones permute their logits to match (see _build_ptv).
    """
    _, weights = _BACKBONES["r3d_18"]
    cats: List[str] = weights.meta["categories"]
    return {_canon(c): i for i, c in enumerate(cats)}


def kinetics_categories(backbone: str = "r3d_18") -> List[str]:
    _, weights = _BACKBONES[backbone]
    return list(weights.meta["categories"])


class ActionRecognitionAnalyzer(TaskAnalyzer):
    """Frozen video classifier used as the machine-vision analyzer."""

    def __init__(self, backbone: str = "r3d_18", clip_size: int = 112):
        super().__init__()
        self.task_name = "action_recognition"
        self.backbone_name = backbone
        self.clip_size = clip_size
        if backbone in _PTV_BACKBONES:
            # Zhao Table-1 backbones (pytorchvideo). SlowFast's fast pathway
            # wants 64 frames: repeat/interp the 16-frame clip temporally.
            net, perm = _build_ptv(backbone, clip_size)
            self.net = net
            self.is_ptv = True
            self.register_buffer("logit_perm", perm)
        else:
            if backbone not in _BACKBONES:
                raise ValueError(f"unknown backbone '{backbone}'")
            ctor, weights = _BACKBONES[backbone]
            self.net = ctor(weights=weights)
            self.is_ptv = False
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1, 1))

    # -- input prep --------------------------------------------------------
    def _prep(self, x):
        """[B,C,T,H,W] in [0,1] -> resized to clip_size, Kinetics-normalised.

        SlowFast (ptv) needs a LIST of two pathway tensors: slow (T=16) +
        fast (T=64). The 16-frame clip is repeated 4x for the fast pathway
        (keeps motion periodicity for the alpha=8 subsampling) — verified
        locally: forward works and emits [B,400].
        """
        b, c, t, h, w = x.shape
        slowfast = self.is_ptv and self.backbone_name == "slowfast"
        if slowfast and t < 64:
            fast = x.repeat_interleave(64 // t, dim=2)
        else:
            fast = x
        def _resize(v):
            tt = v.shape[2]
            if (h, w) != (self.clip_size, self.clip_size):
                v = v.permute(0, 2, 1, 3, 4).reshape(b * tt, c, h, w)
                v = F.interpolate(
                    v, size=(self.clip_size, self.clip_size),
                    mode="bilinear", align_corners=False)
                v = v.reshape(b, tt, c, self.clip_size, self.clip_size)
                v = v.permute(0, 2, 1, 3, 4)
            return (v - self.mean) / self.std
        if slowfast:
            return [_resize(x), _resize(fast)]
        return _resize(x)

    # -- training ----------------------------------------------------------
    def accuracy_loss(
        self, x_hat: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        logits = self.net(self._prep(x_hat))
        if self.is_ptv:
            logits = logits[:, self.logit_perm]  # ptv order -> torchvision order
        loss = F.cross_entropy(logits, target)
        return loss, {"logits": logits.detach()}

    # -- feature distillation ---------------------------------------------
    def features(self, x: torch.Tensor) -> list:
        """Frozen r3d_18 semantic features (stem, layer1, layer2) for distill.

        ptv backbones return [] — feature_distillation's contract is "no
        features -> 0 loss" (omega is 0 in Zhao-pure configs anyway; the
        earlier NotImplementedError broke the unconditional call at
        losses.py:265 and killed the first train step).
        """
        if self.is_ptv:
            return []
        h = self._prep(x)
        net = self.net
        feats = []
        h = net.stem(h)
        feats.append(h)
        h = net.layer1(h)
        feats.append(h)
        h = net.layer2(h)
        feats.append(h)
        return feats

    # -- evaluation --------------------------------------------------------
    @torch.no_grad()
    def predict(self, x_hat: torch.Tensor) -> torch.Tensor:
        logits = self.net(self._prep(x_hat))
        if self.is_ptv:
            logits = logits[:, self.logit_perm]
        return logits  # [B, 400], torchvision class order
