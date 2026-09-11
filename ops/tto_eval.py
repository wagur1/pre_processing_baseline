#!/usr/bin/env python
"""Test-time optimization (TTO) for the UP-VCM preprocessor — lever 2.

Per-clip, at eval time: refine the amortized edit by optimizing a small
residual ``d`` on top of ``pre(x)`` with gradient descent that needs NO
analyzer (deploy-legal: the optimizer sees only the source clip, the frozen
proxy codec, and a frozen DINOv2 feature extractor):

    x_pre = clamp(pre(x, cond) + d)          d init 0
    L(d)  = ||DINO(proxy(x_pre)) − DINO(x)||²   (feature target, once per clip)
          + kappa · adaptive_dct(x_pre)          (bit proxy, lineage value)
          + lam   · ||d||₁                        (stay near the amortized edit)

One ``d`` per clip, conditioned at the mid QP (40) and shared across the QP
grid (the QP-conditional structure stays inside ``pre``). The real
x264/x265 encode + held-out analyzer scoring then run exactly as in the
canonical eval — TTO never sees the analyzer.

Output: per-sequence ``sequence_points.csv`` with methods ``h264/h265``
(anchor) and ``tto+h264/tto+h265`` — merge with the ordinary eval shards via
ops/merge_eval.py conventions (or compare directly within this file's rows).

    python ops/tto_eval.py --ckpt <pth> --index <json> --out outputs/tto_screen \
        --limit 100                       # direction screen first
    python ops/tto_eval.py --ckpt <pth> --index <json> --out outputs/tto_s0 \
        --shard-idx 0 --num-shards 2      # full run, sharded
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.codecs import StandardCodec, ffmpeg_available  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.video_dataset import VideoClipDataset, collate_clips  # noqa: E402
from src.losses import adaptive_dct_loss  # noqa: E402
from src.models.dino_saliency import get_dino  # noqa: E402
from src.models.upvcm import UPVCMPreprocessor  # noqa: E402
from src.models.virtual_codec import VirtualCodec  # noqa: E402

# Sandwich checkpoints (v8): the v8 'src' package is registered under the
# alias 'v8src' (this script already has v7's 'src' imported — a plain import
# would resolve inside v7 and fail). Relative imports inside sandwich.py then
# resolve within the aliased package.
SandwichPreprocessor = None
import importlib as _imp
import importlib.util as _ilu
for _v8 in (Path("/home/wagur1/pre_processing_upgrade_8"),   # local machine
            Path("/kaggle/working/pre_processing_upgrade_8")):  # kernel
    _v8_src = _v8 / "src"
    if (_v8_src / "models" / "sandwich.py").exists():
        try:
            _init = _v8_src / "__init__.py"
            _spec = _ilu.spec_from_file_location(
                "v8src", _init if _init.exists() else None,
                submodule_search_locations=[str(_v8_src)])
            _pkg = _ilu.module_from_spec(_spec)
            sys.modules["v8src"] = _pkg
            if _init.exists():
                _spec.loader.exec_module(_pkg)
            _sand = _imp.import_module("v8src.models.sandwich")
            SandwichPreprocessor = _sand.SandwichPreprocessor
            print(f"[tto] SandwichPreprocessor loaded from {_v8}")
            break
        except Exception as _e:
            sys.modules.pop("v8src", None)
            print(f"[tto] v8 load failed ({type(_e).__name__}: {_e}); POST will be bypassed")
            SandwichPreprocessor = None
from src.tasks.base import build_analyzer  # noqa: E402

QPS = [30, 35, 40, 45, 50]
_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


def _clip_key(path: str) -> str:
    parts = str(path).replace("\\", "/").rstrip("/").split("/")
    return "/".join(parts[-2:])


def _qp_norm(qp: float, lo: float = 20.0, hi: float = 51.0) -> float:
    return (qp - lo) / (hi - lo)


def _dino_frames(x: torch.Tensor) -> torch.Tensor:
    """[B,3,T,H,W] -> [B*T,3,S,S] normalized (S = patch multiple)."""
    b, c, t, h, w = x.shape
    size = max(14, min(h, w) // 14 * 14)
    fr = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    fr = F.interpolate(fr, size=(size, size), mode="bilinear", align_corners=False)
    mean = torch.tensor(_DINO_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_DINO_STD, device=x.device).view(1, 3, 1, 1)
    return (fr - mean) / std


def _dino_feat(x: torch.Tensor, dino) -> torch.Tensor:
    """DINOv2 patch tokens (differentiable w.r.t. x when x requires grad)."""
    return dino.forward_features(_dino_frames(x))["x_norm_patchtokens"]


def refine_clip(pre, proxy, dino, clip: torch.Tensor, steps: int, lr: float,
                lam: float, kappa: float, quality: int = 3) -> torch.Tensor:
    """Optimize the per-clip residual d on top of the amortized edit.

    Conditioned at the mid QP (40); the residual is shared across the grid.
    """
    cond = torch.full((clip.shape[0], 1), _qp_norm(40), device=clip.device,
                      dtype=clip.dtype)
    with torch.no_grad():
        base = pre(clip, cond)
        target = _dino_feat(clip, dino).detach()
    d = torch.zeros_like(base, requires_grad=True)
    opt = torch.optim.Adam([d], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        x_pre = (base + d).clamp(0.0, 1.0)
        x_hat, _ = proxy(x_pre, quality)
        lf = F.mse_loss(_dino_feat(x_hat, dino), target)
        lk = kappa * 0.5 * (adaptive_dct_loss(x_pre, block=8)
                            + adaptive_dct_loss(x_pre, block=16))
        ll = lam * d.abs().mean()
        (lf + lk + ll).backward()
        opt.step()
    with torch.no_grad():
        return (base + d).clamp(0.0, 1.0)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ckpt", required=True)
    a.add_argument("--index", required=True)
    a.add_argument("--config", default="configs/upvcm_ar.yaml")
    a.add_argument("--out", required=True)
    a.add_argument("--limit", type=int, default=0, help="0 = all")
    a.add_argument("--shard-idx", type=int, default=None)
    a.add_argument("--num-shards", type=int, default=1)
    a.add_argument("--steps", type=int, default=30)
    a.add_argument("--lr", type=float, default=2e-3)
    a.add_argument("--lam", type=float, default=0.05)
    a.add_argument("--kappa", type=float, default=10.0)
    a.add_argument("--batch-size", type=int, default=4)
    args = a.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location="cpu")
    model_state = state["model"] if "model" in state else state
    cm = (state.get("cfg") or {}).get("model", {})
    post_restore = None
    if any(k.startswith("post_net.") for k in model_state):
        if SandwichPreprocessor is None:
            raise SystemExit("sandwich checkpoint but v8 SandwichPreprocessor not loadable")
        sand = SandwichPreprocessor(
            s_ch=int(cm.get("s_ch", 16)), editor_ch=int(cm.get("editor_ch", 24)),
            cond_dim=int(cm.get("cond_dim", 1)),
            dino_weight=float(cm.get("dino_weight", 0.5)),
            dino_name=str(cm.get("dino_name", "dinov2_vits14")),
            motion_tau=float(cm.get("motion_tau", 0.1)),
            post_base=int(cm.get("post_base", 32))).to(device)
        sand.load_state_dict(model_state, strict=True)
        sand.eval()
        pre = sand.pre
        post_restore = sand.post_restore
        print("[tto] sandwich checkpoint: PRE refine + POST after codec")
    else:
        pre = UPVCMPreprocessor(
            s_ch=int(cm.get("s_ch", 16)), editor_ch=int(cm.get("editor_ch", 24)),
            cond_dim=int(cm.get("cond_dim", 1)),
            dino_weight=float(cm.get("dino_weight", 0.5)),
            dino_name=str(cm.get("dino_name", "dinov2_vits14")),
            motion_tau=float(cm.get("motion_tau", 0.1))).to(device)
        pre.load_state_dict(model_state, strict=True)
        pre.eval()

    cc = cfg["codec"]
    proxy = VirtualCodec(
        qualities=tuple(cc.get("qualities", [1, 2, 3, 5, 8])),
        block=cc.get("block", 8), q_steps=cc.get("q_steps"),
        step_coarse=cc.get("step_coarse", 0.25),
        step_fine=cc.get("step_fine", 0.03), inter=cc.get("inter", True),
        colorspace=cc.get("colorspace", "yuv420"),
        chroma_step_scale=cc.get("chroma_step_scale", 2.0),
    ).to(device).eval()

    dino = get_dino(cm.get("dino_name", "dinov2_vits14"), device=device)
    if dino is None:
        raise SystemExit("TTO requires DINOv2 (feature target); load failed")

    analyzer = build_analyzer(cfg, role="eval").to(device)
    analyzer.eval()

    ds = VideoClipDataset(index_json=args.index, split="test",
                          num_frames=cfg["data"].get("num_frames", 16),
                          frame_size=cfg["data"].get("frame_size", 128),
                          temporal_stride=cfg["data"].get("temporal_stride", 2),
                          train=False, return_metadata=True)
    if args.shard_idx is not None and args.num_shards > 1:
        ds.samples = [s for s in ds.samples
                      if int(hashlib.md5(_clip_key(s["path"]).encode()).hexdigest()[:8], 16)
                      % args.num_shards == args.shard_idx]
        print(f"[tto] shard {args.shard_idx}/{args.num_shards}: {len(ds.samples)}")
    if args.limit:
        ds.samples = ds.samples[:args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_clips)

    if not ffmpeg_available():
        raise SystemExit("TTO eval needs real x264/x265 via ffmpeg")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    preset = cfg.get("eval", {}).get("preset", "medium")

    n_clips = 0
    with open(out_dir / "sequence_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sequence_id", "path", "class", "codec", "qp",
                    "bpp", "top1", "target_prob"])
        for clips, labels, metadata in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            # refine_clip manages its own grad context (internal optimization,
            # detached return) — do NOT wrap it in no_grad here.
            x_pre = torch.cat([
                refine_clip(pre, proxy, dino, clips[i:i + 1],
                            steps=args.steps, lr=args.lr,
                            lam=args.lam, kappa=args.kappa)
                for i in range(clips.shape[0])], dim=0)
            for name in ("h264", "h265"):
                for qp in QPS:
                    sc = StandardCodec(codec=name, qp=qp, preset=preset)
                    xh, bpps = sc.compress_decompress_items(clips)
                    xhp, bppps = sc.compress_decompress_items(x_pre.cpu())
                    if post_restore is not None:
                        with torch.no_grad():
                            c_ = torch.full((clips.shape[0], 1), _qp_norm(qp),
                                            device=clips.device, dtype=clips.dtype)
                            xh = post_restore(xh.to(device), c_).cpu()
                            xhp = post_restore(xhp.to(device), c_).cpu()
                    logits = analyzer.predict(xh.to(device))
                    logits_p = analyzer.predict(xhp.to(device))
                    probs = logits.softmax(dim=1)
                    probs_p = logits_p.softmax(dim=1)
                    for i, meta in enumerate(metadata):
                        for method, b, lg, pb in (
                            (name, bpps[i], logits[i], probs[i]),
                            (f"tto+{name}", bppps[i], logits_p[i], probs_p[i]),
                        ):
                            w.writerow([
                                meta["sequence_id"], meta["path"], meta["class"],
                                method, qp, f"{float(b):.8f}",
                                int((lg.argmax() == labels[i]).item()),
                                f"{float(pb[labels[i]]):.8f}"])
            n_clips += clips.shape[0]
            print(f"[tto] {n_clips}/{len(ds)} clips refined", flush=True)
    print(f"[tto] wrote {out_dir/'sequence_points.csv'} ({n_clips} clips)")


if __name__ == "__main__":
    main()
