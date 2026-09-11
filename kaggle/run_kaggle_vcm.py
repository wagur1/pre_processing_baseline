"""Kaggle smoke benchmark cho Kinetics-400-5%.

Chay trong mot cell:
    !python /kaggle/working/preprocessing_upgrade_5/kaggle/run_kaggle_vcm.py \
      --data /kaggle/input/kinetics400-5per --videos 8

Script tu tim *.mp4, lay mot so video, encode H.264 o QP 22/27/32/37,
tinh bitrate + PSNR/SSIM xap xi va BD-rate. Quality nay la proxy thi giac;
hay thay `quality()` bang mAP/mIoU/HOTA cua model VCM khi lam benchmark nghien cuu.
"""
from __future__ import annotations
import argparse, json, math, os, shutil, subprocess, tempfile
from pathlib import Path
import sys

# Cho phep chay script bang duong dan tuyet doi trong Kaggle.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from vcm_preprocess.metrics import bd_rate

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode: raise RuntimeError(p.stderr[-2000:])
    return p

def duration(path):
    p = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)])
    return max(float(p.stdout.strip()), 1e-6)

def quality(original, decoded):
    """PSNR + SSIM proxy tren 12 frame mau, tra ve thang diem 0..1."""
    cap0, cap1 = cv2.VideoCapture(str(original)), cv2.VideoCapture(str(decoded))
    vals, ssim = [], []
    while len(vals) < 12:
        ok0, a = cap0.read(); ok1, b = cap1.read()
        if not ok0 or not ok1: break
        a = cv2.resize(a, (320, 180)); b = cv2.resize(b, (320, 180))
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        vals.append(10 * math.log10((255.0 ** 2) / max(mse, 1e-8)))
        # SSIM xap xi theo kenh Y, khong can scikit-image.
        ay, by = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        mu1, mu2 = ay.mean(), by.mean(); v1, v2 = ay.var(), by.var(); cov = np.mean((ay-mu1)*(by-mu2))
        c1, c2 = 6.5025, 58.5225
        ssim.append(((2*mu1*mu2+c1)*(2*cov+c2))/((mu1*mu1+mu2*mu2+c1)*(v1+v2+c2)))
    cap0.release(); cap1.release()
    if not vals: return 0.0
    psnr_norm = max(0.0, min(1.0, (sum(vals)/len(vals) - 20.0) / 35.0))
    return 0.5 * psnr_norm + 0.5 * max(0.0, min(1.0, sum(ssim)/len(ssim)))

def encode(src, dst, qp, vf=None):
    # QP co dinh giup tao duong cong RD de tinh BD-rate.
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if vf: cmd += ["-vf", vf]
    cmd += ["-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-qp", str(qp), "-an", str(dst)]
    run(cmd)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/kaggle/input/kinetics400-5per")
    ap.add_argument("--videos", type=int, default=8); ap.add_argument("--out", default="/kaggle/working/vcm_results.json")
    args = ap.parse_args(); root = Path(args.data)
    videos = sorted(root.rglob("*.mp4"))[:args.videos]
    if not videos: raise SystemExit(f"Khong tim thay MP4 trong {root}")
    qps = [22, 27, 32, 37]; rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, src in enumerate(videos):
            d = duration(src); ref, pp = [], []
            for qp in qps:
                a, b = td/f"{i}_ref_{qp}.mp4", td/f"{i}_pp_{qp}.mp4"
                encode(src, a, qp); encode(src, b, qp, "hqdn3d=1.0:1.0:3:3,eq=contrast=1.05:brightness=0.01,unsharp=5:5:0.25")
                ref.append((a.stat().st_size*8/d/1000, quality(src, a))); pp.append((b.stat().st_size*8/d/1000, quality(src, b)))
            rows.append({"video": str(src.relative_to(root)), "reference": ref, "preprocessed": pp, "bd_rate_percent": bd_rate(ref, pp)})
            print(f"[{i+1}/{len(videos)}] {src.name}: BD-rate={rows[-1]['bd_rate_percent']:.2f}%")
    result = {"dataset": str(root), "videos": len(rows), "rows": rows, "mean_bd_rate_percent": sum(x["bd_rate_percent"] for x in rows)/len(rows)}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
