"""Probe v5: per-CLASS-FOLDER majority vote for the ptv label mapping.

Probe v4's clip-level Hungarian on 512 random clips was too sparse (400
classes, 1.28 clips/class avg). This probe walks the TRAIN set BY CLASS
FOLDER (up to K clips per folder) and takes the per-folder MODE argmax:
folder (canonical tv idx) -> modal ptv idx = the permutation directly.
"""
import collections
import json
import subprocess
import sys

subprocess.run(["git", "clone", "-q",
                "https://github.com/wagur1/pre_processing_baseline.git",
                "/kaggle/working/pre_processing_baseline"], check=False)
sys.path.insert(0, "/kaggle/working/pre_processing_baseline")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pytorchvideo", "opencv-python-headless", "tqdm"],
               check=False)

import glob as _g
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.video_dataset import VideoClipDataset
from src.tasks.action_recognition import kinetics_category_index
from pytorchvideo.models.hub import slow_r50, slowfast_r50

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REPO = "/kaggle/working/pre_processing_baseline"

sample = _g.glob("/kaggle/input/**/*.mp4", recursive=True)
root = "/".join(sample[0].split("/")[:-2])
print("[probe] kinetics root:", root)
subprocess.run([sys.executable, "scripts/build_train_index.py", "--root", root,
                "--out", REPO + "/data/index/kinetics_hash_split.json",
                "--assert-fingerprint", "30f083f8520a"], check=True, cwd=REPO)

ds = VideoClipDataset(REPO + "/data/index/kinetics_hash_split.json",
                      split="train", num_frames=16, frame_size=128,
                      temporal_stride=2, train=False)
print("[probe] train clips:", len(ds))

MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def prep(x):
    b, c, t, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
    x = x.reshape(b, t, c, 112, 112).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


def fix_pool(net):
    def rec(mod):
        for name, ch in mod.named_children():
            if isinstance(ch, nn.AvgPool3d) and any(k > 1 for k in ch.kernel_size):
                setattr(mod, name, nn.AdaptiveAvgPool3d(1))
            else:
                rec(ch)
    rec(net)


models = {}
for name, ctor in (("slow_r50", slow_r50), ("slowfast_r50", slowfast_r50)):
    m = ctor(pretrained=True)
    fix_pool(m)
    models[name] = m.to(DEV).eval()
    print(f"[probe] {name} loaded + pool fixed")

canon = kinetics_category_index()
K = 8
by_folder = collections.defaultdict(list)
for i, rec in enumerate(ds.samples):
    by_folder[rec["class"]].append(i)

votes = {name: collections.defaultdict(collections.Counter) for name in models}
done = 0
with torch.no_grad():
    for folder, idxs in sorted(by_folder.items()):
        for i in idxs[:K]:
            clip, _ = ds[i]
            x = prep(clip[None].to(DEV))
            x64 = x.repeat_interleave(4, dim=2)
            for name, m in models.items():
                inp = [x, x64] if name == "slowfast_r50" else x
                votes[name][folder][m(inp).argmax().item()] += 1
            done += 1
            if done % 256 == 0:
                print(f"[probe] {done} clips")

tv_names = canon
for name in models:
    perm = torch.arange(400)
    confs, mapped, weak = [], 0, 0
    for folder, ctr in votes[name].items():
        tv_idx = tv_names.get(folder)
        if tv_idx is None:
            print(f"[probe] {name}: folder '{folder}' NOT in canon index!")
            continue
        ptv_idx, cnt = ctr.most_common(1)[0]
        conf = cnt / sum(ctr.values())
        confs.append(conf)
        if conf >= 0.5:
            perm[ptv_idx] = tv_idx
            mapped += 1
        else:
            weak += 1
    print(f"[probe] {name}: mapped {mapped} classes (weak {weak}), "
          f"mean majority {sum(confs)/len(confs):.1%}")
    print(f"[probe] {name} PERM_JSON_BEGIN")
    print(json.dumps(perm.tolist()))
    print(f"[probe] {name} PERM_JSON_END")
print("[probe] DONE")
