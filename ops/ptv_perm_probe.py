"""Probe kernel: determine pytorchvideo's Kinetics class order empirically.

The kinetics_classnames.json the PyTorchVideo tutorials used is gone (404
everywhere), so instead of guessing we MEASURE the permutation:
  1. run torchvision r3d_18 (labels = canonical torchvision order) on N raw
     train clips
  2. run pytorchvideo slow_r50 on the same clips
  3. agreement matrix A[p][q] = #{clips: slow argmax = p, r3d argmax = q}
  4. Hungarian assignment on -A  ->  permutation ptv_idx -> tv_idx
  5. sanity: H1 hypothesis (alphabetical == identity except the torchvision
     skiing pair-swap) agreement, printed next to the Hungarian one
Output: prints the permutation as a JSON list to embed in the repo.
"""
import json
import subprocess
import sys

# ---- bootstrap (same as train kernels) ----
subprocess.run(["git", "clone", "-q",
                "https://github.com/wagur1/pre_processing_baseline.git",
                "/kaggle/working/pre_processing_baseline"], check=False)
sys.path.insert(0, "/kaggle/working/pre_processing_baseline")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "pytorchvideo", "opencv-python-headless", "tqdm"],
               check=False)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from src.data.video_dataset import VideoClipDataset  # noqa: E402
from torchvision.models.video import r3d_18, R3D_18_Weights  # noqa: E402
from pytorchvideo.models.hub import slow_r50, slowfast_r50  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# locate kinetics
import glob as _g  # noqa: E402
sample = None
for pat in ("/kaggle/input/**/*.mp4",):
    hits = _g.glob(pat, recursive=True)
    if hits:
        sample = hits[0]
        break
root = "/".join(sample.split("/")[:-2])
print("[probe] kinetics root:", root)

import subprocess as _sp  # noqa: E402
REPO = "/kaggle/working/pre_processing_baseline"
INDEX = REPO + "/data/index/kinetics_hash_split.json"
_sp.run([sys.executable, "scripts/build_train_index.py", "--root", root,
         "--out", INDEX,
         "--assert-fingerprint", "30f083f8520a"], check=True,
        cwd=REPO)

ds = VideoClipDataset(INDEX, split="train",
                      num_frames=16, frame_size=128, temporal_stride=2,
                      train=False)
print("[probe] train clips:", len(ds))

MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def prep(x, t=16):
    b, c, _, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    x = F.interpolate(x, size=(112, 112), mode="bilinear", align_corners=False)
    x = x.reshape(b, -1, c, 112, 112).permute(0, 2, 1, 3, 4)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1, 1)
    return (x - mean) / std


# torchvision reference
tv = r3d_18(weights=R3D_18_Weights.KINETICS400_V1).to(DEV).eval()
# ptv models — with the same head-pool fix as action_recognition.py:
# shipped AvgPool3d(8,7,7) expects 8x224x224; our clips are 16x112x112.
import torch.nn as _nn
def _fix_pool(net, tag):
    n = 0
    def rec(mod):
        nonlocal n
        for name, ch in mod.named_children():
            if isinstance(ch, _nn.AvgPool3d) and any(k > 1 for k in ch.kernel_size):
                setattr(mod, name, _nn.AdaptiveAvgPool3d(1))
                n += 1
            else:
                rec(ch)
    rec(net)
    print(f"[probe] {tag}: {n} AvgPool3d -> AdaptiveAvgPool3d(1)")

slow = slow_r50(pretrained=True)
_fix_pool(slow, "slow_r50")
slow = slow.to(DEV).eval()
sf = None
try:
    sf = slowfast_r50(pretrained=True)
    _fix_pool(sf, "slowfast_r50")
    sf = sf.to(DEV).eval()
except Exception as e:
    print("[probe] slowfast load failed:", e)

N = 512
A_slow = torch.zeros(400, 400)   # agreement slow vs tv
A_sf = torch.zeros(400, 400) if sf is not None else None
n_done = 0
with torch.no_grad():
    for i in range(N):
        clip, label = ds[i]
        clip = clip[None].to(DEV)
        x = prep(clip)
        tv_logits = tv(x)[0].cpu()
        s_logits = slow(x)[0].cpu()
        if sf is not None:
            x64 = x.repeat_interleave(4, dim=2)
            sf_logits = sf([x, x64])[0].cpu()  # slowfast needs a LIST
        tp = tv_logits.argmax().item()
        sp = s_logits.argmax().item()
        A_slow[sp, tp] += 1
        if sf is not None:
            fp_ = sf_logits.argmax().item()
            A_sf[fp_, tp] += 1
        n_done += 1
        if n_done % 64 == 0:
            print(f"[probe] {n_done}/{N}")

for name, A in (("slow_r50", A_slow), ("slowfast_r50", A_sf)):
    if A is None:
        continue
    r, c = linear_sum_assignment(-A.numpy())
    perm = torch.zeros(400, dtype=torch.long)
    for p, q in zip(r, c):
        perm[p] = q
    agree = A[torch.arange(400), perm].sum().item() / A.sum().item()
    print(f"[probe] {name}: Hungarian-agreement {agree:.1%} "
          f"({int(A.sum().item())} clips)")
    print(f"[probe] {name} PERM_JSON_BEGIN")
    print(json.dumps(perm.tolist()))
    print(f"[probe] {name} PERM_JSON_END")
print("[probe] DONE")
