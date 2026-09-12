#!/usr/bin/env python
"""Kaggle kernel source: canonical eval of the Round (b) checkpoint.

Notebook payload: %%bash cell that clones the v7 repo at a pinned commit,
downloads the trained checkpoint from the train kernel's output (attached as
a kernel data source), and runs the canonical protocol eval on a shard of
the 1159-clip test set:

  * held-out analyzer r2plus1d_18 (never a training teacher)
  * real x264 AND x265, preset medium, QP {30,35,40,45,50}
  * per-sequence records (top1 + target_prob + bpp per QP per codec)
  * eval.shard_idx / eval.num_shards split the test set deterministically
    (md5 of the mount-independent clip key) so 2-3 kernels cover all 1159

The merge + BD-rate + bootstrap CI happens locally (ops/merge_eval.py) on
the per-sequence JSONs this kernel emits.
"""

EVAL_BASH = r"""%%bash
set -euo pipefail
export PYTHONUNBUFFERED=1

cd /kaggle/working
REPO=/kaggle/working/pre_processing_baseline

if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch --all -q
  git -C "$REPO" checkout -q __COMMIT__
else
  git clone -q https://github.com/wagur1/pre_processing_baseline.git "$REPO"
  git -C "$REPO" checkout -q __COMMIT__
fi
cd "$REPO"

pip install -q opencv-python-headless pyyaml tqdm scipy matplotlib pandas 2>/dev/null | tail -1 || true
pip install -q pytorchvideo 2>/dev/null | tail -1 || true

# ---- dataset ----
KINETICS_ROOT=""
for c in /kaggle/input/kinetics-train-5per/train /kaggle/input/datasets/rohanmallick/kinetics-train-5per/kinetics400_5per; do
  [ -d "$c" ] && KINETICS_ROOT="$c" && break
done
if [ -z "$KINETICS_ROOT" ]; then
  sample=$(find /kaggle/input -maxdepth 8 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit)
  KINETICS_ROOT=$(dirname "$(dirname "$sample")")
fi
echo "[eval] Kinetics root: $KINETICS_ROOT"

INDEX=data/index/kinetics_hash_split.json
if [ ! -f "$INDEX" ]; then
  python scripts/build_train_index.py --root "$KINETICS_ROOT" --out "$INDEX" --assert-fingerprint 30f083f8520a
fi

# ---- checkpoint from the TRAIN kernel's attached output ----
# The train kernel's output = its whole /kaggle/working, so the checkpoint
# lives at /kaggle/input/<train-slug>/pre_processing_baseline/outputs/<run>/checkpoints/preprocessor.pth
# (do NOT exclude the repo-dir prefix — that excluded exactly this path).
CKPT_SRC=$(find /kaggle/input -name 'preprocessor.pth' -path '*outputs*' 2>/dev/null | head -1 || true)
if [ -z "$CKPT_SRC" ]; then
  # flat checkpoint datasets (file at the dataset root, no 'outputs' path component)
  CKPT_SRC=$(find /kaggle/input -name 'preprocessor.pth' 2>/dev/null | head -1 || true)
fi
if [ -z "$CKPT_SRC" ]; then
  echo "ERROR: no preprocessor.pth in /kaggle/input (attach the train kernel's output as a data source)" >&2
  exit 1
fi
echo "[eval] checkpoint: $CKPT_SRC"

SHARD_ARGS="__SHARD_ARGS__"
OUT=outputs/eval_qpc__SUFFIX__
mkdir -p "$OUT"

python evaluate.py --config __CONFIG__ \
    --ckpt "$CKPT_SRC" \
    --out "$OUT" \
    data.index="$INDEX" \
    __HELD_OUT__ \
    $SHARD_ARGS

echo "[eval] done"
ls -la "$OUT"
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commit", required=True)
    p.add_argument("--config", default="configs/upvcm_ar.yaml")
    p.add_argument("--shard-idx", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=3)
    p.add_argument("--analyzer", choices=["heldout", "teacher"], default="heldout",
                   help="heldout: eval.held_out_backbone=r2plus1d_18 (canonical); "
                        "teacher: no override -> task.backbone (on-teacher, Zhao's arm)")
    a = p.parse_args()

    shard_args = f"eval.shard_idx={a.shard_idx} eval.num_shards={a.num_shards}"
    held_out = "eval.held_out_backbone=r2plus1d_18" if a.analyzer == "heldout" else ""
    bash = (
        EVAL_BASH.replace("__COMMIT__", a.commit)
        .replace("__CONFIG__", a.config)
        .replace("__SHARD_ARGS__", shard_args)
        .replace("__SUFFIX__", f"shard{a.shard_idx}")
        .replace("__HELD_OUT__", held_out)
    )
    print(bash)


if __name__ == "__main__":
    main()
