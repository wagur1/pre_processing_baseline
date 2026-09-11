#!/usr/bin/env bash
# =============================================================================
# upgrade-3 end-to-end on Kaggle (T4/P100). See docs/KAGGLE.md for details.
# Usage:  bash kaggle/run.sh            # full: stage1 -> stage2(STE) -> eval
#         SMOKE=1 bash kaggle/run.sh    # tiny capped run to check wiring
# =============================================================================
set -euo pipefail

trap 'echo "ERROR: kaggle/run.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR

CFG=configs/universal_action_recognition.yaml
INDEX=${INDEX:-data/index/kinetics_3gb.json}
KINETICS_ROOT=${KINETICS_ROOT:-/kaggle/input/kinetics-train-5per/train}
HELD_OUT=${HELD_OUT:-r2plus1d_18}
SEED=${SEED:-0}
OUT_DIR=${OUT_DIR:-outputs/seed_$SEED}
STE_MAX_STEPS=${STE_MAX_STEPS:-400}
[[ "$OUT_DIR" == outputs/* ]] || { echo "ERROR: OUT_DIR must be under outputs/" >&2; exit 1; }
CKPT_DIR=$OUT_DIR/checkpoints
EVAL_STAGE1=$OUT_DIR/eval_stage1
EVAL_STAGE2=$OUT_DIR/eval_stage2
CAP=""
if [[ "${SMOKE:-0}" == "1" ]]; then CAP="train.max_steps=200"; fi

echo "== 0. deps + codec check =="
python -m pip install -q compressai av
ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
grep -q libx264 <<<"$ENCODERS" || { echo "ERROR: ffmpeg libx264 missing" >&2; exit 1; }
grep -q libx265 <<<"$ENCODERS" || { echo "ERROR: ffmpeg libx265 missing" >&2; exit 1; }

echo "== 1. build <=3GB balanced index =="
if [[ ! -d "$KINETICS_ROOT" ]]; then
  KINETICS_ROOT=$(find /kaggle/input -maxdepth 5 -type d -name train -print -quit)
fi
if [[ ! -d "$KINETICS_ROOT" ]]; then
  sample=$(find /kaggle/input -maxdepth 8 -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit)
  if [[ -n "$sample" ]]; then
    candidate=$(dirname "$(dirname "$sample")")
    [[ -d "$candidate" ]] && KINETICS_ROOT="$candidate"
  fi
fi
[[ -n "$KINETICS_ROOT" && -d "$KINETICS_ROOT" ]] || { echo "ERROR: Kinetics train directory not found"; exit 1; }
find "$KINETICS_ROOT" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit | grep -q . || {
  echo "ERROR: no video files found under $KINETICS_ROOT" >&2
  exit 1
}
echo "[data] Kinetics root = $KINETICS_ROOT"
if [[ ! -f "$INDEX" ]] || ! python -c 'import json,sys; sys.exit("test" not in json.load(open(sys.argv[1])))' "$INDEX"; then
  python -m src.data.prepare_3gb --root "$KINETICS_ROOT" --out "$INDEX" \
      --cap-gb 3.0 --val-frac 0.1 --test-frac 0.1 --backbone r3d_18
fi

echo "== 2. STAGE 1: proxy pretrain (A1+A2+A3 anneal) =="
# Do not let an old collapsed checkpoint survive a failed/partial rerun.
if [[ "${CLEAN_RUN:-1}" == "1" ]]; then
  rm -rf -- "$CKPT_DIR" "$EVAL_STAGE1" "$EVAL_STAGE2"
fi
python train.py --config "$CFG" data.index="$INDEX" seed="$SEED" out_dir="$OUT_DIR" train.epochs=5 $CAP
cp "$CKPT_DIR/preprocessor.pth" "$CKPT_DIR/preprocessor_stage1.pth"
echo "[ckpt] backed up Stage-1 best -> $CKPT_DIR/preprocessor_stage1.pth"

echo "== 3. eval STAGE 1 only (proxy pretrain, no STE) =="
python evaluate.py --config "$CFG" \
    --ckpt "$CKPT_DIR/preprocessor_stage1.pth" --out "$EVAL_STAGE1" \
    data.index="$INDEX" seed="$SEED" out_dir="$OUT_DIR" eval.held_out_backbone="$HELD_OUT"

# Fail fast instead of spending another ~20 minutes calibrating an erased clip.
# Smoke runs only validate wiring and skip this statistical threshold.
if [[ "${SMOKE:-0}" != "1" ]]; then
python - "$EVAL_STAGE1/results.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
prep = [c for name, c in r.get("curves", {}).items() if name.startswith("prep+")]
best = max((max(c.get("accuracy", [0.0])) for c in prep), default=0.0)
if best < 0.05:
    raise SystemExit(
        f"ERROR: Stage-1 preprocessor health check failed: best prep accuracy={best:.4f}. "
        "Do not run STE on this checkpoint."
    )
print(f"[health] Stage-1 best preprocessed accuracy={best:.4f} -> continue to STE")
PY
fi

echo "== 4. STAGE 2: real-codec (STE) calibration fine-tune (loads Stage-1, overwrites best) =="
python train.py --config "$CFG" data.index="$INDEX" \
    codec.kind=ste codec.ste_codec=h265 \
    seed="$SEED" out_dir="$OUT_DIR" \
    train.finetune=true train.resume=false train.epochs=6 train.lr=3e-5 train.max_steps="$STE_MAX_STEPS"

echo "== 5. eval STAGE 2 (after STE calibration) on HELD-OUT analyzer + real x264/x265 =="
python evaluate.py --config "$CFG" \
    --ckpt "$CKPT_DIR/preprocessor.pth" --out "$EVAL_STAGE2" \
    data.index="$INDEX" seed="$SEED" out_dir="$OUT_DIR" eval.held_out_backbone="$HELD_OUT"

echo "== 6. STE marginal effect: prep-gain on the REAL codecs, Stage-1 vs Stage-2 =="
python - "$EVAL_STAGE1/results.json" "$EVAL_STAGE2/results.json" <<'PY'
import json, sys
for tag, p in zip(("stage1 (proxy only)", "stage2 (+STE)"), sys.argv[1:]):
    g = json.load(open(p)).get("bd_prep_gain", {})
    print(f"[{tag}]  (negative BD-Rate = savings = the claim)")
    for k, v in g.items():
        rate, acc = v.get("bd_rate_pct"), v.get("bd_accuracy")
        rs = f"{rate:+.2f}%" if rate is not None else "undefined"
        acs = f"{acc:+.4f}" if acc is not None else "undefined"
        print(f"   {k:26s} BD-Rate {rs:>10s}  BD-Acc {acs}")
PY

echo "== done. $EVAL_STAGE1 = pretrain, $EVAL_STAGE2 = +STE. bd_prep_gain vs h264/h265 = the claim =="
