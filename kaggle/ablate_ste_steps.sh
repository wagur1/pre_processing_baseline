#!/usr/bin/env bash
# Controlled STE-step ablation: train Stage 1 once, then fine-tune identical
# copies of that checkpoint for different numbers of real-codec STE steps.
set -euo pipefail

trap 'echo "ERROR: kaggle/ablate_ste_steps.sh failed at line $LINENO: $BASH_COMMAND" >&2' ERR

CFG=${CFG:-configs/universal_action_recognition.yaml}
INDEX=${INDEX:-data/index/kinetics_3gb.json}
KINETICS_ROOT=${KINETICS_ROOT:-/kaggle/input/kinetics-train-5per/train}
HELD_OUT=${HELD_OUT:-r2plus1d_18}
SEED=${SEED:-0}
STE_CODEC=${STE_CODEC:-h265}
STE_STEPS=${STE_STEPS:-"400 1000"}
BASE_OUT=${BASE_OUT:-outputs/ste_steps_ablation/seed_$SEED}
STAGE1_OUT=$BASE_OUT/stage1_base
STAGE1_CKPT=$STAGE1_OUT/checkpoints/preprocessor.pth

[[ "$BASE_OUT" == outputs/* ]] || {
  echo "ERROR: BASE_OUT must be under outputs/" >&2
  exit 1
}

echo "== 0. dependencies and codec check =="
if [[ "${INSTALL_DEPS:-1}" == "1" ]]; then
  python -m pip install -q compressai av
fi
ENCODERS=$(ffmpeg -hide_banner -encoders 2>/dev/null || true)
grep -q libx264 <<<"$ENCODERS" || { echo "ERROR: ffmpeg libx264 missing" >&2; exit 1; }
grep -q libx265 <<<"$ENCODERS" || { echo "ERROR: ffmpeg libx265 missing" >&2; exit 1; }

echo "== 1. locate data and build the shared index =="
if [[ ! -d "$KINETICS_ROOT" ]]; then
  KINETICS_ROOT=$(find /kaggle/input -maxdepth 5 -type d -name train -print -quit)
fi
if [[ ! -d "$KINETICS_ROOT" ]]; then
  sample=$(find /kaggle/input -maxdepth 8 -type f \
    \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' \
       -o -iname '*.webm' -o -iname '*.m4v' \) -print -quit)
  if [[ -n "$sample" ]]; then
    candidate=$(dirname "$(dirname "$sample")")
    [[ -d "$candidate" ]] && KINETICS_ROOT="$candidate"
  fi
fi
[[ -n "$KINETICS_ROOT" && -d "$KINETICS_ROOT" ]] || {
  echo "ERROR: Kinetics train directory not found" >&2
  exit 1
}
echo "[data] Kinetics root = $KINETICS_ROOT"

if [[ ! -f "$INDEX" ]] || ! python -c \
  'import json,sys; sys.exit("test" not in json.load(open(sys.argv[1])))' "$INDEX"; then
  python -m src.data.prepare_3gb --root "$KINETICS_ROOT" --out "$INDEX" \
    --cap-gb 3.0 --val-frac 0.1 --test-frac 0.1 --backbone r3d_18
fi

if [[ "${CLEAN_RUN:-1}" == "1" ]]; then
  rm -rf -- "$BASE_OUT"
fi
mkdir -p "$STAGE1_OUT"

echo "== 2. train Stage 1 exactly once =="
python train.py --config "$CFG" \
  data.index="$INDEX" seed="$SEED" out_dir="$STAGE1_OUT" train.epochs=5

[[ -f "$STAGE1_CKPT" ]] || {
  echo "ERROR: Stage-1 checkpoint was not produced: $STAGE1_CKPT" >&2
  exit 1
}
STAGE1_SHA=$(sha256sum "$STAGE1_CKPT" | awk '{print $1}')
echo "[control] Stage-1 SHA256 = $STAGE1_SHA"

echo "== 3. evaluate and health-check the shared Stage 1 =="
python evaluate.py --config "$CFG" \
  --ckpt "$STAGE1_CKPT" --out "$STAGE1_OUT/eval" \
  data.index="$INDEX" seed="$SEED" out_dir="$STAGE1_OUT" \
  eval.held_out_backbone="$HELD_OUT"

python - "$STAGE1_OUT/eval/results.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
prep = [curve for name, curve in result.get("curves", {}).items()
        if name.startswith("prep+")]
best = max((max(curve.get("accuracy", [0.0])) for curve in prep), default=0.0)
print(f"[health] Stage-1 best preprocessed accuracy={best:.4f}")
if best < 0.05:
    raise SystemExit("ERROR: Stage-1 health check failed; refusing to run STE ablation")
PY

echo "== 4. fine-tune identical checkpoint copies =="
for steps in $STE_STEPS; do
  [[ "$steps" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: invalid STE step count: $steps" >&2
    exit 1
  }

  RUN_OUT=$BASE_OUT/steps_$steps
  rm -rf -- "$RUN_OUT"
  mkdir -p "$RUN_OUT/checkpoints"
  cp "$STAGE1_CKPT" "$RUN_OUT/checkpoints/preprocessor.pth"
  cp "$STAGE1_CKPT" "$RUN_OUT/checkpoints/stage1_source.pth"

  RUN_SHA=$(sha256sum "$RUN_OUT/checkpoints/preprocessor.pth" | awk '{print $1}')
  [[ "$RUN_SHA" == "$STAGE1_SHA" ]] || {
    echo "ERROR: checkpoint copy mismatch for steps=$steps" >&2
    exit 1
  }
  echo "[control] steps=$steps starts from SHA256=$RUN_SHA"

  python train.py --config "$CFG" \
    data.index="$INDEX" codec.kind=ste codec.ste_codec="$STE_CODEC" \
    seed="$SEED" out_dir="$RUN_OUT" \
    train.finetune=true train.resume=false train.epochs=6 train.lr=3e-5 \
    train.max_steps="$steps"

  python evaluate.py --config "$CFG" \
    --ckpt "$RUN_OUT/checkpoints/preprocessor.pth" --out "$RUN_OUT/eval" \
    data.index="$INDEX" seed="$SEED" out_dir="$RUN_OUT" \
    eval.held_out_backbone="$HELD_OUT"
done

echo "== 5. controlled comparison =="
python - "$BASE_OUT" $STE_STEPS <<'PY'
import json
import pathlib
import sys

base = pathlib.Path(sys.argv[1])
steps_list = [int(value) for value in sys.argv[2:]]
pairs = ("prep+h264 vs h264", "prep+h265 vs h265")
summary = {
    "seed": int(base.name.rsplit("_", 1)[-1]),
    "stage1_checkpoint_sha256": None,
    "runs": {},
}

import hashlib
checkpoint = base / "stage1_base" / "checkpoints" / "preprocessor.pth"
summary["stage1_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

print("\n=== SAME Stage-1 checkpoint: STE step ablation ===")
print(f"Stage-1 SHA256: {summary['stage1_checkpoint_sha256']}")
for steps in steps_list:
    path = base / f"steps_{steps}" / "eval" / "results.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    gain = result.get("bd_prep_gain") or {}
    summary["runs"][str(steps)] = {
        "results": str(path),
        "n_eval": result.get("n_eval"),
        "bd_prep_gain": {pair: gain.get(pair) for pair in pairs},
    }
    print(f"\n[{steps} STE steps] n_eval={result.get('n_eval')}")
    for pair in pairs:
        value = gain.get(pair) or {}
        rate = value.get("bd_rate_pct")
        acc = value.get("bd_accuracy")
        rate_s = f"{rate:+.3f}%" if rate is not None else "undefined"
        acc_s = f"{acc:+.4f}" if acc is not None else "undefined"
        print(f"  {pair:24s} BD-Rate {rate_s:>10s} | BD-Acc {acc_s}")

if len(steps_list) == 2:
    first, second = map(str, steps_list)
    print(f"\n[delta: {second} minus {first}; negative BD-Rate delta is better]")
    for pair in pairs:
        a = summary["runs"][first]["bd_prep_gain"].get(pair) or {}
        b = summary["runs"][second]["bd_prep_gain"].get(pair) or {}
        ar, br = a.get("bd_rate_pct"), b.get("bd_rate_pct")
        aa, ba = a.get("bd_accuracy"), b.get("bd_accuracy")
        dr = None if ar is None or br is None else br - ar
        da = None if aa is None or ba is None else ba - aa
        dr_s = f"{dr:+.3f} pp" if dr is not None else "undefined"
        da_s = f"{da:+.4f}" if da is not None else "undefined"
        print(f"  {pair:24s} delta BD-Rate {dr_s:>10s} | delta BD-Acc {da_s}")

out = base / "comparison.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\n[report] wrote {out}")
PY

echo "== done: $BASE_OUT =="
