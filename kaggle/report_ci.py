#!/usr/bin/env python
"""Aggregate BD-Rate across seeds into a mean +/- 95% CI.

    python kaggle/report_ci.py outputs/seed_*/eval/results.json

Reads each run's ``results.json`` (written by ``src.engine._finalize``) and, for
every same-codec ``bd_prep_gain`` pair present, reports mean, std and a 95%
confidence interval of BD-Rate (%) and BD-accuracy across the seeds. Subset
BD-Rate is noisy (+/-3-4% per seed); this is how upgrade-3 turns a single run into
a defensible number. Pure stdlib -- no numpy/scipy needed.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from collections import defaultdict

from scipy.stats import t


def _mean_ci(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    std = math.sqrt(var)
    half = float(t.ppf(0.975, n - 1)) * std / math.sqrt(n)
    return mean, std, half


def main(argv):
    paths = []
    for pat in argv:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        print("no results.json files matched; pass e.g. outputs/seed_*/eval/results.json")
        return 1

    rate = defaultdict(list)   # pair label -> [bd_rate_pct, ...]
    acc = defaultdict(list)    # pair label -> [bd_accuracy, ...]
    for p in paths:
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        for label, v in (res.get("bd_prep_gain") or {}).items():
            if v.get("bd_rate_pct") is not None:
                rate[label].append(float(v["bd_rate_pct"]))
            if v.get("bd_accuracy") is not None:
                acc[label].append(float(v["bd_accuracy"]))

    print(f"\n=== BD-Rate across {len(paths)} run(s) (negative = savings) ===")
    if not rate:
        print("  no bd_prep_gain pairs found (did eval run with ffmpeg for x264/x265?)")
    for label in sorted(rate):
        m, s, h = _mean_ci(rate[label])
        am, _, ah = _mean_ci(acc.get(label, [0.0]))
        n = len(rate[label])
        print(f"  {label:28s}: BD-Rate {m:+.2f}% +/- {h:.2f} (std {s:.2f}, n={n})  |  "
              f"BD-Acc {am:+.4f} +/- {ah:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
