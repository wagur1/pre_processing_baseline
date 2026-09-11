# pre_processing_baseline — Zhao et al. baseline dưới protocol đo công bằng

**Paper**: Zhao, Guo, Zhao et al., *A Preprocessing Framework for Video Machine
Vision under Compression* (Peking University + ByteDance) — PDF tại
`docs/zhao_paper.pdf`. Claims: BD-Rate **−11.7…−19.6%** action recognition
(max −17.6 Slow-only h264 / −16.2 h265), −12…−17.2% tracking.

## Vì sao phải test lại: các điều kiện của Zhao

| # | Zhao (paper) | Protocol chuẩn của dự án | Ảnh hưởng |
|---|---|---|---|
| 1 | **On-teacher eval** — "For each machine vision network, we individually trained a corresponding preprocessor": preprocessor train cho backbone nào thì **chính backbone đó chấm điểm** (6 preprocessor/6 backbone) | Held-out `r2plus1d_18` — analyzer chấm điểm KHÔNG BAO GIỜ thấy khi train | Lớn nhất: preprocessor được tối ưu trực tiếp cho giám khảo. Dựng đo của chính lineage này: cùng họ phương pháp co từ −12…−19 (on-teacher) về −2.4…−3.4 (held-out) |
| 2 | Không CI/bootstrap — point estimate duy nhất | 10k bootstrap clip-level CI + P(BD<0) | Không biết noise |
| 3 | Split không định nghĩa ("trained and tested on the Kinetics400") | Canonical n=1159, md5-hash split, fingerprint `30f083f8520a` | Không tái lập được; n không rõ |
| 4 | Headline = max theo backbone (−17.6 Slow-only; SlowFast chỉ −12.3) | Một cấu hình một số, CI | Cherry-pick |
| 5 | Không gap rule | Gap ≥ −0.05 mọi QP, cả 2 codec | Không thấy vùng QP bị phá |

**Phần KHÔNG unfair** (giữ nguyên để công bằng với Zhao): x264/x265 preset
medium, QP {30,35,40,45,50}, bpp metric, loss Eq.(1) `α(L_D+λL_R)+L_Acc`
với α=10, λ=0.001, MSE distortion, virtual codec — toàn bộ reproduced
trung thực trong codebase này (`configs/zhao_pure_*.yaml`, KHÔNG có
bit-restrainer kappa của lineage).

## Thiết kế thí nghiệm (2 arm, cùng 1 checkpoint)

| Arm | Analyzer chấm | Ý nghĩa |
|---|---|---|
| **A. On-teacher (Zhao-style)** | Chính backbone được train (slowonly / slowfast) | Tái hiện setup của họ — verify claim −12…−19 có thật không trên hạ tầng độc lập |
| **B. Held-out (fair)** | `r2plus1d_18` (không thấy khi train) + CI + gap rule | Số so sánh được với mọi model của dự án (E2 −5.89 / STE −3.56 / v9-b) |

Backbones: **slowonly_r50** (claim-max của paper) + **slowfast_r50** (min),
qua pytorchvideo, label-permutation về đúng thứ tự class torchvision.
Protocol: canonical n=1159, x264+x265 medium QP30–50.

## Cấu trúc

```
docs/zhao_paper.pdf             # paper gốc
configs/zhao_pure_slowonly.yaml # arm train slowonly (eval on-teacher)
configs/zhao_pure_slowfast.yaml # arm train slowfast
src/tasks/action_recognition.py # + slowonly/slowfast (pytorchvideo, permuted)
ops/...                         # Kaggle ops (slug bl-*)
```

Fair arm chạy bằng override: `eval.held_out_backbone=r2plus1d_18` trên cùng
checkpoint arm A.
