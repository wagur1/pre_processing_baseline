# RESULTS — UP-VCM (điền khi có kết quả)

> Template tạo TRƯỚC khi chạy. Comparators đóng từ lineage (cùng instrument:
> protocol chuẩn n=1159, held-out r2plus1d_18, QP30–50, preset medium).

## Trạng thái chạy

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| Tests + smoke local (89/89) | ✅ | full pipeline trên dữ liệu giả |
| Probe hạ tầng Kaggle | ✅ PASS | fingerprint 30f083f8520a, codec checks |
| Train 16ep UP-VCM v1 | ✅ COMPLETE | best epoch 14/16, ~10h, `wagur124705/u7-train-upvcm` |
| Đọc gates từ v1 checkpoint | ✅ | dec=−0.6242 (M1 MỞ), edit=0.0000 (**M2 CHẾT — dead-saddle bug**), stab=−0.0782 (M3 mở nhẹ) |
| Bug fix + v2 retrain | ✅ commit `da97514889b1` | noise-init out conv; kernel `vtk269/u7-train-upvcm-v2` đang chạy |
| Eval sharded v1 (as-trained: M1+M3) | ⏳ | shard 0+1 RUNNING, shard 2 chờ slot GPU |
| Merge + bootstrap CI + gap rule | ☐ | `ops/merge_eval.py` |

**Phát hiện v1 (as-trained):** M2 (ROI editor) không bao giờ mở do **dead-saddle
init** — out conv zero-init × strength gate zero-init ⇒ gradient của cả hai ≡ 0
từ init (xác nhận bằng unit repro: `test_m2_gate_gradients_alive_at_init`).
M1/M3 mở được vì content path của chúng khác 0 lúc init. Kết quả v1 đo cơ chế
M1 + M3; v2 (đã fix) cho phép cả M2 học. Bài học phương pháp luận: identity-at-init
kép KHÔNG an toàn — đúng MỘT tầng zero (gate), tầng nội dung phải sống.

## Comparators (từ v6, cùng protocol)

| Biến thể | BD h264 [CI95] | BD h265 [CI95] | gap rule |
|---|---|---|---|
| (anchor) không pre-processing | 0% (định nghĩa) | 0% | — |
| kappa=10 @16ep (Zhao editor, lineage best) | −3.42% [−5.88, −0.88] | −2.63% [−4.49, −0.79] | PASS |
| kappa=10 @16ep rep1 | −2.52% [−4.92, +0.16] | −2.33% [−4.14, −0.57] | PASS |

Đọc với dung sai ±1pp (re-run noise tầng eval/codec).

## Kết quả UP-VCM

### Gates

```
[G1 non-identity] dec=____ edit=____ stab=____ (>0.01)
[G2 no-blow-up]   RMS ____ (<0.14)
[G3 W healthy]    mean ____ std ____
[G4 deploy purity] diff ____
```

### BD-Rate v1 as-trained (FULL n=1159, 10k bootstrap) — M1+M3, M2 chết

| Codec | BD-Rate | CI95 (bootstrap) | P(BD<0) | gap rule |
|---|---|---|---|---|
| h264 | **−2.46%** | [−4.47, −0.40] | 0.991 | PASS |
| h265 | −0.78% | [−2.22, +0.71] | 0.841 | PASS |

### BD-Rate v2 (đủ M1+M2+M3, FULL n=1159, 10k bootstrap)

| Codec | BD-Rate | CI95 (bootstrap) | P(BD<0) | gap rule |
|---|---|---|---|---|
| h264 | −0.35% | [−2.37, +1.82] | 0.620 | PASS |
| h265 | −1.47% | [−2.92, −0.00] | 0.975 | PASS |

**Bất ngờ v2:** M2 mở (−0.038) nhưng h264 giảm 1.83pp so với v1 (−2.46 → −0.63)
trong khi h265 khá hơn (−0.78 → −1.48). M2 học edit có lợi cho proxy nhưng phá
transfer sang x264 (block 4×4/8×8 + in-loop deblock nhạy với pattern M2 thêm);
x265 (block lớn) chấp nhận tốt hơn. Kết luận: gates mở ≠ gates có ích trên codec
thật; v1 (M1+M3) là cấu hình pre-only tốt nhất của UP-VCM đến giờ — và POST của
v8 (xem RESULTS_sandwich) là nơi bù đắp đúng vai trò M2.

### Per-QP

| QP | anchor h264 | prep h264 | Δbpp h264 | anchor h265 | prep h265 | Δbpp h265 |
|---|---|---|---|---|---|---|
| 30 | | | | | | |
| 35 | | | | | | |
| 40 | | | | | | |
| 45 | | | | | | |
| 50 | | | | | | |

### Module utilization (từ checkpoint)

| Gate | Giá trị | Đọc |
|---|---|---|
| dec_strength (M1) | | >0: nền bị decimate |
| edit_strength (M2) | | >0: ROI edit hoạt động |
| stab_strength (M3) | | >0: ổn định thời gian hoạt động |

## Đọc kết quả (viết sau khi có số)

- So anchor (no-prep) — claim chính của mô hình mới.
- So kappa=10 (mô hình cũ) cùng instrument, dung sai ±1pp: UP-VCM có qua mặt
  lineage best không, và trên codec nào.
- Cơ chế: Δbpp@QP30 (M1/M3 phải ép xuống so với +14% của additive), gap per
  QP (M2 phải giữ/leo ở QP nặng).
