# Bài 22 — Evaluation

Quy ước lớp **positive** là review tích cực.

| Metric | Công thức |
|---|---|
| Accuracy | `(TP + TN) / (TP + TN + FP + FN)` |
| Precision | `TP / (TP + FP)` |
| Recall | `TP / (TP + FN)` |
| F1 | `2 × Precision × Recall / (Precision + Recall)` |

- **TP:** positive dự đoán đúng; **TN:** negative dự đoán đúng.
- **FP:** negative bị dự đoán thành positive; **FN:** positive bị dự đoán thành negative.

Báo cáo kết quả cùng với: **số tham số, thời gian train, số epoch, sequence length và vocabulary size**.
