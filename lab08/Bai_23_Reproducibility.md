# Bài 23 — Reproducibility

Ghi nhận cấu hình của thí nghiệm có kiểm soát Bài 22 (so sánh độ dài chuỗi) để người khác có thể tái lập.

| Thông tin | Cấu hình đã dùng |
|---|---|
| Model | `bert-base-uncased`, fine-tuned bằng `AutoModelForSequenceClassification` |
| Tokenizer | `AutoTokenizer` từ `bert-base-uncased` |
| Dataset | `stanfordnlp/imdb` trên Hugging Face |
| Train split | IMDb train; lấy 1.000 review cân bằng, 500 mỗi nhãn |
| Validation split | Không sử dụng trong thí nghiệm này |
| Test split | IMDb test; lấy 500 review cân bằng, 250 mỗi nhãn |
| Max sequence length | 64, 128, 256 |
| Batch size | 16 |
| Learning rate | `2e-5` |
| Epochs | 1 cho mỗi độ dài chuỗi |
| Optimizer | AdamW |
| Random seed | 42; seed lấy mẫu test là 43 |
| Hardware | Google Colab GPU — NVIDIA T4 (CUDA) |
| Training time | 64: 32,4 giây; 128: 25,2 giây; 256: 50,7 giây (thời gian gồm train và evaluation) |

## Kết quả ghi nhận

| Max sequence length | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 64 | 0,7560 | 0,7336 | 0,8040 | 0,7672 |
| 128 | 0,7960 | 0,7342 | 0,9280 | 0,8198 |
| 256 | 0,8620 | 0,8131 | 0,9400 | 0,8720 |

**Lưu ý:** Thí nghiệm không có validation split và kết quả được đo trên một lần chia dữ liệu, một seed và một epoch. Thời gian chạy phụ thuộc phiên Colab và phần cứng tại thời điểm thực hiện.
