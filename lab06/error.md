# Bài 27 — Error analysis

## Kết quả mô hình

Mô hình bài 26 được đánh giá trên 5.000 review IMDb: **accuracy = 0.8152**, **F1 = 0.8285**. Notebook đã chọn một ví dụ đúng và một ví dụ sai để bắt đầu phân tích. Cần bổ sung thêm hai ví dụ đúng và hai ví dụ sai từ tập test để đủ yêu cầu **3 correct + 3 incorrect**.

## Các ví dụ đã lấy từ output bài 26

| Loại | Test index | Nhãn thật | Dự đoán | Xác suất lớp dự đoán | Review (trích đoạn) |
|---|---:|---|---|---:|---|
| Đúng | 1 | positive | positive | 0.9998 | “this film requires a lot of patience … focuses on mood and character development …” |
| Sai | 0 | negative | positive | 0.9129 | “please give this one a miss … rendered terrible performances … the show is flat flat flat …” |

Attention weights trung bình theo token được in trong output bài 26. Một số token có trọng số cao nhất theo output:

- Ví dụ đúng: `patience` 0.0063, `development` 0.0063, `same` 0.0063, `frances` 0.0063, `interesting` 0.0063.
- Ví dụ sai: `terrible` 0.0150, `rest` 0.0148, `wasn't` 0.0148, `give` 0.0148 (hai vị trí).

## Phân tích 3 dự đoán đúng

| # | Test index / review | Nhãn thật → dự đoán | Nhóm hiện tượng (nếu có) | Attention tập trung vào token nào? | Phân tích của sinh viên |
|---:|---|---|---|---|---|
| 1 | Index 1; review trích ở trên | positive → positive |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

## Phân tích 3 dự đoán sai

| # | Test index / review | Nhãn thật → dự đoán | Nhóm lỗi | Attention tập trung vào token nào? | Phân tích của sinh viên |
|---:|---|---|---|---|---|
| 1 | Index 0; review trích ở trên | negative → positive |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

### Nhóm lỗi gợi ý trong đề

`negation` · `long-range dependency` · `ambiguous wording` · `sarcasm` · `rare word` · `domain shift` · `truncation` · `noise`

### Câu hỏi cần tự trả lời cho từng ví dụ

1. Review chứa tín hiệu sentiment nào? Có phủ định, mệnh đề tương phản, từ hiếm, cách diễn đạt mơ hồ, châm biếm, khác miền dữ liệu, cắt cụt hoặc nhiễu không?
2. Attention có tập trung vào token mà bạn cho là quan trọng không? Ghi lại token và weight quan sát được.
3. Nếu attention không tập trung vào token đó, đây có thể là một giả thuyết giải thích prediction sai không? Đối chiếu giả thuyết với review, prediction và hành vi mô hình trước khi kết luận.

> Attention pattern là thông tin về phân phối trọng số của mô hình. Không kết luận rằng “attention nhìn sai từ nên model sai”; hãy trình bày đây là giả thuyết và đối chiếu với hành vi mô hình.
