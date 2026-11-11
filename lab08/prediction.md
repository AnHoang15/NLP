# Bài 11 — Prediction Before Experiment


## Prediction 1 — Pretrained fine-tuning và from scratch

**Câu hỏi:** Một pretrained Transformer fine-tuned trên IMDb có khả năng đạt kết quả tốt hơn Transformer nhỏ được train from scratch trên cùng dữ liệu không?

**Prediction:**

Mô hình pretrained được fine-tune trên IMDb sẽ đạt kết quả tốt hơn mô hình Transformer nhỏ được train from scratch trên cùng dữ liệu.

**Reason:**

Mô hình pretrained đã học được nhiều đặc trưng ngôn ngữ từ dữ liệu lớn trước đó. Fine-tuning có thể điều chỉnh các đặc trưng này cho bài toán phân loại cảm xúc IMDb. Mô hình nhỏ train from scratch chỉ học từ dữ liệu của bài tập nên có thể cần nhiều dữ liệu và thời gian hơn để đạt chất lượng tương đương.

**Expected evidence:**

So sánh Accuracy và F1-score trên cùng tập test IMDb, với cùng cách chia dữ liệu. Dự đoán được ủng hộ nếu pretrained fine-tuned có các chỉ số cao hơn; thời gian huấn luyện và đường cong loss có thể cung cấp thêm thông tin về hiệu quả học.

## Prediction 2 — Frozen encoder và fine-tuning

**Câu hỏi:** Nếu freeze toàn bộ pretrained encoder và chỉ train classification head, kết quả sẽ thế nào so với fine-tuning toàn bộ?

**Prediction:**

- [ ] A. Tốt hơn fine-tuning toàn bộ
- [ ] B. Tương đương
- [x] C. Kém hơn
- [ ] D. Không thể dự đoán

**Reason:**

Khi freeze encoder, các biểu diễn ngôn ngữ pretrained không được cập nhật để thích nghi với đánh giá cảm xúc trong IMDb; chỉ classification head được học. Fine-tuning toàn bộ có thể điều chỉnh cả encoder theo dữ liệu và nhãn của bài toán, nên mình dự đoán sẽ đạt kết quả tốt hơn. Tuy vậy, freeze encoder có thể huấn luyện nhanh hơn và đôi khi cho kết quả gần tương đương nếu biểu diễn pretrained đã phù hợp.

**Expected evidence:**

Đối chiếu Accuracy và F1-score của hai cách trên cùng test split, dùng cùng dữ liệu và thiết lập huấn luyện. Dự đoán được ủng hộ nếu mô hình fine-tune toàn bộ đạt chỉ số cao hơn; đồng thời ghi nhận thời gian huấn luyện để thấy lợi ích tốc độ của việc freeze encoder.
