# Bài 24–25 — Error Analysis và So sánh với LAB 04

## Bài 24 — Error analysis

Điền ít nhất **3 dự đoán đúng** và **3 dự đoán sai** từ kết quả model.

| Review (rút gọn) | Nhãn thật | Dự đoán | Phân tích |
|---|---|---|---|
| … | … | … | … |
| … | … | … | … |
| … | … | … | … |
| … | … | … | … |
| … | … | … | … |
| … | … | … | … |

Khi phân tích lỗi, xem xét: phủ định, dependency xa, sarcasm, sentiment mơ hồ, từ hiếm/chuyên ngành, review dài, truncation hoặc nhiễu. Với từng lỗi, ghi rõ có bằng chứng cho thấy lỗi liên quan đến sequence modeling hay không; đừng mặc định rằng model sai là do “không hiểu câu”.

## Bài 25 — So sánh với LAB 04

| Thuộc tính | TF-IDF + LR | Average Word2Vec + LR | RNN | LSTM | GRU |
|---|---|---|---|---|---|
| Dạng biểu diễn | Thưa (sparse) | Dense, trung bình vector từ | Dense, chuỗi embedding | Dense, chuỗi embedding | Dense, chuỗi embedding |
| Dùng thứ tự từ | Hạn chế | Không | Có | Có | Có |
| Trạng thái phụ thuộc ngữ cảnh | Không | Không | Có | Có | Có |
| Dependency dài | Hạn chế | Hạn chế | Hạn chế | Cải thiện | Cải thiện |
| Chi phí tính toán | Thấp | Thấp | Cao hơn | Cao hơn | Cao hơn |

**Nhận xét:** RNN/LSTM/GRU xử lý thứ tự từ và cập nhật trạng thái theo ngữ cảnh, điều Average Word2Vec không thể hiện trực tiếp. Tuy nhiên, dependency xa vẫn khó; LSTM/GRU cải thiện khả năng giữ thông tin nhưng không đảm bảo luôn dự đoán tốt hơn, và các recurrent model tốn tính toán hơn.

Dựa trên kết quả thực nghiệm của bạn, nêu model nào hoạt động tốt hơn và dùng metrics/thời gian train làm bằng chứng. Đừng chỉ kết luận dựa trên Accuracy.
