# Bài 28. Prediction verification

Đối chiếu sáu dự đoán ban đầu với kết quả các thí nghiệm đã lưu trong `result.pdf`. Phần dự đoán ban đầu trong `prediction.md` được giữ nguyên.

| Prediction | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Transformer có thể xử lý các token trong sequence song song hơn RNN. | Về kiến trúc, Transformer tính các vị trí của một lớp cùng lúc. Trong lần chạy Bài 21 trên CPU với sequence dài 200, Transformer mất 946,25 giây, còn LSTM mất 252,99 giây. | Đúng | Dự đoán nói về khả năng song song hóa token nên đúng. Thời gian chạy thực tế không nhất thiết nhanh hơn: ở cấu hình Bài 21, Transformer chạy lâu hơn LSTM. |
| 2 | Không có positional information, self-attention không tự biểu diễn thứ tự token. | Trong thử nghiệm Bài 11, khoảng cách L2 giữa biểu diễn gộp của “I love NLP” và “NLP love I” là 0,0000 khi không có positional encoding; khi thêm sinusoidal positional encoding, khoảng cách là 0,0095. | Đúng | Kết quả phù hợp với dự đoán: self-attention không có tín hiệu vị trí nên biểu diễn không phân biệt được thứ tự trong thử nghiệm này. |
| 3 | Hai câu có cùng token nhưng khác thứ tự thường không phân biệt được nếu không có positional information và dùng mean pooling. | Thử nghiệm Bài 11 cho khoảng cách L2 bằng 0,0000 khi không có positional encoding và 0,0095 khi có positional encoding. | Đúng | Với cách gộp mean pooling của thử nghiệm, đổi thứ tự chỉ hoán vị các vector đầu ra; phép gộp tạo cùng biểu diễn câu nếu không thêm thông tin vị trí. |
| 4 | Hai attention heads không nhất thiết học cùng pattern; chúng có thể khác nhau nhưng trùng lặp một phần. | Ở Bài 20, cosine similarity giữa các cặp head nằm trong khoảng 0,228–0,777; entropy attention trung bình của bốn head lần lượt là 1,598; 2,495; 1,891; 2,737. | Đúng | Các head thể hiện mức độ tập trung và pattern khác nhau. Một số head vẫn có độ tương đồng đáng kể, chẳng hạn H2–H3 có cosine similarity 0,777, nên chúng không hoàn toàn độc lập. |
| 5 | Tăng số attention heads không đảm bảo accuracy tăng. | Ở Bài 19, accuracy với 1, 2, 4, 8 heads lần lượt là 0,7878; 0,7868; 0,7945; 0,7850. Cấu hình 4 heads cao nhất trong lần chạy này; 8 heads thấp hơn. | Đúng | Số head tăng không làm accuracy tăng đều. Kết quả này hỗ trợ dự đoán, nhưng chỉ phản ánh dữ liệu và cấu hình của thí nghiệm. |
| 6 | Transformer không luôn nhanh hơn RNN ở mọi sequence length. | Bài 21 ghi nhận Transformer mất 946,25 giây và LSTM mất 252,99 giây trên CPU. Trong Bài 22, thời gian train Transformer tăng từ 111,17 giây ở độ dài 32 lên 1.130,46 giây ở độ dài 256. | Đúng | Kết quả cho thấy tốc độ phụ thuộc vào độ dài sequence, phần cứng và cách cài đặt; khả năng song song hóa không đảm bảo thời gian chạy thấp hơn trong mọi trường hợp. Các lần chạy này không đại diện cho mọi phần cứng hoặc cấu hình. |

## Kết luận

Cả sáu dự đoán ban đầu đều phù hợp với kết quả và đặc tính được quan sát trong các thí nghiệm. Riêng dự đoán 1 nói về khả năng tính song song; nó không có nghĩa Transformer luôn hoàn thành nhanh hơn RNN. Các số liệu về thời gian và accuracy chỉ áp dụng cho cấu hình đã chạy.
