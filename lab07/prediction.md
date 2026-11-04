# Bài 8. Prediction trước experiment

## Prediction 1

**Câu hỏi:** Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?

**Prediction:** Có. Transformer có thể tính các vị trí của một lớp cùng lúc, trong khi RNN phải truyền trạng thái lần lượt qua sequence.

**Reason:** Self-attention tạo các tương tác giữa token bằng phép toán ma trận trên toàn sequence; RNN phụ thuộc trạng thái bước trước. Điều này cho Transformer khả năng song song hóa token tốt hơn trong quá trình huấn luyện.

**Confidence:** Cao.

## Prediction 2

**Câu hỏi:** Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

**Prediction:** Không, bản thân self-attention không biểu diễn thứ tự token nếu không được cung cấp positional information.

**Reason:** Attention tính tương tác dựa trên nội dung các token. Khi hoán vị đầu vào, các đầu ra token cũng hoán vị tương ứng; mô hình không có tín hiệu cho biết token nào đứng trước hay sau.

**Confidence:** Cao.

## Prediction 3

**Câu hỏi:** Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?

**Prediction:** Thông thường không, nếu biểu diễn câu được gộp bằng pooling bất biến theo thứ tự, như mean pooling.

**Reason:** Không có positional information, self-attention là permutation-equivariant: đổi thứ tự token chỉ đổi thứ tự các vector đầu ra. Phép pooling bất biến theo hoán vị sẽ tạo cùng biểu diễn câu cho hai đầu vào có cùng token.

**Confidence:** Cao.

## Prediction 4

**Câu hỏi:** Hai attention heads có nhất thiết học cùng một attention pattern không?

**Prediction:** Không nhất thiết; các head có thể học các pattern khác nhau, nhưng cũng có thể trùng lặp một phần.

**Reason:** Mỗi head có các projection riêng cho query, key và value nên có thể tập trung vào các quan hệ khác nhau giữa token. Kiến trúc không đảm bảo các head sẽ đa dạng hoặc hoàn toàn khác nhau.

**Confidence:** Khá cao.

## Prediction 5

**Câu hỏi:** Tăng số attention heads có chắc chắn làm accuracy tăng không?

**Prediction:** Không.

**Reason:** Accuracy còn phụ thuộc dữ liệu, cách tối ưu, regularization và cấu hình mô hình. Nếu `d_model` giữ cố định thì tăng số head làm chiều mỗi head `d_k = d_model / n_heads` nhỏ đi; thêm head không đảm bảo mô hình học được thông tin hữu ích hơn.

**Confidence:** Cao.

## Prediction 6

**Câu hỏi:** Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

**Prediction:** Không.

**Reason:** Transformer xử lý song song các vị trí nhưng self-attention có chi phí tăng theo bình phương độ dài sequence (`O(T²)`). RNN tính tuần tự nhưng chi phí theo sequence thường tăng tuyến tính (`O(T)`). Với sequence ngắn, overhead của Transformer có thể lớn; tốc độ thực tế còn tùy phần cứng và cách cài đặt.

**Confidence:** Cao.

## Tóm tắt dự đoán

Transformer cho phép song song hóa token tốt hơn RNN, nhưng cần positional information để nhận biết thứ tự và không đảm bảo accuracy cao hơn hay thời gian chạy thấp hơn trong mọi trường hợp. Các attention heads có thể học pattern khác nhau, song số head nhiều hơn không tự động cải thiện kết quả.
