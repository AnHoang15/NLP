# Bài 12 — Prediction trước experiment

## 1. Dot product và attention weight
Nếu dot product của một key lớn hơn dot product của các key còn lại, điều gì xảy ra với attention weight của key đó?

- **Prediction:** Attention weight của key đó sẽ lớn hơn các key còn lại.
- **Reason:** Softmax chuyển score lớn hơn thành xác suất lớn hơn, nên key có dot product lớn sẽ nhận trọng số attention cao hơn.

## 2. Các score bằng nhau
Nếu tất cả score bằng nhau, `s = [1, 1, 1, 1]`, attention weights sẽ là bao nhiêu?

- **Prediction:** Attention weights bằng nhau: `[0.25, 0.25, 0.25, 0.25]`.
- **Reason:** Softmax cho các score giống nhau tạo ra phân phối đều; tổng bốn trọng số bằng 1.

## 3. Một score lớn hơn hẳn
Nếu một score lớn hơn nhiều so với các score còn lại, `s = [1, 1, 10]`, attention sẽ tập trung như thế nào?

- **Prediction:** Attention sẽ tập trung gần như toàn bộ vào score thứ ba; softmax xấp xỉ `[0.00012, 0.00012, 0.99976]`.
- **Reason:** Chênh lệch lớn giữa score thứ ba và hai score còn lại khiến hàm softmax gán cho nó trọng số áp đảo.

## 4. Độ dài sequence và chi phí tính toán
Nếu độ dài sequence tăng từ 10 token lên 1000 token, chi phí tính toán attention thay đổi thế nào?

- **Prediction:** Với self-attention đầy đủ, chi phí tính toán phần attention tăng khoảng 10.000 lần.
- **Reason:** Attention tính tương tác giữa từng cặp token nên chi phí tăng theo bình phương độ dài sequence. Tăng độ dài 100 lần làm số cặp tăng `100² = 10.000` lần.

## 5. Attention và mức độ quan trọng theo con người
Attention có luôn chọn token mà con người cho là quan trọng nhất không?

- **Prediction:** Không. Attention không phải lúc nào cũng tập trung vào token mà con người đánh giá là quan trọng nhất.
- **Reason:** Trọng số attention được học để phục vụ dự đoán của mô hình; chúng thể hiện mức độ mô hình sử dụng thông tin trong phép tính, nhưng không nhất thiết trùng với cách con người diễn giải độ quan trọng.
