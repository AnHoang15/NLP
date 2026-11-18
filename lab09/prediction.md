# LAB 09 — Prediction Before Experiment

Các nội dung dưới đây là dự đoán trước khi xem kết quả thực nghiệm.

## Prediction 1

Nếu dùng pretrained Transformer thay cho classifier đơn giản dựa trên word features, NER sẽ thay đổi như thế nào?

**Dự đoán:** Transformer sẽ có precision, recall và F1 cao hơn baseline, đặc biệt với những từ cần ngữ cảnh để xác định có phải tên thực thể hay không.

**Lý do:** BERT được tiền huấn luyện trên văn bản lớn nên biểu diễn từ có thông tin ngữ cảnh hai chiều. Baseline dựa trên word features thường phụ thuộc nhiều hơn vào đặc trưng bề mặt và những mẫu đã thấy trong tập huấn luyện. Dự đoán này cần được kiểm tra bằng cùng test set và cùng cách tính chỉ số.

## Prediction 2

Entity dài như “United Nations Security Council” sẽ dễ hay khó nhận diện hơn entity một token như “Microsoft”?

**Dự đoán:** Entity dài có khả năng khó nhận diện chính xác toàn bộ span hơn entity một token.

**Lý do:** Muốn được tính đúng ở mức entity, mô hình phải dự đoán đúng loại và cả hai ranh giới của span. Entity dài có nhiều vị trí có thể gán nhãn sai; chỉ cần một token ở đầu hoặc cuối sai thì toàn bộ entity có thể bị tính sai. Ngữ cảnh quanh cụm dài đôi khi giúp nhận biết, nên đây là xu hướng dự đoán chứ không phải kết quả chắc chắn.

## Prediction 3

Entity hiếm xuất hiện trong training data có khả năng gây lỗi nhiều hơn không?

**Dự đoán:** Có. Entity hiếm có khả năng bị bỏ sót hoặc gán sai loại nhiều hơn entity phổ biến.

**Lý do:** Mô hình có ít ví dụ để học cách nhận diện dạng từ và ngữ cảnh của entity hiếm. Pretrained Transformer có thể giảm phần nào vấn đề này nhờ kiến thức ngôn ngữ đã học trước, nhưng mức cải thiện phụ thuộc vào việc entity và ngữ cảnh tương tự có xuất hiện trong dữ liệu tiền huấn luyện hay không.
