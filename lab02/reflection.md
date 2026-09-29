# 24. Reflection

## Câu 1: Nếu tăng n, mô hình nhận thêm thông tin gì?
Mô hình nhìn thấy thêm ngữ cảnh: từ n-1 từ trước thay vì ít hơn. Ví dụ unigram chỉ biết "eats" phổ biến đến đâu, bigram biết "eats" hay đứng sau "cat", trigram biết "eats" hay đứng sau "the cat". Ngữ cảnh dài hơn giúp dự đoán từ tiếp theo chính xác hơn, thể hiện ở training perplexity giảm (1,845 → 152 → 14.5 với MLE).

## Câu 2: Tại sao tăng n lại làm sparsity tăng?
Số n-gram có thể có tăng theo cấp số mũ (V^n) trong khi lượng dữ liệu cố định, nên phần lớn n-gram không xuất hiện hoặc chỉ xuất hiện một lần. Trong thí nghiệm, tỉ lệ token test có n-gram chưa từng thấy là 2.2% (unigram), 22.7% (bigram), 57.2% (trigram).

## Câu 3: Tại sao smoothing cần thiết?
Nếu một n-gram chưa xuất hiện trong training thì MLE cho P = 0, làm xác suất cả câu bằng 0 và perplexity bằng inf (valid/test của mọi model MLE đều là inf). Nhưng chưa quan sát thấy không có nghĩa là không thể xảy ra. Smoothing dành một phần xác suất cho các sự kiện chưa thấy. Add-one quá mạnh khi V lớn (bigram test PPL 8,342), còn add-k với k nhỏ tốt hơn nhiều (1,284).

## Câu 4: Perplexity đo điều gì?
Perplexity là nghịch đảo của xác suất trung bình (trung bình nhân) mà model gán cho mỗi token, có thể hiểu là số lựa chọn tương đương mà model đang phân vân ở mỗi bước. Perplexity thấp nghĩa là model gán xác suất cao hơn cho dữ liệu đánh giá. So sánh chỉ có ý nghĩa khi cùng dữ liệu, cùng tokenization và cùng vocabulary.

## Câu 5: Perplexity thấp hơn có luôn tạo văn bản tốt hơn đối với con người không?
Không. Perplexity chỉ đo xác suất mà model gán cho dữ liệu, không đo tính mạch lạc, ngữ pháp, ý nghĩa hay tính đúng sự thật. Nó cũng dễ bị tác động bởi vocabulary và tokenization, và một model học thuộc train có perplexity rất thấp nhưng vô dụng với câu mới (trigram MLE: train 14.5, valid inf). Vì vậy còn cần đánh giá của con người và đánh giá theo tác vụ.

## Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
N-gram chỉ đếm chuỗi từ nên không hiểu nghĩa: "cat" và "dog" là hai ký hiệu rời rạc, không chia sẻ thống kê với nhau. Nó không nắm được cấu trúc cú pháp hay phụ thuộc xa, không tổng quát hóa được sang cụm từ chưa từng thấy (sparsity), và không có kiến thức về thế giới. Con người hiểu nghĩa và cấu trúc nên có thể tổng quát hóa từ rất ít ví dụ.

## Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
Không. Trigram chỉ dùng 2 từ cuối (giả định Markov bậc 2), 98 từ còn lại bị bỏ qua hoàn toàn. Nên với câu như "The keys that the man dropped yesterday ... are/is", trigram không thể dùng "keys" để chọn động từ phù hợp. Đây là lý do cần các model có thể dùng toàn bộ context như RNN, attention và Transformer.