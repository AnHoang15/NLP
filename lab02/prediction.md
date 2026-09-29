# 12. Prediction trước experiment

## Prediction 1 — Vocabulary khi chuyển unigram → bigram → trigram
Prediction: Không tăng.
Reason: Vocabulary là tập từ duy nhất trong corpus, không phụ thuộc n.
Confidence: Cao

## Prediction 2 — Số lượng n-gram thay đổi như thế nào?
Prediction: Số n-gram duy nhất tăng dần: unigram < bigram < trigram.
Reason: Context dài hơn → tổ hợp đa dạng hơn, ít trùng lặp hơn.
Confidence: Trung bình

## Prediction 3 — Model nào gặp zero probability nhiều hơn?
Prediction: Trigram nhiều nhất, rồi đến bigram, unigram ít nhất.
Reason: n càng lớn, context càng thưa (sparsity), càng dễ chưa từng gặp trong train.
Confidence: Cao

## Prediction 4 — Model nào có perplexity thấp hơn trên training set?
Prediction: Trigram thấp nhất.
Reason: Context dài giúp model khớp sát với chính dữ liệu train đã thấy.
Confidence: Cao

## Prediction 5 — Corpus rất nhỏ, trigram có chắc chắn tốt hơn bigram không?
Prediction: Không.
Reason: Corpus nhỏ → nhiều trigram trong test chưa thấy → dễ gặp zero-prob, perplexity tăng.
Confidence: Cao