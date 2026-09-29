# 15. Log probability

Câu hỏi: Tại sao log probability giúp tránh vấn đề numerical underflow?

## Vấn đề

Xác suất của một câu là tích của nhiều xác suất có điều kiện, mỗi xác suất nằm trong khoảng (0, 1):

P(S) = P(w1) · P(w2|w1) · ... · P(wT|context)

Mỗi thừa số thường rất nhỏ (vd. 0.01 - 0.001). Câu càng dài thì tích càng nhỏ theo cấp số mũ. Kiểu số thực 64-bit (float64) chỉ biểu diễn được số dương nhỏ nhất cỡ 10^-308. Khi tích nhỏ hơn ngưỡng này, máy tính làm tròn về 0.0. Đó là numerical underflow: kết quả bị sai thành 0 dù xác suất thật vẫn dương.


## Cách log giải quyết

Vì log biến tích thành tổng:

log P(S) = log P(w1) + log P(w2|w1) + ... + log P(wT|context)

Mỗi số hạng log P chỉ cỡ -2 đến -20, nên tổng của cả nghìn số hạng vẫn là một số bình thường . Cộng các số vừa phải thay vì nhân các số cực nhỏ, nên không chạm ngưỡng làm tròn về 0.


---

# 16. Experiment 2 — MLE vs Laplace smoothing

Thiết lập: corpus C4 30K documents, chia theo document train / valid / test = 8 : 1 : 1 (seed 42). Vocabulary theo train: V = 214,567.

| Model | Train PPL | Valid PPL | Test PPL | Test zero-prob % |
|---|---|---|---|---|
| Bigram MLE | 152.1 | inf | inf | 22.66 |
| Bigram Laplace | 6,397.7 | 8,296.7 | 8,341.9 | 0 |
| Trigram MLE | 14.5 | inf | inf | 57.20 |
| Trigram Laplace | 35,179.3 | 57,497.4 | 57,973.8 | 0 |

Câu hỏi: Tại sao kết quả lại như vậy?

**1. MLE: train rất thấp, valid/test bằng inf.**
MLE chỉ dựa vào count trong train, nên trên chính train nó gán xác suất cao (trigram còn "học thuộc" nhiều hơn bigram: 14.5 so với 152.1). Nhưng trên valid/test, 22.7% token của bigram và 57.2% token của trigram nằm trong n-gram chưa từng thấy, tức P = 0, log 0 = -inf, nên perplexity là inf. Tỉ lệ từ ngoài vocabulary (OOV) chỉ khoảng 2.3%, vậy phần lớn zero-prob là do sparsity của n-gram chứ không phải từ mới. n càng lớn, tỉ lệ này càng cao. Đ

**2. Laplace: hết inf nhưng perplexity rất cao.**
Laplace cộng 1 cho mọi n-gram nên không còn xác suất 0, perplexity trở thành số hữu hạn. Nhưng mẫu số là C(h) + V với V rất lớn (214,567), nên khối lượng xác suất bị chia cho hàng trăm nghìn n-gram chưa từng xuất hiện. Ví dụ với context hiếm C(h) = 1 và C(h, w) = 1: MLE cho P = 1, còn Laplace chỉ cho 2/(1 + V) ≈ 9.3 × 10^-6. Vì vậy Laplace làm perplexity tăng cả trên train (bigram 152 → 6,398; trigram 14.5 → 35,179).

**3. Trigram Laplace tệ hơn bigram Laplace.**
Context 2 từ hiếm hơn context 1 từ, nhiều context có C(h) = 0 và khi đó P = 1/V cho mọi từ (gần như phân phối đều). Nên trigram mất nhiều xác suất vào n-gram chưa thấy hơn bigram.


---

# 19. Experiment 3 — Perplexity

30K documents, 8:1:1, seed 42, V = 214,567.

| Model | Unique n-grams | Train PPL | Valid PPL | Test PPL |
|---|---|---|---|---|
| Unigram MLE | 214,567 | 1,845.3 | inf | inf |
| Unigram Laplace | 214,567 | 1,853.2 | 1,951.0 | 1,961.1 |
| Bigram MLE | 2,617,160 | 152.1 | inf | inf |
| Bigram Laplace | 2,617,160 | 6,397.7 | 8,296.7 | 8,341.9 |
| Trigram MLE | 6,126,667 | 14.5 | inf | inf |
| Trigram Laplace | 6,126,667 | 35,179.3 | 57,497.4 | 57,973.8 |

Tác động của corpus size (valid PPL với Laplace, vocabulary cố định; trong ngoặc là % token valid có n-gram chưa thấy trong train):

| Train | Tokens | Unigram | Bigram | Trigram |
|---|---|---|---|---|
| 10% | ~0.94M | 2,156.5 (4.7%) | 30,879.8 (38.8%) | 108,824.3 (73.9%) |
| 25% | ~2.35M | 1,996.7 (3.3%) | 18,423.4 (31.4%) | 87,065.7 (67.4%) |
| 50% | ~4.70M | 1,960.1 (2.6%) | 12,398.7 (26.6%) | 71,737.4 (62.3%) |
| 100% | 9.41M | 1,951.0 (2.1%) | 8,296.7 (22.4%) | 57,497.4 (57.1%) |

**1. Tại sao training perplexity thay đổi?**
Với MLE, training perplexity giảm mạnh khi n tăng (1,845 → 152 → 14.5) vì context dài giúp đoán từ tiếp theo dễ hơn trên chính dữ liệu đã dùng để đếm. Trigram có 6.1M n-gram khác nhau (bằng khoảng 65% số token train), hầu hết chỉ xuất hiện một lần nên model gần như học thuộc train. Với Laplace thì ngược lại (1,853 → 6,398 → 35,179): add-one lấy xác suất của n-gram đã thấy chia cho hàng trăm nghìn n-gram chưa thấy, và context càng dài thì càng nhiều context hiếm nên mất càng nhiều.

**2. Tại sao validation/test có thể không cải thiện?**
Tăng n cho model nhiều thông tin ngữ cảnh hơn, nhưng dữ liệu để ước lượng mỗi context lại ít đi (sparsity). Kết quả: với MLE, valid/test là inf vì 2.1% (unigram), 22.4% (bigram), 57.1% (trigram) token thuộc n-gram chưa từng thấy. Với Laplace, valid PPL thậm chí tăng theo n (1,951 → 8,297 → 57,498), tức trigram tệ nhất, do sparsity cộng với việc add-one quá mạnh khi V lớn. Valid và test rất gần nhau (8,297 và 8,342 với bigram) nên kết luận này không phải do may rủi khi chia tập.

**3. Dấu hiệu overfitting.**
- MLE: train 14.5 nhưng valid/test là inf (trigram).
- Khoảng cách train → valid của Laplace tăng theo n: unigram +5% (1,853 → 1,951), bigram +30% (6,398 → 8,297), trigram +63% (35,179 → 57,498).
- Số tham số (unique n-gram) tăng từ 0.2M lên 6.1M trong khi số token train cố định ở 9.4M.

**4. Tác động của corpus size.**
Khi tăng dữ liệu, model bậc cao cải thiện nhiều nhất: từ 10% lên 100% train, valid PPL của trigram giảm 108,824 → 57,498 (gần một nửa), bigram giảm 30,880 → 8,297 (gần 4 lần), unigram chỉ giảm 2,157 → 1,951. Tỉ lệ n-gram chưa thấy cũng giảm theo (trigram 73.9% → 57.1%). Với 9.4M token, hơn một nửa trigram của valid vẫn chưa xuất hiện trong train, nên trigram vẫn chưa vượt được unigram/bigram ở đây. Model bậc cao cần lượng dữ liệu lớn hơn nhiều, và xu hướng cho thấy khoảng cách thu hẹp dần khi corpus tăng.


---

# 23. Context length

Câu hỏi: Context dài hơn có luôn tốt hơn không?

 (30K documents, 8:1:1, seed 42, V = 214,567).
  Ngoài MLE và Laplace còn thử add-k với k chọn trên valid (lưới 10, 1, 0.1, 0.01, 0.001, 0.0001).

**Số lượng n-gram và n-gram chưa từng thấy (unseen):**

| Model | Unique n-grams (train) | Unique n-grams (test) | Unseen types (test) | Unseen tokens % (test) | Unseen context % (test) |
|---|---|---|---|---|---|
| Unigram | 214,567 | 58,897 | 17,385 (29.5%) | 2.20 | 0.00 |
| Bigram | 2,617,160 | 495,779 | 238,833 (48.2%) | 22.66 | 2.20 |
| Trigram | 6,126,667 | 893,844 | 647,445 (72.4%) | 57.20 | 22.00 |

**Perplexity:**

| Model | Smoothing | Train PPL | Valid PPL | Test PPL |
|---|---|---|---|---|
| Unigram | MLE | 1,845.3 | inf | inf |
| Unigram | Laplace (cũng là add-k tốt nhất, k = 1) | 1,853.2 | 1,951.0 | 1,961.1 |
| Bigram | MLE | 152.1 | inf | inf |
| Bigram | Laplace | 6,397.7 | 8,296.7 | 8,341.9 |
| Bigram | add-k (k = 0.001) | 232.5 | 1,259.2 | 1,284.1 |
| Trigram | MLE | 14.5 | inf | inf |
| Trigram | Laplace | 35,179.3 | 57,497.4 | 57,973.8 |
| Trigram | add-k (k = 0.0001) | 43.7 | 9,501.5 | 9,615.0 |

**Trả lời: Không.** Context dài hơn không luôn tốt hơn.

**1. Training perplexity giảm đều theo n** (MLE: 1,845 → 152 → 14.5; add-k: 1,853 → 233 → 43.7). Context dài luôn khớp train hơn, nhưng đây chỉ là khả năng ghi nhớ, không phản ánh khả năng tổng quát hóa.

**2. Valid/test không giảm đều theo n.** Với add-k tốt nhất, test PPL là 1,961 (unigram) → 1,284 (bigram) → 9,615 (trigram): bigram tốt nhất, trigram tệ nhất. Bigram là điểm cân bằng giữa thông tin ngữ cảnh và lượng dữ liệu hiện có. Valid và test gần nhau (1,259 và 1,284) nên kết quả ổn định. Kết luận này cũng phụ thuộc vào smoothing: với add-one, unigram lại tốt nhất vì add-one làm hỏng bigram và trigram nặng hơn.

**3. Số n-gram tăng rất nhanh.** Từ 0.21M (unigram) lên 2.6M (bigram) và 6.1M (trigram), trong khi train chỉ có 9.4M token. Khoảng 86% trigram chỉ xuất hiện một lần (Experiment 1), tức mỗi tham số của trigram được ước lượng từ gần như một quan sát.

**4. Unseen n-gram tăng theo n.** Tỉ lệ token test có n-gram chưa từng thấy là 2.2% → 22.7% → 57.2%; tỉ lệ loại n-gram chưa thấy là 29.5% → 48.2% → 72.4%. Ở trigram, 22% vị trí còn có cả context chưa thấy, nên model không có thông tin ngữ cảnh nào và phải quay về phân phối đều (P = 1/V).

**5. Overfitting tăng theo n.** Tỉ lệ test PPL / train PPL (add-k) là 1.06 (unigram), 5.5 (bigram), 220 (trigram).

