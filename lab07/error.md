# Bài 26 - Error analysis

## Mục tiêu

Chọn **3 dự đoán đúng** và **3 dự đoán sai** từ Transformer IMDb ở notebook Bài 23. Với mỗi review, đối chiếu văn bản gốc, nhãn thật, nhãn dự đoán và xác suất positive; sau đó ghi nhận yếu tố có thể liên quan đến kết quả.

## Kết quả tổng quát của Bài 23

Trên 25.000 review test, mô hình có accuracy `0.8071`, precision `0.8420`, recall `0.7562` và F1 `0.7968` cho lớp positive. Confusion matrix theo thứ tự `[negative, positive]`:

| Nhãn thật | Dự đoán negative | Dự đoán positive |
|---|---:|---:|
| negative | 10.726 | 1.774 |
| positive | 3.048 | 9.452 |

Mô hình bỏ sót 3.048 review positive (false negative), nhiều hơn số review negative bị đoán thành positive (1.774 false positive). Dùng các con số này làm bối cảnh; không thay dự đoán của mô hình sau khi đã xem nhãn thật.

## Lấy ví dụ đúng và sai từ notebook Bài 23

Chạy cell sau trong runtime Bài 23 sau khi đã chạy các cell tải dữ liệu, huấn luyện và đánh giá. Đoạn mã dùng đúng các biến `test_predictions`, `test_probabilities`, `y_test`, `test_texts`, `tokenizer` và chọn mẫu ngẫu nhiên có seed cố định.

```python
import numpy as np
import pandas as pd

correct_indices = np.flatnonzero(test_predictions == y_test)
incorrect_indices = np.flatnonzero(test_predictions != y_test)
if len(correct_indices) < 3 or len(incorrect_indices) < 3:
    raise ValueError("Cần ít nhất 3 dự đoán đúng và 3 dự đoán sai.")

rng = np.random.default_rng(26)
selected = list(rng.choice(correct_indices, size=3, replace=False))
selected += list(rng.choice(incorrect_indices, size=3, replace=False))

label_name = {0: "negative", 1: "positive"}
rows = []
for group, index in zip(["Đúng"] * 3 + ["Sai"] * 3, selected):
    review = test_texts[index]
    token_count = len(tokenizer.texts_to_sequences([review])[0])
    rows.append({
        "Nhóm": group,
        "Index": int(index),
        "Nhãn thật": label_name[int(y_test[index])],
        "Dự đoán": label_name[int(test_predictions[index])],
        "P(positive)": round(float(test_probabilities[index]), 4),
        "Tokens trước padding/truncation": token_count,
        "Review": review[:700] + ("..." if len(review) > 700 else ""),
    })

display(pd.DataFrame(rows))
```

Nếu mẫu ngẫu nhiên không thể hiện rõ các hiện tượng cần thảo luận, chọn lại trong nhóm đúng/sai nhưng vẫn giữ nguyên nhãn dự đoán của mô hình. Không chỉnh sửa dữ liệu hoặc kết quả để tạo ví dụ thuận tiện.

## Bảng phân tích sáu ví dụ

Điền bảng sau sau khi xem output của cell trên. Một review có thể có nhiều yếu tố; ghi yếu tố có bằng chứng trong văn bản và nêu ngắn gọn vì sao nó có thể gây khó cho mô hình.

| # | Index | Đúng/Sai | Nhãn thật → dự đoán | P(positive) | Hiện tượng có liên quan | Bằng chứng trong review và phân tích |
|---:|---:|---|---|---:|---|---|
| 1 |  | Đúng |  |  |  |  |
| 2 |  | Đúng |  |  |  |  |
| 3 |  | Đúng |  |  |  |  |
| 4 |  | Sai |  |  |  |  |
| 5 |  | Sai |  |  |  |  |
| 6 |  | Sai |  |  |  |  |

### Các hiện tượng cần xem xét

- **Negation:** phủ định như “not good” hoặc “didn't enjoy”; kiểm tra xem ý nghĩa của phủ định có đảo cực tính không.
- **Contrast:** các mệnh đề trái chiều nối bằng “but”, “although”, “however”; chỉ ra phần nào có thể chi phối nhãn toàn review.
- **Long sequence / truncation:** kiểm tra số token và `MAX_LEN` của Bài 23 (`128`). Nếu review dài hơn giới hạn, phần cuối bị cắt; ghi rõ khi kết luận dựa trên việc cắt này.
- **Rare word:** từ hiếm hoặc không có trong vocabulary có thể thành `<OOV>`; chỉ kết luận khi tokenization cho thấy điều đó.
- **Ambiguous sentiment:** review trung tính, pha trộn, mỉa mai hoặc thiếu ý kiến rõ ràng.
- **Domain-specific term:** từ chỉ có sắc thái trong ngữ cảnh phim, diễn xuất hoặc thể loại.
- **Noise:** lỗi chính tả, dấu câu, HTML, câu vụn hoặc nội dung không liên quan.

Không cần ép mỗi hiện tượng xuất hiện trong một ví dụ. Nếu không thấy bằng chứng, ghi “không rõ trong mẫu này”.

## Attention visualization (Transformer)

Nếu muốn kiểm tra phân phối attention trên một trong sáu review, cell dưới đây dùng layer `transformer_encoder` của Bài 23 để lấy attention scores. Nó giả định mô hình trong runtime có tên biến `model` và có một Transformer Encoder như trong notebook Bài 23.

```python
import matplotlib.pyplot as plt
import tensorflow as tf

encoder = model.get_layer("transformer_encoder")
encoder_vectors, valid_mask = encoder.input
attention_mask = valid_mask[:, tf.newaxis, :]
_, attention_scores = encoder.attention(
    query=encoder_vectors,
    value=encoder_vectors,
    key=encoder_vectors,
    attention_mask=attention_mask,
    return_attention_scores=True,
    training=False,
)
attention_model = tf.keras.Model(model.input, attention_scores)

# Thay index bằng một Index trong bảng sáu ví dụ.
index = int(selected[3])
weights = attention_model.predict(X_test[index:index + 1], verbose=0)[0]
token_ids = X_test[index]
valid_ids = token_ids[token_ids != 0]
tokens = [tokenizer.index_word.get(int(token_id), "<UNK>") for token_id in valid_ids]

head = 0  # Có thể đổi sang head khác để quan sát phân phối.
tick_step = max(1, len(tokens) // 30)
tick_positions = range(0, len(tokens), tick_step)
plt.figure(figsize=(12, 8))
plt.imshow(weights[head, :len(tokens), :len(tokens)], cmap="viridis", aspect="auto")
plt.xticks(tick_positions, [tokens[i] for i in tick_positions], rotation=90)
plt.yticks(tick_positions, [tokens[i] for i in tick_positions])
plt.xlabel("Key token")
plt.ylabel("Query token")
plt.title(f"Attention weights - test index {index}, head {head + 1}")
plt.colorbar(label="Attention weight")
plt.tight_layout()
plt.show()
```

Ghi token nào nhận/phân phối attention nổi bật và đối chiếu với văn bản, nhãn thật, dự đoán. **Attention visualization chỉ cho thấy phân phối attention weights. Attention không đồng nghĩa với lời giải thích đầy đủ cho dự đoán.** Phân biệt hành vi của mô hình với cách con người diễn giải review.

## Kết luận

Trong sáu ví dụ, mô hình xử lý tốt trường hợp nào? Lỗi thường gặp liên quan đến hiện tượng nào? Có ví dụ nào mà xác suất dự đoán tự tin nhưng sai không? Nêu bằng chứng cụ thể từ các review đã chọn và giới hạn của phân tích mẫu nhỏ.

## AI assistance statement

Khung phân tích và mã lấy mẫu được tạo với Codex/ChatGPT. Ghi lại phần đã tự sửa, chạy và xác minh theo yêu cầu môn học.
