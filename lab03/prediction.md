# LAB 03 — Prediction

## Phần 1: Prediction trước experiment (Mục 9)

### Prediction 1 — Những từ nào gần nhau nhất? (doctor, physician, hospital, banana, car)

- **Prediction:** `doctor` – `physician` gần nhất, tiếp theo là `doctor` – `hospital`. `banana` và `car` xa nhóm y tế; `banana` xa nhất.
- **Reason:** `doctor` và `physician` xuất hiện trong ngữ cảnh gần như giống nhau (treated, patient, clinic...). `hospital` thường đồng xuất hiện với `doctor` nhưng không thay thế được cho nhau. `banana` và `car` hầu như không chung ngữ cảnh với nhóm y tế.
- **Confidence:** Cao (~80%) cho thứ hạng nhóm y tế so với `banana`/`car`; trung bình (~50%) cho thứ tự `physician` vs `hospital`.

### Prediction 2 — Tăng context window từ 2 → 5, similarity có thay đổi không?

- **Prediction:** Có thay đổi. Window lớn hơn làm các cặp liên quan theo chủ đề (`doctor`–`hospital`) tăng similarity; các cặp tương đồng về chức năng/cú pháp có thể giảm nhẹ do nhiễu từ ngữ cảnh xa.
- **Reason:** Window nhỏ bắt quan hệ cú pháp/gần nghĩa; window lớn bắt quan hệ chủ đề (topical) nhưng lẫn thêm từ không liên quan.
- **Confidence:** Cao (~85%) là có thay đổi; trung bình (~55%) về hướng thay đổi.

### Prediction 3 — Dimension 50 → 100 → 300, chất lượng có chắc chắn tăng không?

- **Prediction:** Không chắc chắn. Thường tăng từ 50 → 100, sau đó lợi ích giảm dần hoặc không tăng; với corpus nhỏ, 300 chiều có thể không tốt hơn, thậm chí overfit.
- **Reason:** Chất lượng phụ thuộc vào lượng dữ liệu, không chỉ capacity. Dimension lớn cần nhiều dữ liệu hơn, đồng thời tốn thêm thời gian train và bộ nhớ.
- **Confidence:** Trung bình–cao (~70%).

### Prediction 4 — `doctor` và `physician` có chắc chắn gần nhau nếu corpus chỉ có 100 câu?

- **Prediction:** Không chắc chắn.
- **Reason:** Corpus nhỏ → hai từ có thể xuất hiện rất ít (thậm chí dưới `min_count`), ngữ cảnh không đủ để học vector ổn định; kết quả phụ thuộc nhiều vào khởi tạo ngẫu nhiên và nhiễu. Nếu hai từ xuất hiện trong các câu gần giống nhau thì vẫn có thể gần, nhưng không đảm bảo.
- **Confidence:** Cao (~75%).



## Phần 2: CBOW vs Skip-gram (Mục 16)

Câu: `the cat eats fish` — window = 1

### CBOW (context → target)

| Context | Target |
|---|---|
| [cat] | the |
| [the, eats] | cat |
| [cat, fish] | eats |
| [eats] | fish |

→ 4 training examples (mỗi vị trí là 1 mẫu, input là tập context).

### Skip-gram (target → context)

| Target | Context |
|---|---|
| the | cat |
| cat | the |
| cat | eats |
| eats | cat |
| eats | fish |
| fish | eats |

→ 6 training pairs (mỗi cặp target–context là 1 mẫu).

### Khác biệt

- **CBOW:** input = các từ context, output = từ target; mỗi vị trí tạo 1 mẫu.
- **Skip-gram:** input = từ target, output = từng từ context; mỗi vị trí tạo nhiều mẫu.