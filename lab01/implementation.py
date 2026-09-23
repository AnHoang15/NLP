"""
Part E
8.1 Mục tiêu: Tự xây một phiên bản TF-IDF tối giản trên một corpus nhỏ.


8.2 Các hàm cần xây dựng:
    build_vocabulary()
    compute_counts()
    compute_tf()
    compute_idf()
    compute_tfidf()
    cosine_similarity()

8.3 Corpus kiểm thử:
    D1 = "cat eats fish"
    D2 = "dog eats fish"
    D3 = "cat likes fish"

8.4 Unit tests: mỗi hàm có ít nhất 1 test, dùng assert với sai số 1e-9.

Kiểm chứng bằng thư viện sklearn (so sánh giá trị TF-IDF, cosine similarity)
nằm ở file riêng: verify_sklearn.py — chạy sau khi có file này.
"""

import sys
import math

# Ép output ra UTF-8 để tránh lỗi UnicodeEncodeError trên Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==================================================================
# PHẦN 1 — CÁC HÀM CỐT LÕI (8.2)
# ==================================================================

def tokenize(text):
    text = text.lower()
    tokens = []
    current = []
    for ch in text:
        if ch.isalpha():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


# 1) build_vocabulary

def build_vocabulary(tokenized_docs):
    unique_terms = set()
    for tokens in tokenized_docs:
        unique_terms.update(tokens)
    vocabulary = sorted(unique_terms)
    term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    return vocabulary, term_to_index


# 2) compute_counts

def compute_counts(tokens, term_to_index):
    counts = [0] * len(term_to_index)
    for tok in tokens:
        idx = term_to_index.get(tok)
        if idx is not None:
            counts[idx] += 1
    return counts

# 3) compute_tf

def compute_tf(counts):
    total_tokens = sum(counts)
    if total_tokens == 0:
        return [0.0] * len(counts)
    return [c / total_tokens for c in counts]


# 4) compute_idf

def compute_idf(count_matrix):
    n_docs = len(count_matrix)
    vocab_size = len(count_matrix[0]) if n_docs > 0 else 0
    df = [0] * vocab_size
    for counts in count_matrix:
        for j, c in enumerate(counts):
            if c > 0:
                df[j] += 1
    idf = [math.log((1 + n_docs) / (1 + df_j)) + 1 for df_j in df]
    return idf


# 5) compute_tfidf

def compute_tfidf(tf, idf):
    return [tf_i * idf_i for tf_i, idf_i in zip(tf, idf)]

# 6) cosine_similarity
#    cos(a, b) = (a . b) / (||a|| * ||b||)

def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ==================================================================
# PHẦN 2 — UNIT TESTS (8.3 + 8.4)
# ==================================================================

# Corpus kiểm thử theo đúng đề bài 8.3
D1 = "cat eats fish"
D2 = "dog eats fish"
D3 = "cat likes fish"
TEST_CORPUS = [D1, D2, D3]


def approx_equal(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_tokenize():
    assert tokenize(D1) == ["cat", "eats", "fish"]
    assert tokenize(D2) == ["dog", "eats", "fish"]
    assert tokenize(D3) == ["cat", "likes", "fish"]
    # kiểm tra lowercase + bỏ dấu câu
    assert tokenize("Cat, EATS fish!!") == ["cat", "eats", "fish"]
    print("test_tokenize: OK")


def test_build_vocabulary():
    # Từ vựng mong đợi (sắp xếp alphabet): cat, dog, eats, fish, likes
    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)

    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocabulary == expected_vocab, f"Expected {expected_vocab}, got {vocabulary}"
    assert len(term_to_index) == 5
    assert term_to_index["cat"] == 0
    assert term_to_index["fish"] == 3
    print("test_build_vocabulary: OK")


def test_compute_counts():
    # D1 = "cat eats fish" -> cat=1, dog=0, eats=1, fish=1, likes=0
    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)

    counts_d1 = compute_counts(tokenized_docs[0], term_to_index)
    assert counts_d1 == [1, 0, 1, 1, 0], f"Got {counts_d1}"

    counts_d2 = compute_counts(tokenized_docs[1], term_to_index)
    assert counts_d2 == [0, 1, 1, 1, 0], f"Got {counts_d2}"

    counts_d3 = compute_counts(tokenized_docs[2], term_to_index)
    assert counts_d3 == [1, 0, 0, 1, 1], f"Got {counts_d3}"
    print("test_compute_counts: OK")


def test_compute_tf():
    
    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    counts_d1 = compute_counts(tokenized_docs[0], term_to_index)
    tf_d1 = compute_tf(counts_d1)

    tf_cat = tf_d1[term_to_index["cat"]]
    expected_value = 1 / 3
    assert abs(tf_cat - expected_value) < 1e-9, f"tf_cat={tf_cat}, expected={expected_value}"


    tf_dog = tf_d1[term_to_index["dog"]]
    assert approx_equal(tf_dog, 0.0)

    # tổng TF của 1 document (khi mọi từ đều unique) phải bằng 1
    assert approx_equal(sum(tf_d1), 1.0)
    print("test_compute_tf: OK")


def test_compute_idf():

    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    count_matrix = [compute_counts(t, term_to_index) for t in tokenized_docs]
    idf = compute_idf(count_matrix)

    idf_fish = idf[term_to_index["fish"]]
    expected_idf_fish = math.log((1 + 3) / (1 + 3)) + 1  # = 1.0
    assert abs(idf_fish - expected_idf_fish) < 1e-9

    idf_dog = idf[term_to_index["dog"]]
    expected_idf_dog = math.log(4 / 2) + 1
    assert abs(idf_dog - expected_idf_dog) < 1e-9

    assert idf_fish == min(idf)
    print("test_compute_idf: OK")


def test_compute_tfidf():
    # TF-IDF(cat, D1) = TF(cat, D1) * IDF(cat) = (1/3) * (log(4/3)+1)
    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    count_matrix = [compute_counts(t, term_to_index) for t in tokenized_docs]
    idf = compute_idf(count_matrix)

    tf_d1 = compute_tf(count_matrix[0])
    tfidf_d1 = compute_tfidf(tf_d1, idf)

    expected_tfidf_cat = (1 / 3) * (math.log(4 / 3) + 1)  # df(cat)=2
    tfidf_cat = tfidf_d1[term_to_index["cat"]]
    assert abs(tfidf_cat - expected_tfidf_cat) < 1e-9, (
        f"tfidf_cat={tfidf_cat}, expected={expected_tfidf_cat}"
    )

    # "dog" không có trong D1 -> TF-IDF(dog, D1) phải bằng 0
    assert approx_equal(tfidf_d1[term_to_index["dog"]], 0.0)
    print("test_compute_tfidf: OK")


def test_cosine_similarity():
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert approx_equal(cosine_similarity(a, b), 0.0)

    zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(zero, v) == 0.0

    # so sánh thực tế trên corpus D1, D2, D3:
 
    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    count_matrix = [compute_counts(t, term_to_index) for t in tokenized_docs]
    idf = compute_idf(count_matrix)
    tfidf_matrix = [compute_tfidf(compute_tf(c), idf) for c in count_matrix]

    sim_d1_d2 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    sim_d1_d3 = cosine_similarity(tfidf_matrix[0], tfidf_matrix[2])
    sim_d2_d3 = cosine_similarity(tfidf_matrix[1], tfidf_matrix[2])

    assert abs(sim_d1_d2 - sim_d1_d3) < 1e-9
    assert sim_d2_d3 < sim_d1_d2
    print("test_cosine_similarity: OK")


def run_all_tests():
    print("=" * 60)
    print("CHẠY UNIT TESTS (8.3 + 8.4) trên corpus D1, D2, D3")
    print("=" * 60)
    test_tokenize()
    test_build_vocabulary()
    test_compute_counts()
    test_compute_tf()
    test_compute_idf()
    test_compute_tfidf()
    test_cosine_similarity()
    print("\nTất cả các test đều PASS.\n")


# ==================================================================
# PHẦN 3 — DEMO PIPELINE ĐẦY ĐỦ TRÊN 1 CORPUS NHỎ KHÁC (8.1)
# ==================================================================
DEMO_CORPUS = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "Cats and dogs are great pets",
    "I love my cat and my dog",
    "The mat and the log are on the floor",
]


def run_demo():
    print("=" * 60)
    print("DEMO PIPELINE TF-IDF TRÊN CORPUS NHỎ")
    print("=" * 60)

    print(f"Corpus ({len(DEMO_CORPUS)} documents):")
    for i, doc in enumerate(DEMO_CORPUS):
        print(f"  [{i}] {doc}")

    tokenized_docs = [tokenize(doc) for doc in DEMO_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    print(f"\nVocabulary ({len(vocabulary)} từ): {vocabulary}")

    count_matrix = [compute_counts(tokens, term_to_index) for tokens in tokenized_docs]
    tf_matrix = [compute_tf(counts) for counts in count_matrix]
    idf = compute_idf(count_matrix)
    tfidf_matrix = [compute_tfidf(tf, idf) for tf in tf_matrix]

    print("\nTop 3 từ TF-IDF cao nhất mỗi document:")
    for i, vec in enumerate(tfidf_matrix):
        ranked = sorted(zip(vocabulary, vec), key=lambda x: x[1], reverse=True)
        top3 = [(t, round(v, 3)) for t, v in ranked[:3] if v > 0]
        print(f"  doc[{i}]: {top3}")

    n_docs = len(DEMO_CORPUS)
    print("\nMa trận cosine similarity giữa các document:")
    header = "        " + "".join(f"doc{j:<8}" for j in range(n_docs))
    print(header)
    for i in range(n_docs):
        row_vals = [f"{cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]):<10.3f}" for j in range(n_docs)]
        print(f"  doc{i}: " + "".join(row_vals))

    best_pair, best_sim = None, -1
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])
            if sim > best_sim:
                best_sim = sim
                best_pair = (i, j)
    print(
        f"\n-> Hai document giống nhau nhất: doc[{best_pair[0]}] và doc[{best_pair[1]}] "
        f"(cosine similarity = {best_sim:.4f})"
    )


if __name__ == "__main__":
    run_all_tests()
    run_demo()