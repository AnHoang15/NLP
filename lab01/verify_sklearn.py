"""
8.5:Kiểm chứng bản TF-IDF tự viết (implementation.py) bằng thư viện sklearn.

- Dùng sklearn.feature_extraction.text.TfidfVectorizer để tính TF-IDF
  "chuẩn" cho corpus D1, D2, D3 (8.3), rồi so sánh từng giá trị với kết
  quả tính thủ công trong implementation.py.
- Ghi kết quả kiểm chứng vào cuối file result.csv 
"""

import sys
import os
import csv
import math

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer

from implementation import (
    tokenize,
    build_vocabulary,
    compute_counts,
    compute_tf,
    compute_idf,
    compute_tfidf,
    cosine_similarity,
    D1,
    D2,
    D3,
    TEST_CORPUS,
)

RESULT_CSV = "result.csv"


def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return vec if norm == 0 else [x / norm for x in vec]


def main():
    
    print("KIỂM CHỨNG BẰNG THƯ VIỆN sklearn (TfidfVectorizer)")
    print("Corpus:")
    for i, d in enumerate(TEST_CORPUS, start=1):
        print(f"  D{i} = \"{d}\"")

    # 1) Tính TF-IDF bằng implementation.py 

    tokenized_docs = [tokenize(d) for d in TEST_CORPUS]
    vocabulary, term_to_index = build_vocabulary(tokenized_docs)
    count_matrix = [compute_counts(t, term_to_index) for t in tokenized_docs]
    idf = compute_idf(count_matrix)
    tf_matrix = [compute_tf(c) for c in count_matrix]
    tfidf_matrix_raw = [compute_tfidf(tf, idf) for tf in tf_matrix]
    # sklearn mặc định chuẩn hoá L2 từng dòng -> chuẩn hoá bản tự viết để so sánh công bằng
    tfidf_matrix_manual = [l2_normalize(v) for v in tfidf_matrix_raw]

    # 2) Tính TF-IDF bằng sklearn 
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, smooth_idf=True, norm="l2")
    sk_matrix = vectorizer.fit_transform(TEST_CORPUS).toarray()
    sk_vocab = vectorizer.get_feature_names_out().tolist()

    vocab_match = sk_vocab == vocabulary
    print(f"\nVocabulary tự xây : {vocabulary}")
    print(f"Vocabulary sklearn: {sk_vocab}")
    print(f"-> Vocabulary khớp nhau: {vocab_match}")

    # 3) So sánh từng giá trị TF-IDF 
    comparison_rows = [] 
    max_diff = 0.0
    for i, doc_name in enumerate(["D1", "D2", "D3"]):
        for j, term in enumerate(vocabulary):
            manual_val = tfidf_matrix_manual[i][j]
            sk_val = sk_matrix[i][j]
            diff = abs(manual_val - sk_val)
            max_diff = max(max_diff, diff)
            comparison_rows.append((doc_name, term, manual_val, sk_val, diff))

    tolerance = 1e-9
    all_pass = max_diff < tolerance
    print(f"\nSai lệch lớn nhất giữa TF-IDF tự viết và sklearn: {max_diff:.2e}")
    print(f"Kết luận: {'PASS' if all_pass else 'FAIL'} (ngưỡng cho phép = {tolerance:.0e})")

    print("\nChi tiết so sánh (document, term, manual, sklearn, diff):")
    for doc_name, term, manual_val, sk_val, diff in comparison_rows:
        if manual_val > 0 or sk_val > 0:
            print(f"  {doc_name:<4} {term:<10} manual={manual_val:.6f}  sklearn={sk_val:.6f}  diff={diff:.2e}")

    # 4) So sánh cosine similarity giữa các cặp document
    pairs = [(0, 1, "D1-D2"), (0, 2, "D1-D3"), (1, 2, "D2-D3")]
    sim_rows = []  # (pair, manual_sim, sklearn_sim, diff)
    for i, j, label in pairs:
        manual_sim = cosine_similarity(tfidf_matrix_manual[i], tfidf_matrix_manual[j])
        sk_sim = cosine_similarity(list(sk_matrix[i]), list(sk_matrix[j]))
        sim_rows.append((label, manual_sim, sk_sim, abs(manual_sim - sk_sim)))

    print("\nSo sánh cosine similarity (manual vs sklearn):")
    for label, manual_sim, sk_sim, diff in sim_rows:
        print(f"  {label}: manual={manual_sim:.6f}  sklearn={sk_sim:.6f}  diff={diff:.2e}")

    # 5) Ghi kết quả vào result.csv
    file_exists = os.path.exists(RESULT_CSV)

    encoding = "utf-8" if file_exists else "utf-8-sig"

    with open(RESULT_CSV, "a", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["8. KIỂM CHỨNG TF-IDF TỰ VIẾT BẰNG THƯ VIỆN sklearn"])
        writer.writerow([])

        writer.writerow(["Corpus kiểm thử (8.3)"])
        writer.writerow(["D1", D1])
        writer.writerow(["D2", D2])
        writer.writerow(["D3", D3])
        writer.writerow([])

        writer.writerow(["Vocabulary tự xây", "; ".join(vocabulary)])
        writer.writerow(["Vocabulary sklearn", "; ".join(sk_vocab)])
        writer.writerow(["Vocabulary khớp nhau", vocab_match])
        writer.writerow([])

        writer.writerow(["document", "term", "tfidf_manual", "tfidf_sklearn", "abs_diff"])
        for doc_name, term, manual_val, sk_val, diff in comparison_rows:
            writer.writerow([doc_name, term, f"{manual_val:.6f}", f"{sk_val:.6f}", f"{diff:.2e}"])
        writer.writerow([])

        writer.writerow(["cap_document", "cosine_manual", "cosine_sklearn", "abs_diff"])
        for label, manual_sim, sk_sim, diff in sim_rows:
            writer.writerow([label, f"{manual_sim:.6f}", f"{sk_sim:.6f}", f"{diff:.2e}"])
        writer.writerow([])

        writer.writerow(["Sai lệch lớn nhất (TF-IDF)", f"{max_diff:.2e}"])
        writer.writerow(["Ngưỡng cho phép", f"{tolerance:.0e}"])
        writer.writerow(["Kết luận", "PASS" if all_pass else "FAIL"])

    print(f"\nĐã ghi thêm kết quả kiểm chứng vào cuối file: {RESULT_CSV}")


if __name__ == "__main__":
    main()