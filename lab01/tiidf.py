"""
Part D
Pipeline: Raw documents -> Tokenizer -> CountVectorizer -> TF -> IDF -> TF-IDF matrix
Dữ liệu: c4-train_00000-of-01024-30K_json.gz 
"""

import sys
import csv
import gzip
import json
import re
import time

# Ép output ra UTF-8 để tránh lỗi UnicodeEncodeError trên Windows (cp1252)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

DATA_PATH = "C:/HUS/NLP/lab01/c4-train.00000-of-01024-30K.json.gz"
N_DOCS = 30000          # số văn bản dùng để demo 
RESULT_CSV = "result.csv"  # file ghi toàn bộ kết quả 7.3 / 7.4 / 7.5
DOC_INDEX = 0               # index văn bản được chọn để xem TF-IDF cao nhất (câu 7.5)
TOP_K = 20                  # số lượng terms lấy ra ở mỗi danh sách (câu 7.5)


# ------------------------------------------------------------------
# BƯỚC 0: Raw documents — đọc dữ liệu thô từ file .gz
# ------------------------------------------------------------------
def load_documents(path, n_docs=None):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n_docs is not None and i >= n_docs:
                break
            obj = json.loads(line)
            docs.append(obj["text"])
    return docs


# ------------------------------------------------------------------
# BƯỚC 1: Tokenizer — tách văn bản thành các token (từ)
# ------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")

def tokenizer(text: str):
    text = text.lower()
    return TOKEN_RE.findall(text)


def main():
    t0 = time.time()
    print("Đang đọc dữ liệu...")
    documents = load_documents(DATA_PATH, N_DOCS)
    print(f"-> Đã đọc {len(documents)} văn bản trong {time.time() - t0:.2f}s")

    # Xem thử tokenizer hoạt động trên 1 văn bản
    sample_tokens = tokenizer(documents[0])
    print(f"\nVí dụ tokenizer trên văn bản đầu tiên ({len(sample_tokens)} token):")
    print(sample_tokens[:15], "...")

    # ------------------------------------------------------------------
    # BƯỚC 2: CountVectorizer — đếm tần suất token, tạo ma trận đếm (Bag of Words)
    # ------------------------------------------------------------------
    t1 = time.time()
    count_vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,     # tắt cảnh báo vì đã tự cung cấp tokenizer
        min_df=2,               # bỏ từ chỉ xuất hiện ở đúng 1 văn bản (không giới hạn max_features)
    )
    count_matrix = count_vectorizer.fit_transform(documents)
    print(f"\n[CountVectorizer] shape = {count_matrix.shape}, "
          f"thời gian = {time.time() - t1:.2f}s")
    vocab = count_vectorizer.get_feature_names_out()
    print(f"Kích thước từ vựng: {len(vocab)}")

    # ------------------------------------------------------------------
    # BƯỚC 3: TF — Term Frequency (chuẩn hoá số đếm theo từng văn bản)
    #   TF(t, d) = số lần từ t xuất hiện trong d / tổng số từ trong d
    # ------------------------------------------------------------------
    t2 = time.time()
    row_sums = np.array(count_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # tránh chia cho 0
    tf_matrix = count_matrix.multiply(1 / row_sums[:, None]).tocsr()
    print(f"\n[TF] shape = {tf_matrix.shape}, thời gian = {time.time() - t2:.2f}s")

    # ------------------------------------------------------------------
    # BƯỚC 4: IDF — Inverse Document Frequency
    #   IDF(t) = log( (1 + N) / (1 + df(t)) ) + 1   (smooth idf, giống sklearn)
    # ------------------------------------------------------------------
    t3 = time.time()
    n_docs_total = count_matrix.shape[0]
    df = np.array((count_matrix > 0).sum(axis=0)).flatten()  # document frequency
    idf = np.log((1 + n_docs_total) / (1 + df)) + 1
    idf_diag = sparse.diags(idf)
    print(f"\n[IDF] tính xong {len(idf)} giá trị idf, thời gian = {time.time() - t3:.2f}s")

    # ------------------------------------------------------------------
    # BƯỚC 5: TF-IDF matrix = TF * IDF, sau đó chuẩn hoá L2 theo hàng
    # ------------------------------------------------------------------
    t4 = time.time()
    tfidf_matrix = tf_matrix @ idf_diag
    norms = sparse.linalg.norm(tfidf_matrix, axis=1)
    norms[norms == 0] = 1
    tfidf_matrix = sparse.diags(1 / norms) @ tfidf_matrix
    tfidf_matrix = tfidf_matrix.tocsr()
    print(f"\n[TF-IDF] shape = {tfidf_matrix.shape}, thời gian = {time.time() - t4:.2f}s")

    # ------------------------------------------------------------------
    # Đối chiếu với TfidfTransformer của sklearn (kiểm tra kết quả đúng)
    # ------------------------------------------------------------------
    sk_tfidf = TfidfTransformer(smooth_idf=True, norm="l2").fit_transform(count_matrix)
    diff = abs(tfidf_matrix - sk_tfidf).max()
    print(f"\nSai lệch lớn nhất so với sklearn TfidfTransformer: {diff:.2e} "
          f"(gần 0 nghĩa là kết quả khớp)")

    # In thử top-10 từ có điểm TF-IDF cao nhất trong văn bản đầu tiên
    row = tfidf_matrix.getrow(0).toarray().flatten()
    top_idx = row.argsort()[::-1][:10]
    print("\nTop 10 từ TF-IDF cao nhất trong văn bản đầu tiên:")
    for i in top_idx:
        if row[i] > 0:
            print(f"  {vocab[i]:<15} {row[i]:.4f}")

    # ------------------------------------------------------------------
    # 7.3. KIỂM TRA KÍCH THƯỚC
    #   X thuộc R^(N x V):  N = số document, V = kích thước vocabulary
    # ------------------------------------------------------------------
    N = tfidf_matrix.shape[0]   
    V = tfidf_matrix.shape[1]  
    print("\n" + "=" * 60)
    print("7.3. KIỂM TRA KÍCH THƯỚC")
    print("=" * 60)
    print(f"Number of documents = {N}")
    print(f"Vocabulary size     = {V}")
    print(f"Matrix shape        = {tfidf_matrix.shape}")
    print(f"X thuộc R^(N x V) với N = {N} (số document), V = {V} (kích thước vocabulary)")

    # ------------------------------------------------------------------
    # 7.4. KIỂM TRA SPARSITY
    #   S = 1 - nnz(X) / (N * V)
    # ------------------------------------------------------------------
    nnz = tfidf_matrix.nnz  # số phần tử khác 0
    sparsity = 1 - nnz / (N * V)
    print("\n" + "=" * 60)
    print("7.4. KIỂM TRA SPARSITY")
    print("=" * 60)
    print(f"nnz(X)   = {nnz}")
    print(f"N x V    = {N * V}")
    print(f"Sparsity S = 1 - nnz(X)/(N*V) = {sparsity:.6f}  ({sparsity*100:.4f}% các phần tử bằng 0)")
    print(
        "\n-> Trả lời: mỗi document chỉ dùng một phần rất nhỏ vocabulary (vài chục/vài trăm "
        "từ khác nhau), nhưng vector vẫn phải có đủ V chiều vì tất cả document phải được "
        "biểu diễn trên CÙNG một không gian đặc trưng chung (chung một trục toạ độ ứng với "
        "toàn bộ từ vựng của corpus) thì mới so sánh được với nhau (vd: cosine similarity). "
        "Những chiều mà document không dùng tới đơn giản là = 0, khiến ma trận X rất thưa (sparse)."
    )

    # ------------------------------------------------------------------
    # 7.5. INSPECT VOCABULARY
    # ------------------------------------------------------------------
   
    print("7.5. INSPECT VOCABULARY")
    
    # (a) Top-K terms phổ biến nhất theo document frequency
    top_df_idx = df.argsort()[::-1][:TOP_K]
    top_df_terms = [(vocab[i], int(df[i])) for i in top_df_idx]
    print(f"\n(a) Top {TOP_K} terms theo document frequency (df):")
    for term, val in top_df_terms:
        print(f"  {term:<15} df={val}")

    # (b) Top-K terms có IDF cao nhất
    top_idf_idx = idf.argsort()[::-1][:TOP_K]
    top_idf_terms = [(vocab[i], float(idf[i])) for i in top_idf_idx]
    print(f"\n(b) Top {TOP_K} terms có IDF cao nhất:")
    for term, val in top_idf_terms:
        print(f"  {term:<15} idf={val:.4f}")

    # (c) Top-K terms có TF-IDF cao nhất trong document DOC_INDEX
    doc_row = tfidf_matrix.getrow(DOC_INDEX).toarray().flatten()
    top_tfidf_idx = doc_row.argsort()[::-1][:TOP_K]
    top_tfidf_terms = [(vocab[i], float(doc_row[i])) for i in top_tfidf_idx if doc_row[i] > 0]
    print(f"\n(c) Top {len(top_tfidf_terms)} terms có TF-IDF cao nhất trong document #{DOC_INDEX}:")
    for term, val in top_tfidf_terms:
        print(f"  {term:<15} tfidf={val:.4f}")

    # So sánh 3 danh sách
    set_df = {t for t, _ in top_df_terms}
    set_idf = {t for t, _ in top_idf_terms}
    set_tfidf = {t for t, _ in top_tfidf_terms}
    overlap_df_idf = set_df & set_idf
    overlap_df_tfidf = set_df & set_tfidf
    overlap_idf_tfidf = set_idf & set_tfidf
    print("\nSo sánh 3 danh sách:")
    print(f"  Giao giữa (df) và (idf)   : {sorted(overlap_df_idf) or '(rỗng)'}")
    print(f"  Giao giữa (df) và (tfidf) : {sorted(overlap_df_tfidf) or '(rỗng)'}")
    print(f"  Giao giữa (idf) và (tfidf): {sorted(overlap_idf_tfidf) or '(rỗng)'}")
    print(
        "\n-> Nhận xét: danh sách (a) toàn các từ phổ biến/stop-word (the, a, to, of...) vì df "
        "cao đồng nghĩa idf thấp -> hầu như KHÔNG trùng với danh sách (b). Danh sách (b) là các "
        "từ cực hiếm (chỉ xuất hiện ở rất ít document) nên idf cao nhưng chưa chắc có mặt trong "
        "document đang xét -> càng ít trùng với danh sách (c). Danh sách (c) là các từ vừa xuất "
        "hiện trong document đang xét, vừa có idf đủ cao (tức từ đặc trưng riêng cho document đó)."
    )

    # ------------------------------------------------------------------
    # Câu hỏi
    # ------------------------------------------------------------------
    
    print("CÂU HỎI")
   
    q1 = (
        "Một term xuất hiện rất nhiều trong corpus có nhất thiết có TF-IDF cao không? "
        "-> KHÔNG. Term xuất hiện ở nhiều document (df cao) sẽ có IDF thấp "
        "(IDF = log((1+N)/(1+df))+1, df càng lớn thì IDF càng nhỏ), nên dù TF trong 1 "
        "document nào đó cao, tích TF*IDF vẫn có thể thấp. Ví dụ điển hình là các stop-word "
        "(the, is, a...) xuất hiện ở hầu hết mọi document nhưng TF-IDF gần như bằng 0."
    )
    q2 = (
        "Một term có IDF cao có nhất thiết có TF-IDF cao trong mọi document không? "
        "-> KHÔNG. IDF là một hằng số tính trên toàn bộ corpus, không phụ thuộc vào từng "
        "document cụ thể. Nếu term đó không xuất hiện (hoặc xuất hiện rất ít) trong một "
        "document, thì TF của nó trong document đó gần bằng 0, khiến TF-IDF = TF*IDF cũng "
        "gần bằng 0 tại document đó, bất kể IDF cao đến đâu."
    )
    print(q1)
    print()
    print(q2)

    # ------------------------------------------------------------------
    # GHI TOÀN BỘ KẾT QUẢ RA result.csv
    # ------------------------------------------------------------------
    with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow(["Section", "Metric", "Value"])
        writer.writerow(["7.3", "Number of documents (N)", N])
        writer.writerow(["7.3", "Vocabulary size (V)", V])
        writer.writerow(["7.3", "Matrix shape", f"({N}, {V})"])
        writer.writerow([])

        writer.writerow(["Section", "Metric", "Value"])
        writer.writerow(["7.4", "nnz(X)", nnz])
        writer.writerow(["7.4", "N x V", N * V])
        writer.writerow(["7.4", "Sparsity S", f"{sparsity:.6f}"])
        writer.writerow(["7.4", "Sparsity (%)", f"{sparsity*100:.4f}%"])
        writer.writerow([])

        writer.writerow([f"7.5(a) Top {TOP_K} terms theo document frequency"])
        writer.writerow(["rank", "term", "document_frequency"])
        for rank, (term, val) in enumerate(top_df_terms, start=1):
            writer.writerow([rank, term, val])
        writer.writerow([])

        writer.writerow([f"7.5(b) Top {TOP_K} terms theo IDF"])
        writer.writerow(["rank", "term", "idf"])
        for rank, (term, val) in enumerate(top_idf_terms, start=1):
            writer.writerow([rank, term, f"{val:.6f}"])
        writer.writerow([])

        writer.writerow([f"7.5(c) Top terms theo TF-IDF trong document #{DOC_INDEX}"])
        writer.writerow(["rank", "term", "tfidf"])
        for rank, (term, val) in enumerate(top_tfidf_terms, start=1):
            writer.writerow([rank, term, f"{val:.6f}"])
        writer.writerow([])

        writer.writerow(["So sánh 3 danh sách"])
        writer.writerow(["overlap_df_idf", "; ".join(sorted(overlap_df_idf)) or "(rỗng)"])
        writer.writerow(["overlap_df_tfidf", "; ".join(sorted(overlap_df_tfidf)) or "(rỗng)"])
        writer.writerow(["overlap_idf_tfidf", "; ".join(sorted(overlap_idf_tfidf)) or "(rỗng)"])
        writer.writerow([])

        writer.writerow(["Câu hỏi", "Trả lời"])
        writer.writerow([
            "Tại sao một document chỉ dùng một phần rất nhỏ vocabulary nhưng vector vẫn có chiều V?",
            "Vì tất cả document phải nằm chung một không gian đặc trưng (chung V chiều ứng với "
            "toàn bộ từ vựng của corpus) để so sánh được với nhau; các chiều không dùng tới thì bằng 0, "
            "khiến ma trận rất thưa (sparse).",
        ])
        writer.writerow([
            "Một term xuất hiện rất nhiều trong corpus có nhất thiết có TF-IDF cao không?",
            q1,
        ])
        writer.writerow([
            "Một term có IDF cao có nhất thiết có TF-IDF cao trong mọi document không?",
            q2,
        ])

    print(f"\nĐã ghi toàn bộ kết quả ra file: {RESULT_CSV}")
    print(f"\nTổng thời gian chạy pipeline: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()