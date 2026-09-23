"""
10. PART G — APPLICATION: BUILD A DOCUMENT SEARCH ENGINE

10.1 Bài toán: xây hệ thống tìm kiếm đơn giản trên corpus 30K documents.
     Input : User query (chuỗi văn bản)
     Output: Top-K relevant documents

10.2 Pipeline:
     30K Documents -> TF-IDF Index -> User Query -> TF-IDF Query Vector -> Cosine Similarity -> Ranking -> Top-K Documents

10.3 Query examples: "medical image classification", "transformer language model", "deep learning healthcare", "natural language processing"

10.4 Kết quả hiển thị mỗi query: bảng Rank | Document ID | Similarity | Document preview
"""

import sys
import gzip
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "C:/HUS/NLP/lab01/c4-train.00000-of-01024-30K.json.gz"
N_DOCS = 30000       
TOP_K = 5             
PREVIEW_LEN = 100    

# 10.3 Query examples
EXAMPLE_QUERIES = [
    "medical image classification",
    "transformer language model",
    "deep learning healthcare",
    "natural language processing",
]

# ĐỌC DỮ LIỆU (giữ cả text, url, timestamp cho phần preview/metadata)
def load_documents(path, n_docs):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            obj = json.loads(line)
            docs.append(obj)
    return docs


def make_preview(text, length=PREVIEW_LEN):
    text = " ".join(text.split())  
    if len(text) <= length:
        return text
    return text[:length].rstrip() + "..."


# 10.2 PIPELINE — DOCUMENT SEARCH ENGINE

class DocumentSearchEngine:
    """
    30K Documents -> TF-IDF Index -> User Query -> TF-IDF Query Vector -> Cosine Similarity -> Ranking -> Top-K Documents
    """

    def __init__(self, documents):
        self.documents = documents         
        self.texts = [d["text"] for d in documents]

        t0 = time.time()
        # Bước 1: xây TF-IDF Index cho toàn bộ 30K documents
        self.vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.texts)
        self.index_build_time = time.time() - t0

    def search(self, query, top_k=TOP_K):
        # Bước 2: User Query -> TF-IDF Query Vector (dùng chung vocabulary/idf đã fit)
        query_vector = self.vectorizer.transform([query])

        # Bước 3: Cosine Similarity giữa query và toàn bộ documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Bước 4: Ranking -> Top-K Documents
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                {
                    "rank": rank,
                    "doc_id": int(idx),
                    "similarity": float(similarities[idx]),
                    "preview": make_preview(self.texts[idx]),
                    "url": self.documents[idx].get("url", ""),
                }
            )
        return results


# 10.4 IN KẾT QUẢ THEO ĐÚNG FORMAT: Rank | Document ID | Similarity | Preview
def print_results_table(query, results):
    print(f'\nQuery: "{query}"')
    print(f"{'Rank':<6}{'Document ID':<14}{'Similarity':<12}{'Document preview'}")
 
    for r in results:
        print(f"{r['rank']:<6}{r['doc_id']:<14}{r['similarity']:<12.4f}{r['preview']}")
  
# MAIN
def main():
    print("Đang đọc dữ liệu...")
    documents = load_documents(DATA_PATH, N_DOCS)
    print(f"-> Đã đọc {len(documents)} documents.\n")

    print("Đang xây dựng TF-IDF index...")
    engine = DocumentSearchEngine(documents)
    print(
        f"-> Index xây xong: vocabulary size = {len(engine.vectorizer.get_feature_names_out())}, "
        f"matrix shape = {engine.tfidf_matrix.shape}, "
        f"thời gian = {engine.index_build_time:.2f}s"
    )

    # 10.3 Chạy thử với các query mẫu trong đề bài

    print("KẾT QUẢ TÌM KIẾM VỚI CÁC QUERY MẪU (10.3)")
    for query in EXAMPLE_QUERIES:
        t0 = time.time()
        results = engine.search(query, top_k=TOP_K)
        elapsed_ms = (time.time() - t0) * 1000
        print_results_table(query, results)
        print(f"(thời gian truy vấn: {elapsed_ms:.2f} ms)")

    # Chế độ nhập query tuỳ ý (tuỳ chọn) — gõ 'exit' để thoát
    print("NHẬP QUERY TÙY Ý (gõ 'exit' để thoát)")
    while True:
        try:
            user_query = input("\nQuery > ").strip()
        except EOFError:
            break
        if not user_query or user_query.lower() == "exit":
            break
        t0 = time.time()
        results = engine.search(user_query, top_k=TOP_K)
        elapsed_ms = (time.time() - t0) * 1000
        print_results_table(user_query, results)
        print(f"(thời gian truy vấn: {elapsed_ms:.2f} ms)")


if __name__ == "__main__":
    main()