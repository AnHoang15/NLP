"""
11. ĐÁNH GIÁ SEARCH ENGINE — Precision@K, Recall@K, Mean Reciprocal Rank (MRR)

11.1 Evaluation set: 8 query, mỗi query có relevance labels 

11.2 Precision@K = #relevant documents retrieved / K
11.3 Recall@K    = #relevant documents retrieved / #relevant documents
11.4 Mean Reciprocal Rank:
        Nếu document relevant đầu tiên của query q xuất hiện ở rank r_q:
        MRR = (1/|Q|) * sum_{q in Q} (1 / r_q)
"""

import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from search_engine import DocumentSearchEngine, load_documents, make_preview

DATA_PATH = "C:/HUS/NLP/lab01/c4-train.00000-of-01024-30K.json.gz"
N_DOCS = 30000
K = 5  # Precision@K, Recall@K dùng K = 5 theo đúng đề bài


# 11.1 EVALUATION SET — 8 query, relevance labels
EVALUATION_SET = [
    {
        "query": "chocolate cake recipe",
        "relevant": [22650, 15788, 3049, 764],
    },
    {
        "query": "yoga class benefits",
        "relevant": [7480, 7848, 19141, 16318],
    },
    {
        "query": "resume writing tips",
        "relevant": [5786, 8393, 16085, 2281, 27781],
    },
    {
        "query": "hotel room booking",
        "relevant": [24406, 15561, 28910, 20004, 19671],
    },
    {
        "query": "dog training tips",
        "relevant": [1987],
    },
    {
        "query": "car insurance quote",
        "relevant": [18966, 12563, 492, 16788, 2640, 29273, 1648],
    },
    {
        "query": "python programming tutorial",
        "relevant": [1814],
    },
    {
        "query": "wedding photography packages",
        "relevant": [10695, 24809],
    },
]


# 11.2 Precision@K
def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]
    n_relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return n_relevant_retrieved / k

# 11.3 Recall@K
def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    n_relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return n_relevant_retrieved / len(relevant_ids)


# 11.4 Mean Reciprocal Rank — cần rank của document relevant ĐẦU TIÊN

def first_relevant_rank(full_ranking, relevant_ids):
    """full_ranking: list doc_id đã sắp xếp theo similarity giảm dần.
    Trả về rank (1-based) của document relevant đầu tiên xuất hiện,
    hoặc None nếu không document relevant nào nằm trong full_ranking."""
    for rank, doc_id in enumerate(full_ranking, start=1):
        if doc_id in relevant_ids:
            return rank
    return None


def mean_reciprocal_rank(reciprocal_ranks):
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# MAIN
def main():
    print("Đang đọc dữ liệu và xây TF-IDF index (giống search_engine.py)...")
    documents = load_documents(DATA_PATH, N_DOCS)
    engine = DocumentSearchEngine(documents)
    print(f"-> Index sẵn sàng: {len(documents)} documents, "
          f"thời gian build = {engine.index_build_time:.2f}s\n")


    print(f"ĐÁNH GIÁ TRÊN {len(EVALUATION_SET)} QUERY (K = {K})")

    all_precision, all_recall, all_rr = [], [], []

    for item in EVALUATION_SET:
        query = item["query"]
        relevant_ids = set(item["relevant"])

        # Lấy top-K để tính Precision@K / Recall@K
        top_k_results = engine.search(query, top_k=K)
        retrieved_top_k = [r["doc_id"] for r in top_k_results]

        # Lấy FULL ranking (toàn bộ 30K, sắp xếp giảm dần) để tính MRR
        query_vector = engine.vectorizer.transform([query])
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(query_vector, engine.tfidf_matrix).flatten()
        full_ranking = sims.argsort()[::-1].tolist()

        p_at_k = precision_at_k(retrieved_top_k, relevant_ids, K)
        r_at_k = recall_at_k(retrieved_top_k, relevant_ids, K)
        rank_first_relevant = first_relevant_rank(full_ranking, relevant_ids)
        rr = 1 / rank_first_relevant if rank_first_relevant else 0.0

        all_precision.append(p_at_k)
        all_recall.append(r_at_k)
        all_rr.append(rr)

        print(f'\nQuery: "{query}"')
        print(f"  #relevant documents (đã biết) = {len(relevant_ids)}")
        print(f"  Top-{K} retrieved: {retrieved_top_k}")
        print(f"  Trong đó relevant: {[d for d in retrieved_top_k if d in relevant_ids]}")
        print(f"  Precision@{K} = {p_at_k:.4f}")
        print(f"  Recall@{K}    = {r_at_k:.4f}")
        print(f"  Rank của relevant document đầu tiên (r_q) = {rank_first_relevant}")
        print(f"  Reciprocal Rank (1/r_q) = {rr:.4f}")

    mean_p = sum(all_precision) / len(all_precision)
    mean_r = sum(all_recall) / len(all_recall)
    mrr = mean_reciprocal_rank(all_rr)

    
    print("KẾT QUẢ TỔNG HỢP (3 metric chính)")
    print(f"Mean Precision@{K} = {mean_p:.4f}")
    print(f"Mean Recall@{K}    = {mean_r:.4f}")
    print(f"MRR                = {mrr:.4f}")

    
    print("GIẢI THÍCH")
    print(
        f"""
Precision@{K} = #relevant documents retrieved / {K}
  -> Đo trong {K} kết quả TOP ĐẦU TIÊN trả về, bao nhiêu phần trăm THỰC SỰ liên quan.
     Ví dụ query "hotel room booking": cả {K}/{K} kết quả đều liên quan
     (Precision@{K} = 1.0) vì corpus có rất nhiều trang đặt phòng khách sạn tương tự nhau.
     Ngược lại "dog training tips" chỉ có đúng 1 document liên quan trong toàn corpus,
     nên Precision@{K} tối đa có thể đạt được chỉ là 1/{K} = {1/K:.2f} (không có cách
     nào retrieved đủ {K} document liên quan vì thực tế không tồn tại đủ {K} document đó).
     => Precision cao không phải lúc nào cũng do search tốt — đôi khi do CHỦ ĐỀ đó có
     nhiều tài liệu liên quan sẵn trong corpus (dễ trúng), không phải do thuật toán giỏi hơn.

Recall@{K} = #relevant documents retrieved / #relevant documents (tổng số relevant đã biết)
  -> Đo: trong SỐ LƯỢNG document liên quan mà ta biết có tồn tại, thuật toán tìm ra được
     bao nhiêu phần trăm trong top-{K}. Với query có ít relevant document (vd "python
     programming tutorial" chỉ có 1 relevant) thì Recall dễ đạt 1.0 nếu nó lọt vào top-{K}.
     Với query có nhiều relevant document hơn (vd "car insurance quote" có 7 relevant),
     Recall@{K} bị giới hạn tối đa ở {K}/7 = {K/7:.3f} vì top-{K} không đủ chỗ chứa hết
     7 document liên quan, dù thuật toán có tìm đúng cả {K} vị trí.
     => Recall thấp có thể chỉ vì K quá nhỏ so với số lượng relevant document thực có,
     không hẳn vì thuật toán tìm sai.

MRR = trung bình của 1/r_q trên tất cả query
  -> Đo tốc độ tìm ra document liên quan ĐẦU TIÊN (không quan tâm bao nhiêu document liên
     quan khác nằm sau đó). MRR = {mrr:.4f} nghĩa là trung bình, document liên quan đầu
     tiên xuất hiện ở vị trí khoảng 1/{mrr:.4f} ≈ {1/mrr if mrr > 0 else float('inf'):.1f}
     trong danh sách kết quả. MRR gần 1.0 => hầu hết các query đều tìm thấy kết quả đúng
     ngay ở rank 1; MRR càng nhỏ => người dùng phải cuộn xuống xa hơn mới thấy kết quả liên quan.
     => MRR phù hợp để đánh giá trải nghiệm "câu trả lời đầu tiên có đúng không" (giống công cụ
     tìm kiếm thực tế), trong khi Precision/Recall@{K} đánh giá TOÀN BỘ chất lượng của {K} kết
     quả đầu, không chỉ mỗi kết quả đầu tiên.
"""
    )


if __name__ == "__main__":
    main()