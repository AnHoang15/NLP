"""
12. CASE ANALYSIS — Phân tích chi tiết kết quả search engine

Chọn 2 query có kết quả TỐT + 2 query có kết quả KÉM (dựa trên Precision@5 / Recall@5 / RR đã đo trong evaluate_search.py).

Với mỗi query, phân tích:
    - Query
    - Expected relevant documents
    - Retrieved documents (top-5)
    - Analysis: 5 câu hỏi (vì sao đứng đầu, từ nào đóng góp similarity,
      lexical overlap, có bỏ sót relevant document nào không, root cause)

Cuối cùng:  1 "failure case quan trọng nhất".
"""

import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.metrics.pairwise import cosine_similarity

from search_engine import DocumentSearchEngine, load_documents, make_preview
from evaluate_search import (
    EVALUATION_SET,
    precision_at_k,
    recall_at_k,
    first_relevant_rank,
)

DATA_PATH = "C:/HUS/NLP/lab01/c4-train.00000-of-01024-30K.json.gz"
N_DOCS = 30000
K = 5

# Chọn thủ công dựa trên kết quả đã đo ở evaluate_search.py:

GOOD_QUERIES = ["car insurance quote", "resume writing tips"]
BAD_QUERIES = ["dog training tips", "wedding photography packages"]

# Failure case được đào sâu nhất ở cuối bài
DEEP_DIVE_QUERY = "dog training tips"


def get_eval_item(query):
    for item in EVALUATION_SET:
        if item["query"] == query:
            return item
    raise ValueError(f"Không tìm thấy query '{query}' trong EVALUATION_SET")


def analyze_query(engine, documents, query, relevant_ids, vocab, idf):
    """Trả về dict chứa mọi thông tin cần để in Analysis cho 1 query."""
    query_vector = engine.vectorizer.transform([query])
    sims = cosine_similarity(query_vector, engine.tfidf_matrix).flatten()
    full_ranking = sims.argsort()[::-1]

    top5_ids = full_ranking[:K].tolist()
    top5_sims = [float(sims[i]) for i in top5_ids]

    # từ trong query sau khi qua đúng bộ tiền xử lý của vectorizer (analyzer)
    analyzer = engine.vectorizer.build_analyzer()
    query_terms = analyzer(query)
    query_term_set = set(query_terms)

    top1_id = top5_ids[0]
    top1_vec = engine.tfidf_matrix[top1_id].toarray().flatten()
    query_vec_dense = query_vector.toarray().flatten()

    # đóng góp từng từ vào similarity = tfidf_query[t] * tfidf_doc[t]
    contributions = query_vec_dense * top1_vec
    nonzero_idx = np.nonzero(contributions)[0]
    contrib_terms = sorted(
        [(vocab[i], float(contributions[i])) for i in nonzero_idx],
        key=lambda x: x[1],
        reverse=True,
    )

    # lexical overlap giữa query và document đứng đầu
    top1_terms = set(analyzer(documents[top1_id]["text"]))
    overlap = query_term_set & top1_terms

    # relevant document nào KHÔNG nằm trong top-5, và rank thật của nó là bao nhiêu
    rank_lookup = {doc_id: rank + 1 for rank, doc_id in enumerate(full_ranking)}
    missed = [
        {"doc_id": rid, "rank": rank_lookup.get(rid), "similarity": float(sims[rid])}
        for rid in relevant_ids
        if rid not in top5_ids
    ]

    return {
        "query": query,
        "query_terms": query_terms,
        "top5_ids": top5_ids,
        "top5_sims": top5_sims,
        "contrib_terms": contrib_terms,
        "overlap": overlap,
        "missed": missed,
        "full_ranking": full_ranking,
    }


def print_case(engine, documents, query, label):
    item = get_eval_item(query)
    relevant_ids = set(item["relevant"])
    vocab = engine.vectorizer.get_feature_names_out()
    idf = engine.vectorizer.idf_

    info = analyze_query(engine, documents, query, relevant_ids, vocab, idf)

    p5 = precision_at_k(info["top5_ids"], relevant_ids, K)
    r5 = recall_at_k(info["top5_ids"], relevant_ids, K)
    r_q = first_relevant_rank(info["full_ranking"].tolist(), relevant_ids)
    rr = 1 / r_q if r_q else 0.0

    print(f"[{label}]  Query: \"{query}\"   (P@5={p5:.2f}, R@5={r5:.2f}, RR={rr:.2f})")

    print("\nQuery (sau tiền xử lý):", info["query_terms"])

    print("\nExpected relevant documents:")
    for rid in sorted(relevant_ids):
        preview = make_preview(documents[rid]["text"], 90)
        print(f"  D{rid:<6} {preview}")

    print("\nRetrieved documents (top-5):")
    for rank, (doc_id, sim) in enumerate(zip(info["top5_ids"], info["top5_sims"]), start=1):
        mark = " relevant" if doc_id in relevant_ids else " not relevant"
        preview = make_preview(documents[doc_id]["text"], 90)
        print(f"  {rank}. D{doc_id:<6} sim={sim:.4f}  [{mark}]  {preview}")

    print("\nAnalysis:")
    top1_id = info["top5_ids"][0]
    print(f"  1) Vì sao document D{top1_id} đứng đầu?")
    if info["contrib_terms"]:
        top_reasons = ", ".join(f"'{t}'" for t, _ in info["contrib_terms"][:5])
        print(f"     -> Vì có nhiều từ trong query trùng với D{top1_id} và những từ đó có "
              f"trọng số TF-IDF cao (đặc biệt: {top_reasons}), khiến tích vô hướng "
              f"(dot product) giữa 2 vector lớn nhất trong toàn corpus.")
    else:
        print("     -> Không có từ chung nào giữa query và document -> điểm similarity đến "
              "từ nguyên nhân khác (hiếm gặp, cần kiểm tra lại).")

    print("\n  2) Những từ nào đóng góp nhiều nhất vào similarity (query_tfidf x doc_tfidf)?")
    for term, val in info["contrib_terms"][:8]:
        print(f"     {term:<15} đóng góp = {val:.4f}")

    print(f"\n  3) Có lexical overlap giữa query và D{top1_id} không?")
    if info["overlap"]:
        print(f"     -> CÓ, {len(info['overlap'])}/{len(info['query_terms'])} từ trong query "
              f"xuất hiện trực tiếp trong document: {sorted(info['overlap'])}")
    else:
        print("     -> KHÔNG có từ nào trùng trực tiếp — similarity (nếu > 0) không thể đến "
              "từ TF-IDF/cosine dạng bag-of-words (mô hình này không hiểu từ đồng nghĩa).")

    print("\n  4) Có relevant document nào bị bỏ sót trong top-5 không?")
    if info["missed"]:
        for m in info["missed"]:
            print(f"     -> D{m['doc_id']} KHÔNG lọt vào top-5, thực tế xếp hạng #{m['rank']} "
                  f"(similarity={m['similarity']:.4f})")
    else:
        print("     -> KHÔNG, mọi relevant document đã biết đều nằm trong top-5.")

    print("\n  5) Nếu có vấn đề, nguyên nhân đến từ đâu (preprocessing/TF/IDF/vocabulary/"
          "lexical matching/khác)?")
    if p5 < 0.5 or (info["missed"]):
        print(f"     -> Nguyên nhân chính là LEXICAL MATCHING: TF-IDF + cosine similarity chỉ "
              f"khớp được các từ XUẤT HIỆN Y NGUYÊN trong cả query lẫn document. Các document "
              f"relevant nhưng dùng từ vựng khác (đồng nghĩa, cách diễn đạt khác) sẽ bị chấm "
              f"điểm similarity thấp dù ngữ nghĩa đúng, trong khi document dùng đúng 1-2 từ "
              f"khớp query nhưng lặp lại nhiều lần (TF cao) có thể bị xếp hạng cao hơn dù xa "
              f"chủ đề thực sự.")
    else:
        print(f"     -> Không có vấn đề đáng kể: query và top kết quả có lexical overlap tốt, "
              f"IDF phân biệt tốt các từ đặc trưng, TF phản ánh đúng mức độ liên quan.")

    return info


def deep_dive(engine, documents, query):
    item = get_eval_item(query)
    relevant_ids = set(item["relevant"])
    vocab = engine.vectorizer.get_feature_names_out()
    idf = engine.vectorizer.idf_
    info = analyze_query(engine, documents, query, relevant_ids, vocab, idf)

    top1_id = info["top5_ids"][0]
    relevant_ranked = sorted(
        relevant_ids, key=lambda rid: int(np.where(info["full_ranking"] == rid)[0][0])
    )
    true_doc_id = relevant_ranked[0]
    true_rank = int(np.where(info["full_ranking"] == true_doc_id)[0][0]) + 1

    analyzer = engine.vectorizer.build_analyzer()
    query_terms_ordered = analyzer(query)
    query_terms = set(query_terms_ordered)
    top1_all_terms = analyzer(documents[top1_id]["text"])
    true_all_terms = analyzer(documents[true_doc_id]["text"])
    top1_terms = set(top1_all_terms)
    true_terms = set(true_all_terms)

    overlap_top1 = query_terms & top1_terms
    overlap_true = query_terms & true_terms

    from collections import Counter
    count_top1 = Counter(top1_all_terms)
    count_true = Counter(true_all_terms)

    query_vec_dense = engine.vectorizer.transform([query]).toarray().flatten()
    top1_vec = engine.tfidf_matrix[top1_id].toarray().flatten()
    true_vec = engine.tfidf_matrix[true_doc_id].toarray().flatten()
    sim_top1 = float(cosine_similarity(query_vec_dense.reshape(1, -1), top1_vec.reshape(1, -1))[0][0])
    sim_true = float(cosine_similarity(query_vec_dense.reshape(1, -1), true_vec.reshape(1, -1))[0][0])


    print("FAILURE CASE QUAN TRỌNG NHẤT")
    print(f'\nQuery:\n  "{query}"  -> sau tiền xử lý: {query_terms_ordered}')
    print(f"\nRelevant document (đúng theo nhãn):\n  D{true_doc_id}: "
          f'"{make_preview(documents[true_doc_id]["text"], 200)}"')
    print(f"  -> Xếp hạng THỰC TẾ của document này: #{true_rank} (similarity = {sim_true:.4f})")

    print(f"\nDocument bị xếp hạng #1 THAY VÌ document đúng:\n  D{top1_id}: "
          f'"{make_preview(documents[top1_id]["text"], 200)}"')
    print(f"  -> similarity = {sim_top1:.4f}")

    print(f"\nSo sánh lexical overlap (từ nào trong query xuất hiện trong document):")
    print(f"  Overlap với D{top1_id} (rank #1, SAI)   : {sorted(overlap_top1)}")
    print(f"  Overlap với D{true_doc_id} (relevant thật): {sorted(overlap_true)}")

    if overlap_top1 == overlap_true:
        print(
            f"\n  -> LƯU Ý QUAN TRỌNG: cả 2 document có CÙNG một tập từ overlap với query "
            f"({sorted(overlap_top1)}) — nghĩa là đây KHÔNG phải lỗi do thiếu từ khớp (không "
            f"giống trường hợp kinh điển 'heart attack treatment' vs 'myocardial infarction "
            f"therapy' mà đề bài nêu, nơi 2 câu có nghĩa giống nhau nhưng không có từ nào trùng "
            f"chữ viết). Ở đây nguyên nhân nằm sâu hơn: TẦN SUẤT của từng từ."
        )

    print(f"\nTần suất các biến thể liên quan tới từ 'training' trong mỗi document:")
    for term in vocab:
        pass  # (chỉ để rõ ý — không dùng)
    variant_top1 = {w: c for w, c in count_top1.items() if "train" in w}
    variant_true = {w: c for w, c in count_true.items() if "train" in w}
    print(f"  D{top1_id} (rank #1, SAI)    : {variant_top1}  "
          f"-> tổng nhắc tới ý niệm 'training' = {sum(variant_top1.values())}")
    print(f"  D{true_doc_id} (relevant thật): {variant_true}  "
          f"-> tổng nhắc tới ý niệm 'training' = {sum(variant_true.values())}")

    print(
        f"\nGIẢI THÍCH:\n"
        f"  Cả 2 document đều nói về 'dog' và 'training' — lexical overlap với query giống hệt "
        f"nhau. Nhưng khi nhìn vào SỐ LẦN xuất hiện của đúng token 'training' (token mà query "
        f"dùng): D{top1_id} có {count_top1.get('training', 0)} lần, còn D{true_doc_id} chỉ có "
        f"{count_true.get('training', 0)} lần — THẤP HƠN, dù D{true_doc_id} mới thực sự là "
        f"document chuyên về dịch vụ huấn luyện chó.\n"
        f"  Lý do: D{true_doc_id} diễn đạt cùng một khái niệm bằng NHIỀU BIẾN THỂ từ khác nhau — "
        f"{variant_true} — trong khi vectorizer (không có bước stemming/lemmatization) coi "
        f"'training', 'trainer', 'trainers' là 3 TOKEN HOÀN TOÀN KHÁC NHAU trong vocabulary. "
        f"Tổng số lần nhắc đến khái niệm 'huấn luyện' ở D{true_doc_id} thực ra CAO HƠN hoặc "
        f"tương đương D{top1_id} ({sum(variant_true.values())} so với "
        f"{sum(variant_top1.values())}), nhưng vì bị CHIA NHỎ ra nhiều token khác nhau, mỗi "
        f"token nhận trọng số TF thấp hơn, khiến điểm TF-IDF cho riêng token 'training' — token "
        f"duy nhất mà query thực sự dùng — bị đánh giá thấp hơn D{top1_id} (nơi tác giả dùng "
        f"nhất quán một từ 'training' xuyên suốt).\n\n"
        f"  ROOT CAUSE = PREPROCESSING (thiếu stemming/lemmatization), không phải do thiếu "
        f"lexical overlap hay do IDF/vocabulary. Đây là một dạng lỗi khác — nhưng cùng họ với "
        f"ví dụ 'heart attack treatment' vs 'myocardial infarction therapy' mà đề bài nêu: cả "
        f"hai đều cho thấy TF-IDF bag-of-words CHỈ đếm được từ trùng khớp Y NGUYÊN về mặt chữ "
        f"viết, không hiểu được rằng 'training', 'trainer', 'trainers' hay 'heart attack',"
        f"'myocardial infarction' đang cùng nói về một khái niệm. Khắc phục: thêm bước "
        f"stemming/lemmatization (gộp 'training'/'trainer'/'trainers' về 1 gốc từ 'train') sẽ "
        f"giải quyết được lớp lỗi này; còn lỗi do khác biệt hoàn toàn về mặt từ vựng (đồng nghĩa "
        f"không cùng gốc như ví dụ y khoa) thì cần đến word embeddings / mô hình ngôn ngữ mới "
        f"xử lý được."
    )


def main():
    print("Đang đọc dữ liệu và xây TF-IDF index...")
    documents = load_documents(DATA_PATH, N_DOCS)
    engine = DocumentSearchEngine(documents)
    print(f"-> Index sẵn sàng: {len(documents)} documents.\n")

   
    print("PHẦN 1 — 2 QUERY CÓ KẾT QUẢ TỐT")
    
    for q in GOOD_QUERIES:
        print_case(engine, documents, q, "TỐT")

    print("PHẦN 2 — 2 QUERY CÓ KẾT QUẢ KÉM")
    
    for q in BAD_QUERIES:
        print_case(engine, documents, q, "KÉM")

    deep_dive(engine, documents, DEEP_DIVE_QUERY)


if __name__ == "__main__":
    main()