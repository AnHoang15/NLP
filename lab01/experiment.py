"""
9.1-9.5 THỰC NGHIỆM SO SÁNH 3 PIPELINE TIỀN XỬ LÝ VĂN BẢN

9.1 Pipeline A - Minimal:    Raw text -> Lowercase -> Tokenization
9.2 Pipeline B - Normalized: Raw text -> Lowercase -> Punctuation normalization -> Tokenization -> Stopword handling
9.3 Pipeline C - Extended:   Raw text -> Normalization -> Subword tokenization (BPE tự viết)
9.4 So sánh: Vocabulary size, Average tokens/document, Matrix sparsity,OOV rate, Search performance
9.5 Câu hỏi phân tích: trả lời bằng kết quả thực nghiệm đo được ở trên.
"""

import sys
import re
import gzip
import json
import time
import random
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "C:/HUS/NLP/lab01/c4-train.00000-of-01024-30K.json.gz"
N_DOCS = 5000              # số văn bản dùng cho thực nghiệm (tăng lên nếu máy khoẻ)
TRAIN_RATIO = 0.8          # tỉ lệ train/test để đo OOV rate
RANDOM_SEED = 42

BPE_NUM_MERGES = 300       # số lần merge khi huấn luyện BPE (pipeline C)
BPE_TRAIN_TOP_WORDS = 5000 # chỉ dùng N từ phổ biến nhất để học merges (cho nhanh)

N_QUERY_DOCS = 30          # số document dùng để tạo truy vấn thử nghiệm search
QUERY_LEN = 5              # số từ khoá lấy từ mỗi document để làm query
TOP_KS = [1, 5, 10]        # Hit@K sẽ tính cho các K này

random.seed(RANDOM_SEED)


# ĐỌC DỮ LIỆU
def load_documents(path, n_docs):
    docs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_docs:
                break
            docs.append(json.loads(line)["text"])
    return docs


WORD_RE = re.compile(r"[a-zA-ZÀ-ỹ]+")


def tokenize_words(text):
    """Tách văn bản thành các từ (chuỗi ký tự chữ cái liên tiếp)."""
    return WORD_RE.findall(text)


PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'",  
    "\u201c": '"', "\u201d": '"',  
    "\u2013": "-", "\u2014": "-",  
    "\u2026": "...",                
}


def normalize_punctuation(text):
    for k, v in PUNCT_MAP.items():
        text = text.replace(k, v)
    return text


STOPWORDS = set(
    """a an the and or but if then else for of in on at to from by with without
    into onto is are was were be been being do does did have has had this that
    these those it its i you he she we they them his her their our your my me
    him us not no so as than too very can will would should could may might
    must shall""".split()
)


# 9.1 PIPELINE A — MINIMAL: Raw text -> Lowercase -> Tokenization

def pipeline_A(text):
    text = text.lower()
    return tokenize_words(text)


# 9.2 PIPELINE B — NORMALIZED: Raw text -> Lowercase -> Punctuation normalization -> Tokenization -> Stopword handling

def pipeline_B(text):
    text = text.lower()
    text = normalize_punctuation(text)
    tokens = tokenize_words(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


# 9.3 PIPELINE C — EXTENDED: Raw text -> Normalization -> Subword tokenization. Cài đặt thuật toán BPE (Byte-Pair Encoding) từ đầu,

class SimpleBPE:
    """BPE tối giản: học các cặp ký tự được ghép lại nhiều nhất, lặp lại
    `num_merges` lần trên tập từ (word -> tần suất) của corpus."""

    END = "</w>"  # đánh dấu kết thúc từ, giúp phân biệt hậu tố với tiền tố

    def __init__(self, num_merges=300):
        self.num_merges = num_merges
        self.merges = []          
        self._cache = {}         

    def _word_to_symbols(self, word):
        return list(word) + [self.END]

    def train(self, word_freq):
        # word_freq: dict {word: số lần xuất hiện trong tập train}
        vocab = {
            tuple(self._word_to_symbols(w)): c for w, c in word_freq.items()
        }
        for _ in range(self.num_merges):
            pair_counts = defaultdict(int)
            for symbols, freq in vocab.items():
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  

            self.merges.append(best_pair)
            a, b = best_pair
            merged_symbol = a + b
            new_vocab = {}
            for symbols, freq in vocab.items():
                new_symbols, i = [], 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new_symbols.append(merged_symbol)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                key = tuple(new_symbols)
                new_vocab[key] = new_vocab.get(key, 0) + freq
            vocab = new_vocab

    def encode_word(self, word):
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = self._word_to_symbols(word)
        for a, b in self.merges:
            if len(symbols) == 1:
                break
            merged_symbol = a + b
            new_symbols, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(merged_symbol)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        self._cache[word] = symbols
        return symbols


def normalize_for_C(text):
    text = text.lower()
    text = normalize_punctuation(text)
    return text


def make_pipeline_C(bpe):
    def pipeline_C(text):
        text = normalize_for_C(text)
        words = tokenize_words(text)
        subwords = []
        for w in words:
            subwords.extend(bpe.encode_word(w))
        return subwords
    return pipeline_C

def build_count_matrix(documents, tokenizer_fn):
    vectorizer = CountVectorizer(tokenizer=tokenizer_fn, token_pattern=None)
    matrix = vectorizer.fit_transform(documents)
    return matrix, vectorizer


def compute_sparsity(matrix):
    n, v = matrix.shape
    return 1 - matrix.nnz / (n * v)


def compute_oov_rate(train_vocab, test_token_lists):
    total, oov = 0, 0
    for tokens in test_token_lists:
        for t in tokens:
            total += 1
            if t not in train_vocab:
                oov += 1
    return oov / total if total > 0 else 0.0


def build_query_from_doc(text, query_len=QUERY_LEN):
    """Query 'giả lập người dùng': lấy N từ khoá xuất hiện nhiều nhất
    trong document, sau khi bỏ stopword — độc lập với pipeline đang test,
    để đánh giá công bằng khả năng search của từng pipeline."""
    tokens = pipeline_B(text)  # dùng chung 1 cách trích từ khoá cho mọi pipeline
    freq = Counter(tokens)
    most_common = [w for w, _ in freq.most_common(query_len)]
    return " ".join(most_common)


def evaluate_search(documents, tokenizer_fn, query_doc_indices, top_ks=TOP_KS):
    """Xây TF-IDF cho toàn bộ corpus với 1 pipeline, sau đó với mỗi document
    trong query_doc_indices: tạo query từ chính document đó, tìm kiếm trên
    toàn bộ corpus bằng cosine similarity, và xem document gốc xếp hạng bao nhiêu."""
    vectorizer = TfidfVectorizer(tokenizer=tokenizer_fn, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(documents)

    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in top_ks}
    latencies = []

    for idx in query_doc_indices:
        query_text = build_query_from_doc(documents[idx])
        if not query_text.strip():
            continue
        t0 = time.time()
        query_vec = vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
        latencies.append(time.time() - t0)

        ranking = np.argsort(sims)[::-1]
        rank_of_true_doc = int(np.where(ranking == idx)[0][0]) + 1  # 1-based
        reciprocal_ranks.append(1 / rank_of_true_doc)
        for k in top_ks:
            if rank_of_true_doc <= k:
                hits_at_k[k] += 1

    n = len(reciprocal_ranks)
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    hit_rates = {k: hits_at_k[k] / n if n else 0.0 for k in top_ks}
    avg_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
    vocab_size = len(vectorizer.get_feature_names_out())
    return {
        "mrr": mrr,
        "hit_rates": hit_rates,
        "avg_latency_ms": avg_latency_ms,
        "vocab_size": vocab_size,
        "n_queries": n,
    }

# MAIN

def main():
    t_start = time.time()
    print("Đang đọc dữ liệu...")
    documents = load_documents(DATA_PATH, N_DOCS)
    print(f"-> Đã đọc {len(documents)} văn bản.\n")

    # Train/test split (dùng để đo OOV rate)
    indices = list(range(len(documents)))
    random.shuffle(indices)
    n_train = int(len(indices) * TRAIN_RATIO)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    train_docs = [documents[i] for i in train_idx]
    test_docs = [documents[i] for i in test_idx]
    print(f"Train/Test split: {len(train_docs)} train / {len(test_docs)} test documents.\n")

    # Huấn luyện BPE cho Pipeline C trên tập train
    print("Đang huấn luyện BPE cho Pipeline C...")
    t0 = time.time()
    word_freq = Counter()
    for doc in train_docs:
        word_freq.update(tokenize_words(normalize_for_C(doc)))
    top_words = dict(word_freq.most_common(BPE_TRAIN_TOP_WORDS))

    bpe = SimpleBPE(num_merges=BPE_NUM_MERGES)
    bpe.train(top_words)
    pipeline_C = make_pipeline_C(bpe)
    print(f"-> Học được {len(bpe.merges)} merge rules trên "
          f"{len(top_words)} từ phổ biến nhất, thời gian = {time.time() - t0:.2f}s\n")

    pipelines = {
        "A - Minimal": pipeline_A,
        "B - Normalized": pipeline_B,
        "C - Extended": pipeline_C,
    }


    # 9.4 SO SÁNH: Vocabulary size, Average tokens/document, Matrix sparsity, OOV rate
   
    print("9.4 SO SÁNH CÁC PIPELINE (trên toàn bộ", len(documents), "documents)")

    results = {}
    for name, fn in pipelines.items():
        t0 = time.time()

        # Tokenize toàn bộ documents (dùng lại cho vocab size / avg tokens / OOV)
        all_tokens = [fn(doc) for doc in documents]
        vocab = set()
        total_tokens = 0
        for toks in all_tokens:
            vocab.update(toks)
            total_tokens += len(toks)
        vocab_size = len(vocab)
        avg_tokens = total_tokens / len(documents)

        # Matrix + sparsity (dùng CountVectorizer cho nhanh, tokenizer riêng từng pipeline)
        matrix, _ = build_count_matrix(documents, fn)
        sparsity = compute_sparsity(matrix)

        # OOV rate: vocab học từ train, đo trên test
        train_tokens = [fn(d) for d in train_docs]
        test_tokens = [fn(d) for d in test_docs]
        train_vocab = set(t for toks in train_tokens for t in toks)
        oov_rate = compute_oov_rate(train_vocab, test_tokens)

        elapsed = time.time() - t0
        results[name] = {
            "vocab_size": vocab_size,
            "avg_tokens": avg_tokens,
            "matrix_shape": matrix.shape,
            "sparsity": sparsity,
            "oov_rate": oov_rate,
            "time_s": elapsed,
        }
        print(f"\n[{name}] (thời gian tính = {elapsed:.2f}s)")
        print(f"  Vocabulary size       = {vocab_size}")
        print(f"  Average tokens/doc    = {avg_tokens:.2f}")
        print(f"  Matrix shape          = {matrix.shape}")
        print(f"  Matrix sparsity       = {sparsity:.4%}")
        print(f"  OOV rate (train->test)= {oov_rate:.4%}")


    # Search performance

    print("9.4 SEARCH PERFORMANCE")
    query_doc_indices = random.sample(range(len(documents)), min(N_QUERY_DOCS, len(documents)))

    for name, fn in pipelines.items():
        t0 = time.time()
        search_result = evaluate_search(documents, fn, query_doc_indices)
        elapsed = time.time() - t0
        results[name]["search"] = search_result
        results[name]["search_build_time_s"] = elapsed

        hit_str = ", ".join(f"Hit@{k}={v:.1%}" for k, v in search_result["hit_rates"].items())
        print(f"\n[{name}] (build index + chạy {search_result['n_queries']} query, "
              f"tổng thời gian = {elapsed:.2f}s)")
        print(f"  MRR                = {search_result['mrr']:.4f}")
        print(f"  {hit_str}")
        print(f"  Query latency TB   = {search_result['avg_latency_ms']:.2f} ms/query")


    # Bảng tổng hợp
    print("BẢNG TỔNG HỢP")
    header = f"{'Metric':<26}" + "".join(f"{name:<20}" for name in pipelines)
    print(header)
    print("-" * len(header))
    rows = [
        ("Vocabulary size", lambda r: f"{r['vocab_size']}"),
        ("Avg tokens/doc", lambda r: f"{r['avg_tokens']:.2f}"),
        ("Matrix sparsity", lambda r: f"{r['sparsity']:.4%}"),
        ("OOV rate", lambda r: f"{r['oov_rate']:.4%}"),
        ("Search MRR", lambda r: f"{r['search']['mrr']:.4f}"),
        ("Search Hit@5", lambda r: f"{r['search']['hit_rates'][5]:.1%}"),
    ]
    for label, fn_fmt in rows:
        line = f"{label:<26}" + "".join(f"{fn_fmt(results[name]):<20}" for name in pipelines)
        print(line)

    # 9.5 CÂU HỎI PHÂN TÍCH — trả lời bằng số liệu thực nghiệm ở trên
    A, B, C = results["A - Minimal"], results["B - Normalized"], results["C - Extended"]
    print("9.5 CÂU HỎI PHÂN TÍCH (trả lời bằng kết quả thực nghiệm)")

    print(
        f"\n1) Lowercasing làm thay đổi vocabulary như thế nào?\n"
        f"   -> Không đo trực tiếp trong thực nghiệm này (cả 3 pipeline đều lowercase), "
        f"nhưng có thể suy luận gián tiếp: nếu KHÔNG lowercase, mỗi biến thể viết hoa/thường "
        f"của cùng 1 từ (vd 'The', 'the', 'THE') sẽ bị tách thành các token khác nhau trong "
        f"vocabulary, làm vocabulary phình to giả tạo mà không thêm thông tin. Lowercasing "
        f"gộp các biến thể này lại, giảm vocabulary size mà (thường) không mất ý nghĩa "
        f"(trừ trường hợp cần phân biệt tên riêng, ví dụ 'Apple' công ty vs 'apple' quả táo)."
    )

    print(
        f"\n2) Stopword removal có luôn cải thiện representation không?\n"
        f"   -> KHÔNG LUÔN LUÔN. Thực nghiệm cho thấy Pipeline B (có loại stopword) có "
        f"vocabulary size = {B['vocab_size']} so với Pipeline A = {A['vocab_size']} "
        f"(giảm {A['vocab_size'] - B['vocab_size']} từ), matrix sparsity "
        f"{B['sparsity']:.4%} so với {A['sparsity']:.4%}, và search MRR = "
        f"{B['search']['mrr']:.4f} so với A = {A['search']['mrr']:.4f}. "
        f"Nếu MRR của B {'cao hơn' if B['search']['mrr'] > A['search']['mrr'] else 'thấp hơn hoặc bằng'} "
        f"A, chứng tỏ loại stopword {'giúp' if B['search']['mrr'] > A['search']['mrr'] else 'không nhất thiết cải thiện'} "
        f"chất lượng tìm kiếm trong lần chạy này — vì stopword vừa làm giảm nhiễu (điểm cộng), "
        f"vừa có thể xoá mất một phần ngữ cảnh cú pháp/collocation (điểm trừ), tuỳ tác vụ mà "
        f"lợi ích khác nhau."
    )

    print(
        f"\n3) Việc loại punctuation có thể làm mất thông tin gì?\n"
        f"   -> Loại punctuation làm mất: (a) ranh giới câu (dấu chấm, chấm hỏi) -> khó tách "
        f"câu cho các tác vụ cần cấp độ câu; (b) sắc thái/ý nghĩa do dấu câu tạo ra (vd dấu "
        f"chấm than thể hiện cảm xúc mạnh, dấu ngoặc kép đánh dấu trích dẫn); (c) các từ viết "
        f"tắt có dấu nháy đơn bị tách sai nếu không xử lý cẩn thận (vd \"don't\" có thể bị tách "
        f"thành 'don' và 't' nếu tokenizer coi nháy đơn là ký tự phân cách); (d) số liệu có dấu "
        f"phẩy/chấm (vd '1,000.50') bị vỡ thành các mảnh vô nghĩa. Cả 3 pipeline ở đây đều bỏ "
        f"toàn bộ ký tự không phải chữ cái nên đều mất các thông tin này ở mức độ khác nhau; "
        f"Pipeline B có bước 'punctuation normalization' trước tokenization giúp chuẩn hoá các "
        f"biến thể dấu câu (nháy cong, gạch ngang dài) về 1 dạng thống nhất trước khi loại bỏ, "
        f"nên tránh được hiện tượng cùng 1 từ bị tách thành 2 token khác nhau chỉ vì kiểu dấu "
        f"nháy khác nhau."
    )

    smallest_vocab_name = min(results, key=lambda k: results[k]["vocab_size"])
    highest_sparsity_name = max(results, key=lambda k: results[k]["sparsity"])
    print(
        f"\n4) Pipeline nào tạo ra sparse matrix nhiều nhất (sparsity cao nhất)?\n"
        f"   -> Thực nghiệm: {highest_sparsity_name} có sparsity cao nhất "
        f"({results[highest_sparsity_name]['sparsity']:.4%}), vì vocabulary lớn "
        f"({results[highest_sparsity_name]['vocab_size']} từ) trong khi mỗi document chỉ dùng "
        f"một phần rất nhỏ trong số đó. Ngược lại {smallest_vocab_name} có vocabulary nhỏ nhất "
        f"({results[smallest_vocab_name]['vocab_size']} đơn vị) nên sparsity thường thấp hơn — "
        f"đặc biệt Pipeline C (subword/BPE) có xu hướng cho vocabulary nhỏ và dày đặc hơn vì "
        f"các subword được tái sử dụng giữa nhiều từ khác nhau."
    )

    best_search_name = max(results, key=lambda k: results[k]["search"]["mrr"])
    print(
        f"\n5) Pipeline nào cho search tốt nhất?\n"
        f"   -> Theo MRR đo được: {best_search_name} tốt nhất "
        f"(MRR = {results[best_search_name]['search']['mrr']:.4f}), so với "
        f"A={A['search']['mrr']:.4f}, B={B['search']['mrr']:.4f}, C={C['search']['mrr']:.4f}. "
        f"Kết quả này phụ thuộc nhiều vào cách xây dựng query trong thực nghiệm (lấy từ khoá "
        f"tần suất cao sau khi loại stopword) — pipeline nào có representation khớp tốt với "
        f"kiểu query này (từ khoá nội dung, không nhiễu bởi stopword) thường sẽ có MRR cao hơn."
    )

    vocab_sizes_sorted = sorted(results.items(), key=lambda kv: kv[1]["vocab_size"])
    mrr_sorted = sorted(results.items(), key=lambda kv: kv[1]["search"]["mrr"], reverse=True)
    vocab_order = [name for name, _ in vocab_sizes_sorted]
    mrr_order = [name for name, _ in mrr_sorted]
    consistent = vocab_order == list(reversed(mrr_order))
    print(
        f"\n6) Search tốt hơn có đồng nghĩa với vocabulary nhỏ hơn không?\n"
        f"   -> Thứ tự vocabulary (nhỏ -> lớn): {vocab_order}\n"
        f"   -> Thứ tự search MRR (tốt -> kém): {mrr_order}\n"
        f"   -> {'Hai thứ tự này KHỚP NHAU' if consistent else 'Hai thứ tự này KHÔNG khớp nhau hoàn toàn'}, "
        f"cho thấy vocabulary nhỏ hơn {'CÓ' if consistent else 'KHÔNG NHẤT THIẾT'} đi kèm search "
        f"tốt hơn trong lần chạy này. Về lý thuyết, vocabulary nhỏ hơn không đảm bảo chất lượng "
        f"tìm kiếm tốt hơn — nó chỉ ảnh hưởng đến việc biểu diễn có 'gộp' các biến thể của từ "
        f"lại hay không (giảm sparsity, tăng khả năng match), còn chất lượng search còn phụ "
        f"thuộc vào việc vocabulary đó có giữ được đủ tín hiệu phân biệt nội dung hay không "
        f"(loại quá tay có thể làm mất tín hiệu, gộp quá tay như BPE quá ít merge có thể làm "
        f"mất luôn ranh giới từ có ý nghĩa)."
    )

    print(f"\nTổng thời gian chạy toàn bộ thực nghiệm: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()