import argparse
import functools
import importlib
import json
import os
import random
import re
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

try:
    stem_module = importlib.import_module("nltk.stem")
    _stemmer = stem_module.PorterStemmer()
except Exception:
    _stemmer = None

try:
    corpus_module = importlib.import_module("nltk.corpus")
    wn = corpus_module.wordnet
    wn.synsets("test")          # raises LookupError if the wordnet data is missing
    WORDNET_OK = True
except Exception:
    wn = None
    WORDNET_OK = False

TOKEN_RE = re.compile(r"[a-z]+")


# ----------------------------------------------------------------------------
# 1. Pre-processing
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def stem(word):
    return _stemmer.stem(word) if _stemmer else word


def content_words(text):
    """Lower-case, tokenise, remove stop words."""
    return [w for w in TOKEN_RE.findall(text.lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def preprocess(text):
    """Stop-word removal followed by stemming (paper, Section III-A)."""
    return [stem(w) for w in content_words(text)]


# ----------------------------------------------------------------------------
# 2. Possibilistic Fuzzy C-Means (Pal et al. 2005), Euclidean distance
# ----------------------------------------------------------------------------
class PFCMScratch:
    """
    J = sum_k sum_i (a*u_ik^m + b*t_ik^eta) * d_ik^2  +  sum_i gamma_i * sum_k (1 - t_ik)^eta

    u_ik = d_ik^(-2/(m-1)) / sum_j d_jk^(-2/(m-1))            (membership, rows sum to 1)
    t_ik = 1 / (1 + ((b/gamma_i) * d_ik^2)^(1/(eta-1)))        (typicality, NOT forced to sum to 1)
    v_i  = sum_k (a*u_ik^m + b*t_ik^eta) x_k / sum_k (a*u_ik^m + b*t_ik^eta)
    gamma_i = K * sum_k u_ik^m d_ik^2 / sum_k u_ik^m
    """

    def __init__(self, n_clusters, a=1.0, b=1.0, m=2.0, eta=2.0, K=1.0,
                 max_iters=30, tol=1e-4, random_state=42, eps=1e-10):
        self.n_clusters = n_clusters
        self.a, self.b, self.m, self.eta, self.K = a, b, m, eta, K
        self.max_iters, self.tol = max_iters, tol
        self.random_state, self.eps = random_state, eps
        self.cluster_centers_ = None
        self.u_ = None
        self.t_ = None
        self.labels_ = None

    def _sqdist(self, X, V):
        x2 = (X * X).sum(1, keepdims=True)
        v2 = (V * V).sum(1)[None, :]
        return np.maximum(x2 - 2.0 * X @ V.T + v2, self.eps)

    def _memberships(self, d2):
        w = d2 ** (-1.0 / (self.m - 1.0))
        return w / w.sum(axis=1, keepdims=True)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.random_state)
        V = X[rng.choice(X.shape[0], self.n_clusters, replace=False)].copy()

        for it in range(self.max_iters):
            t0 = time.time()
            d2 = self._sqdist(X, V)
            u = self._memberships(d2)
            um = u ** self.m
            gamma = self.K * (um * d2).sum(0) / np.maximum(um.sum(0), self.eps)
            t = 1.0 / (1.0 + ((self.b / np.maximum(gamma, self.eps))[None, :] * d2)
                       ** (1.0 / (self.eta - 1.0)))
            w = self.a * um + self.b * (t ** self.eta)
            V_new = (w.T @ X) / np.maximum(w.sum(0)[:, None], self.eps)
            shift = float(np.linalg.norm(V_new - V))
            V = V_new.astype(np.float32)
            print(f"  PFCM iter {it+1:2d}/{self.max_iters}  centroid shift={shift:.6f}  ({time.time()-t0:.1f}s)")
            if shift < self.tol:
                print(f"  PFCM converged after {it+1} iterations")
                break

        d2 = self._sqdist(X, V)
        self.cluster_centers_ = V
        self.u_ = self._memberships(d2)
        self.labels_ = np.argmax(self.u_, axis=1)
        return self


# ----------------------------------------------------------------------------
# 3. Per-cluster keyword index   imp(K_i) = Cnt(K_i) > Th        (eq. 15)
# ----------------------------------------------------------------------------
def build_cluster_index(X_binary, labels, n_clusters, th_ratio):
    n_terms = X_binary.shape[1]
    index = np.zeros((n_clusters, n_terms), dtype=np.float32)
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        counts = X_binary[members].sum(axis=0)
        th = max(2.0, th_ratio * len(members))
        index[c] = np.where(counts > th, counts, 0.0)
    return index


# ----------------------------------------------------------------------------
# 4. Query reformulation with WordNet synonyms
# ----------------------------------------------------------------------------
def reformulate(question, vocab, use_wordnet=True):
    weights = {}
    for w in content_words(question):
        weights[stem(w)] = 1.0
        if use_wordnet and WORDNET_OK:
            for syn in wn.synsets(w)[:3]:
                for lemma in syn.lemma_names():
                    lemma = lemma.lower()
                    if "_" not in lemma and lemma != w:
                        weights.setdefault(stem(lemma), 0.5)
    vec = np.zeros(len(vocab), dtype=np.float32)
    for term, wt in weights.items():
        j = vocab.get(term)
        if j is not None:
            vec[j] = max(vec[j], wt)
    return vec


# ----------------------------------------------------------------------------
# 5. Retrieval + evaluation
# ----------------------------------------------------------------------------
def retrieve(queries, qids, X_norm, cluster_index, labels, vocab, doc_ids,
             top_k, use_wordnet):
    idx_norm = cluster_index / np.maximum(np.linalg.norm(cluster_index, axis=1, keepdims=True), 1e-9)
    ranked, whole_cluster, no_match = {}, {}, 0
    for qid in qids:
        q = reformulate(queries[qid], vocab, use_wordnet)
        qn = np.linalg.norm(q)
        if qn == 0:                      # none of the query words is in the vocabulary
            no_match += 1
            ranked[qid], whole_cluster[qid] = {}, []
            continue
        best = int(np.argmax(idx_norm @ (q / qn)))
        members = np.where(labels == best)[0]
        whole_cluster[qid] = [doc_ids[i] for i in members]        # what the paper "releases"
        scores = X_norm[members] @ q
        order = np.argsort(-scores)[:top_k]
        ranked[qid] = {doc_ids[members[i]]: float(scores[i]) for i in order}
    return ranked, whole_cluster, no_match


def paper_precision(whole_cluster, qrels):
    """Precision exactly as defined in the paper: |relevant AND retrieved| / |retrieved|,
    where 'retrieved' is the entire released cluster. Averaged over queries."""
    vals = []
    for qid, docs in whole_cluster.items():
        if not docs:
            vals.append(0.0)
            continue
        rel = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        vals.append(len(rel.intersection(docs)) / len(docs))
    return float(np.mean(vals))


def load_dataset(dataset, target_size, seed=42):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split="test")
    if target_size and target_size < len(corpus):
        random.seed(seed)
        relevant = {d for docs in qrels.values() for d in docs}
        all_ids = list(corpus.keys())                       # stable order
        pool = [d for d in all_ids if d not in relevant]
        extra = random.sample(pool, min(max(0, target_size - len(relevant)), len(pool)))
        keep = relevant | set(extra)
        corpus = {d: corpus[d] for d in all_ids if d in keep}   # filter the list, not the set
    print(f"Corpus={len(corpus):,}  queries={len(queries):,}")
    return corpus, queries, qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nq")
    ap.add_argument("--target_size", type=int, default=20000)
    ap.add_argument("--n_clusters", type=int, default=64)
    ap.add_argument("--n_terms", type=int, default=2000)
    ap.add_argument("--th_ratio", type=float, default=0.05)
    ap.add_argument("--no_wordnet", action="store_true")
    args = ap.parse_args()

    if not args.no_wordnet and not WORDNET_OK:
        print("WARNING: WordNet not available -> running WITHOUT synonym expansion.\n"
              "         pip install nltk ; python -c \"import nltk; nltk.download('wordnet')\"")
    use_wordnet = (not args.no_wordnet) and WORDNET_OK

    corpus, queries, qrels = load_dataset(args.dataset, args.target_size)
    doc_ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]

    print("Building term matrix...")
    cv = CountVectorizer(analyzer=preprocess, max_features=args.n_terms)
    counts = cv.fit_transform(texts).toarray().astype(np.float32)
    vocab = cv.vocabulary_
    X_binary = (counts > 0).astype(np.float32)
    X_norm = counts / np.maximum(np.linalg.norm(counts, axis=1, keepdims=True), 1e-9)

    print(f"Fitting PFCM (K={args.n_clusters}) on {X_norm.shape[0]:,} documents...")
    pfcm = PFCMScratch(n_clusters=args.n_clusters).fit(X_norm)
    index = build_cluster_index(X_binary, pfcm.labels_, args.n_clusters, args.th_ratio)

    qids = [q for q in queries if q in qrels]
    ranked, whole, no_match = retrieve(queries, qids, X_norm, index, pfcm.labels_,
                                       vocab, doc_ids, 100, use_wordnet)

    from beir.retrieval.evaluation import EvaluateRetrieval
    ndcg, _map, recall, precision = EvaluateRetrieval().evaluate(
        {q: qrels[q] for q in qids}, ranked, [1, 3, 5, 10, 100])
    p_paper = paper_precision(whole, qrels)

    print("\n=== Murali & Damodaram (2015) reimplementation ===")
    print(f"dataset={args.dataset}  docs={len(doc_ids):,}  clusters={args.n_clusters}  wordnet={use_wordnet}")
    print(f"queries with no keyword in vocabulary: {no_match} / {len(qids)}")
    print(f"Precision (paper definition, whole cluster released): {p_paper:.5f}")
    print("NDCG:", ndcg)
    print("Recall:", recall)
    print("Precision@k (ranked inside cluster):", precision)
    print("MAP:", _map)

    os.makedirs("cache_cross", exist_ok=True)
    with open(f"cache_cross/pfcm_murali_{args.dataset}_{len(doc_ids)}.json", "w") as f:
        json.dump({"paper_precision": p_paper, "ndcg": ndcg, "recall": recall,
                   "precision": precision, "map": _map, "no_match": no_match,
                   "wordnet": use_wordnet}, f, indent=2)


if __name__ == "__main__":
    main()
