import os
import time
import pickle
import random
import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

"""from sentence_transformers import SentenceTransformer
from kmean_retrival import KMeansScratch
import faiss  -kmean"""
from sentence_transformers import SentenceTransformer
from fuzzycmeans_scratch import FuzzyCMeansScratch
import faiss

# ----------------------------- CONFIG ---------------------------------
DATASET          = "nq"
DATA_DIR         = "datasets"
SPLIT            = "test"

TARGET_CORPUS_SIZE = 200_000
EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
EMBED_BATCH_SIZE  = 256

N_CLUSTERS        = 64
FUZZINESS_M       = 2.0  # Only used for fuzzy c-means, not K-Means.
CLUSTERS_TO_SEARCH = 1
TOP_K              = 100

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)


# ------------------------ 1. LOAD DATASET -------------------------------
def load_data():
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET}.zip"
    data_path = util.download_and_unzip(url, DATA_DIR)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=SPLIT)
    print(f"Loaded corpus={len(corpus):,}  queries={len(queries):,}  qrel-queries={len(qrels):,}")
    return corpus, queries, qrels


# ------------------------ 2. SUBSAMPLE -----------------------------------
def subsample_corpus(corpus, qrels, target_size):
    if target_size is None or target_size >= len(corpus):
        return corpus

    relevant_doc_ids = set()
    for qid, docs in qrels.items():
        relevant_doc_ids.update(docs.keys())

    all_ids = list(corpus.keys())
    extra_needed = max(0, target_size - len(relevant_doc_ids))
    pool = [d for d in all_ids if d not in relevant_doc_ids]
    extra_ids = random.sample(pool, min(extra_needed, len(pool)))

    keep_ids_set = relevant_doc_ids | set(extra_ids)
    keep_ids = [d for d in all_ids if d in keep_ids_set]

    sub_corpus = {doc_id: corpus[doc_id] for doc_id in keep_ids}
    print(f"Subsampled corpus: {len(sub_corpus):,} docs "
          f"({len(relevant_doc_ids):,} kept for relevance, {len(extra_ids):,} random)")
    return sub_corpus

# ------------------------ 3. EMBEDDING -----------------------------------
def embed_texts(model, texts, batch_size, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    np.save(cache_path, embeddings)
    return embeddings


def build_corpus_texts(corpus, doc_ids):
    return [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]


# ------------------------ 4. K-MEANS CLUSTERING ---------------------------
"""def fit_kmeans(embeddings, n_clusters, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached K-Means model from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Fitting K-Means with K={n_clusters} on {embeddings.shape[0]:,} vectors...")
    t0 = time.time()
    km = KMeansScratch(
        n_clusters=n_clusters,
        max_iters=20,
        chunk_size=2000,
        random_state=42,
    )
    km.fit(embeddings)
    print(f"K-Means fit in {time.time() - t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(km, f)
    return km"""
# ------------------------ 4. FUZZY C-MEANS CLUSTERING ---------------------
def fit_fuzzy_cmeans(embeddings, n_clusters, m, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached Fuzzy C-Means model from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"Fitting Fuzzy C-Means with K={n_clusters}, m={m} on {embeddings.shape[0]:,} vectors...")
    t0 = time.time()
    fcm = FuzzyCMeansScratch(
        n_clusters=n_clusters,
        m=m,
        max_iters=20,
        chunk_size=2000,
        random_state=42,
    )
    fcm.fit(embeddings)
    print(f"Fuzzy C-Means fit in {time.time() - t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump(fcm, f)
    return fcm


# ------------------------ 5. CLUSTER-LOCAL FAISS INDICES -----------------
def build_cluster_indices(embeddings, cluster_labels, n_clusters, dim):
    cluster_indices = {}
    cluster_doc_positions = {}

    for c in range(n_clusters):
        member_positions = np.where(cluster_labels == c)[0]
        cluster_doc_positions[c] = member_positions
        if len(member_positions) == 0:
            cluster_indices[c] = None
            continue
        sub_embeddings = embeddings[member_positions]
        index = faiss.IndexFlatIP(dim)
        index.add(sub_embeddings.astype("float32"))
        cluster_indices[c] = index

    return cluster_indices, cluster_doc_positions


# ------------------------ 6. QUERY ROUTING + SEARCH -----------------------
"""def route_and_search(query_embeddings, centroids, cluster_indices,
                      cluster_doc_positions, doc_ids, top_k, clusters_to_search):
    norm_centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    sims_to_centroids = query_embeddings @ norm_centroids.T
    nearest_clusters = np.argsort(-sims_to_centroids, axis=1)[:, :clusters_to_search] -kmean"""
# ------------------------ 6. QUERY ROUTING + SEARCH -----------------------
def route_and_search(query_embeddings, fcm, cluster_indices,
                      cluster_doc_positions, doc_ids, top_k, clusters_to_search):
    """Route each query to clusters with the highest Fuzzy C-Means membership."""
    query_memberships = fcm.predict_proba(query_embeddings)
    nearest_clusters = np.argsort(-query_memberships, axis=1)[:, :clusters_to_search]

    results = {}
    latencies = []

    for qi in range(query_embeddings.shape[0]):
        t0 = time.time()
        q_vec = query_embeddings[qi:qi+1].astype("float32")
        candidates = []

        for c in nearest_clusters[qi]:
            index = cluster_indices[c]
            if index is None or index.ntotal == 0:
                continue
            k = min(top_k, index.ntotal)
            scores, local_idx = index.search(q_vec, k)
            positions = cluster_doc_positions[c][local_idx[0]]
            for pos, score in zip(positions, scores[0]):
                candidates.append((doc_ids[pos], float(score)))

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:top_k]
        results[f"q{qi}"] = {doc_id: score for doc_id, score in candidates}
        latencies.append(time.time() - t0)

    return results, latencies

    results = {}
    latencies = []

    for qi in range(query_embeddings.shape[0]):
        t0 = time.time()
        q_vec = query_embeddings[qi:qi+1].astype("float32")
        candidates = []

        for c in nearest_clusters[qi]:
            index = cluster_indices[c]
            if index is None or index.ntotal == 0:
                continue
            k = min(top_k, index.ntotal)
            scores, local_idx = index.search(q_vec, k)
            positions = cluster_doc_positions[c][local_idx[0]]
            for pos, score in zip(positions, scores[0]):
                candidates.append((doc_ids[pos], float(score)))

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:top_k]
        results[f"q{qi}"] = {doc_id: score for doc_id, score in candidates}
        latencies.append(time.time() - t0)

    return results, latencies


# ------------------------------- MAIN -------------------------------------
def main():
    corpus, queries, qrels = load_data()
    corpus = subsample_corpus(corpus, qrels, TARGET_CORPUS_SIZE)

    doc_ids = list(corpus.keys())
    query_ids = list(queries.keys())

    print("Loading embedding model:", EMBED_MODEL)
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    corpus_texts = build_corpus_texts(corpus, doc_ids)
    query_texts = [queries[q] for q in query_ids]

    corpus_embeddings = embed_texts(
        model, corpus_texts, EMBED_BATCH_SIZE, f"{CACHE_DIR}/corpus_emb.npy"
    )
    query_embeddings = embed_texts(
        model, query_texts, EMBED_BATCH_SIZE, f"{CACHE_DIR}/query_emb.npy"
    )

    """km = fit_kmeans(corpus_embeddings, N_CLUSTERS, f"{CACHE_DIR}/kmeans_k{N_CLUSTERS}.pkl")
    cluster_labels = km.predict(corpus_embeddings)"""
    fcm = fit_fuzzy_cmeans(corpus_embeddings, N_CLUSTERS, FUZZINESS_M, f"{CACHE_DIR}/fuzzy_cmeans_k{N_CLUSTERS}.pkl")
    cluster_labels = fcm.labels_

    cluster_indices, cluster_doc_positions = build_cluster_indices(
        corpus_embeddings, cluster_labels, N_CLUSTERS, dim
    )

    """print(f"Routing {len(query_ids):,} queries (searching top-{CLUSTERS_TO_SEARCH} cluster(s) each)...")
    results, latencies = route_and_search(
        query_embeddings, km.cluster_centers_, cluster_indices,
        cluster_doc_positions, doc_ids, TOP_K, CLUSTERS_TO_SEARCH
    )"""
    print(f"Routing {len(query_ids):,} queries (searching top-{CLUSTERS_TO_SEARCH} cluster(s) each, by membership)...")
    results, latencies = route_and_search(
    query_embeddings, fcm, cluster_indices,
    cluster_doc_positions, doc_ids, TOP_K, CLUSTERS_TO_SEARCH
)

    results_by_qid = {query_ids[i]: results[f"q{i}"] for i in range(len(query_ids))}

    retriever = EvaluateRetrieval()
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results_by_qid, [1, 3, 5, 10, 100]
    )

    """print("\nK-Means Hard-Routing Results")
    print(f"K (clusters)         : {N_CLUSTERS}")
    print(f"Clusters searched/q  : {CLUSTERS_TO_SEARCH}")"""
    print("\nFuzzy C-Means Routing Results")
    print(f"K (clusters)         : {N_CLUSTERS}")
    print(f"Fuzziness (m)        : {FUZZINESS_M}")
    print(f"Clusters searched/q  : {CLUSTERS_TO_SEARCH}")
    print(f"Corpus size          : {len(corpus):,}")
    print(f"Avg latency / query  : {np.mean(latencies)*1000:.2f} ms")
    print(f"P95 latency / query  : {np.percentile(latencies, 95)*1000:.2f} ms")
    print("\nNDCG:", ndcg)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MAP:", _map)

    """with open(f"{CACHE_DIR}/kmeans_results.pkl", "wb") as f:
        pickle.dump({
            "ndcg": ndcg, "recall": recall, "precision": precision, "map": _map,
            "latencies": latencies, "n_clusters": N_CLUSTERS,
            "clusters_searched": CLUSTERS_TO_SEARCH, "corpus_size": len(corpus),
        }, f)
    print(f"\nSaved metrics to {CACHE_DIR}/kmeans_results.pkl")"""
    with open(f"{CACHE_DIR}/fuzzy_results.pkl", "wb") as f:
        pickle.dump({
            "ndcg": ndcg, "recall": recall, "precision": precision, "map": _map,
            "latencies": latencies, "n_clusters": N_CLUSTERS, "m": FUZZINESS_M,
            "clusters_searched": CLUSTERS_TO_SEARCH, "corpus_size": len(corpus),
        }, f)
    print(f"\nSaved metrics to {CACHE_DIR}/fuzzy_results.pkl")


if __name__ == "__main__":
    main()