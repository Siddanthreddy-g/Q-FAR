import argparse
import json
import os
import time

import numpy as np


def load_beir(dataset):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    path = util.download_and_unzip(url, "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split="test")
    print(f"Loaded {dataset}: corpus={len(corpus):,} queries={len(queries):,}")
    return corpus, queries, qrels


def get_embeddings(corpus, queries, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    dpath, qpath = f"{cache_dir}/docs.npy", f"{cache_dir}/queries.npy"
    if os.path.exists(dpath) and os.path.exists(qpath):
        print("Loading cached embeddings from", cache_dir)
        return np.load(dpath), np.load(qpath)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    doc_ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]
    qtexts = [queries[q] for q in queries]
    d = model.encode(texts, batch_size=128, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    q = model.encode(qtexts, batch_size=128, show_progress_bar=True,
                     normalize_embeddings=True, convert_to_numpy=True)
    np.save(dpath, d)
    np.save(qpath, q)
    return d, q


def search_flat(d_emb, q_emb, doc_ids, query_ids, top_k=100):
    import faiss
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(d_emb.astype("float32"))
    t0 = time.time()
    scores, ids = index.search(q_emb.astype("float32"), top_k)
    ms = (time.time() - t0) / len(query_ids) * 1000
    results = {query_ids[i]: {doc_ids[j]: float(s) for j, s in zip(ids[i], scores[i])}
               for i in range(len(query_ids))}
    return results, ms, float(len(doc_ids))


def search_clustered(d_emb, q_emb, labels, route_scores, n_clusters, doc_ids, query_ids,
                     top_k=100, n_search=1):
    """route_scores[i, c] = how well query i matches cluster c (higher = better).
    Returns results, ms per query, and the average number of documents searched per query."""
    import faiss
    indices, positions = {}, {}
    for c in range(n_clusters):
        pos = np.where(labels == c)[0]
        positions[c] = pos
        if len(pos):
            ix = faiss.IndexFlatIP(d_emb.shape[1])
            ix.add(d_emb[pos].astype("float32"))
            indices[c] = ix
    chosen = np.argsort(-route_scores, axis=1)[:, :n_search]
    results, t_all, scanned = {}, 0.0, 0
    for i, qid in enumerate(query_ids):
        t0 = time.time()
        qv = q_emb[i:i + 1].astype("float32")
        cands = []
        for c in chosen[i]:
            if c not in indices:
                continue
            scanned += indices[c].ntotal
            k = min(top_k, indices[c].ntotal)
            s, loc = indices[c].search(qv, k)
            cands += [(doc_ids[positions[c][j]], float(sc)) for j, sc in zip(loc[0], s[0])]
        cands.sort(key=lambda x: -x[1])
        results[qid] = dict(cands[:top_k])
        t_all += time.time() - t0
    return results, t_all / len(query_ids) * 1000, scanned / len(query_ids)


def report(name, qrels, results, ms, scanned, n_docs):
    from beir.retrieval.evaluation import EvaluateRetrieval
    ndcg, _map, recall, precision = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
    print(f"\n--- {name} ---   avg latency {ms:.2f} ms/query")
    print(f"avg documents searched per query: {scanned:,.0f} ({100 * scanned / n_docs:.1f}% of the corpus)")
    print(f"NDCG@10 = {ndcg['NDCG@10']:.4f}   Recall@100 = {recall['Recall@100']:.4f}   (Contriever-paper metrics)")
    print("NDCG:", ndcg)
    print("Recall:", recall)
    print("Precision:", precision)
    print("MAP:", _map)
    return {"ndcg": ndcg, "recall": recall, "precision": precision, "map": _map,
            "latency_ms": ms, "avg_docs_searched": scanned, "pct_of_corpus": 100 * scanned / n_docs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--algo", choices=["flat", "kmeans", "fuzzy", "all"], default="all")
    ap.add_argument("--k", type=int, default=0, help="number of clusters (0 = automatic)")
    ap.add_argument("--clusters_to_search", type=int, nargs="+", default=[1],
                    help="clusters searched per query; give several (e.g. 1 2 4 8 16) for a curve from one fit")
    ap.add_argument("--m", type=float, default=2.0, help="Fuzzy C-Means fuzziness (2.0 = as specified)")
    ap.add_argument("--fcm_iters", type=int, default=20, help="max Fuzzy C-Means iterations")
    ap.add_argument("--subsample", type=int, default=0,
                    help="keep this many documents, same selection as your NQ runs (0 = whole corpus)")
    ap.add_argument("--emb_dir", default="",
                    help="folder holding corpus_emb.npy and query_emb.npy to re-use, e.g. 'cache'")
    args = ap.parse_args()

    corpus, queries, qrels = load_beir(args.dataset)
    if args.subsample:
        from cross_test_their_models import subsample_corpus
        corpus = subsample_corpus(corpus, qrels, args.subsample)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    if args.emb_dir:
        print("Re-using embeddings from", args.emb_dir)
        d_emb = np.load(f"{args.emb_dir}/corpus_emb.npy")
        q_emb = np.load(f"{args.emb_dir}/query_emb.npy")
    else:
        d_emb, q_emb = get_embeddings(corpus, queries, f"cache_{args.dataset}_{len(doc_ids)}")
    assert len(d_emb) == len(doc_ids) and len(q_emb) == len(query_ids), \
        "embeddings do not match this corpus (wrong --subsample or --emb_dir?)"
    n_docs = len(doc_ids)

    # ~300 documents per cluster, between 8 and 64 clusters (your NQ run used 64 clusters)
    k = args.k or int(min(64, max(8, n_docs // 300)))
    print(f"Clusters K = {k}, searching {args.clusters_to_search} cluster(s) per query")
    all_out = {"dataset": args.dataset, "docs": n_docs, "k": k,
               "clusters_to_search": args.clusters_to_search}
    summary = []

    def do(key, label, res, ms, scanned):
        out = report(label, qrels, res, ms, scanned, n_docs)
        all_out[key] = out
        summary.append((label, out["avg_docs_searched"], out["pct_of_corpus"],
                        out["ndcg"]["NDCG@10"], out["recall"]["Recall@100"], ms))

    if args.algo in ("flat", "all"):
        res, ms, scanned = search_flat(d_emb, q_emb, doc_ids, query_ids)
        do("flat", "FLAT bge-small (exhaustive)", res, ms, scanned)

    if args.algo in ("kmeans", "all"):
        from kmean_retrival import KMeansScratch
        km = KMeansScratch(n_clusters=k, max_iters=20, chunk_size=2000, random_state=42).fit(d_emb)
        labels = km.predict(d_emb)
        cents = km.cluster_centers_ / np.maximum(np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True), 1e-9)
        sizes = np.bincount(labels, minlength=k)
        print(f"K-Means cluster sizes min/max {sizes.min()}/{sizes.max()} (balanced = {n_docs // k})")
        for ns in args.clusters_to_search:
            res, ms, scanned = search_clustered(d_emb, q_emb, labels, q_emb @ cents.T, k,
                                                doc_ids, query_ids, n_search=ns)
            do(f"kmeans_s{ns}", f"K-Means (from scratch), K={k}, {ns} cluster(s) searched",
               res, ms, scanned)

    if args.algo in ("fuzzy", "all"):
        from fuzzycmeans_scratch import FuzzyCMeansScratch
        fcm = FuzzyCMeansScratch(n_clusters=k, m=args.m, max_iters=args.fcm_iters, chunk_size=2000,
                                 random_state=42).fit(d_emb)
        # Collapse check: if every centroid sits at the data mean, the highest membership is
        # about 1/K, the centroid spread is about 0 and the cluster sizes are wildly uneven.
        C = fcm.cluster_centers_
        spread = np.linalg.norm(C[:, None] - C[None], axis=2)[np.triu_indices(k, 1)].mean()
        sizes = np.bincount(fcm.labels_, minlength=k)
        print(f"FCM check (m={args.m}): avg highest membership {fcm.u_.max(1).mean():.4f} "
              f"(collapsed = {1 / k:.4f}) | centroid spread {spread:.2e} | "
              f"cluster sizes min/max {sizes.min()}/{sizes.max()} (balanced = {n_docs // k})")
        memberships = fcm.predict_proba(q_emb)
        for ns in args.clusters_to_search:
            res, ms, scanned = search_clustered(d_emb, q_emb, fcm.labels_, memberships, k,
                                                doc_ids, query_ids, n_search=ns)
            do(f"fuzzy_m{args.m}_s{ns}", f"Fuzzy C-Means (from scratch, m={args.m}), K={k}, {ns} cluster(s) searched",
               res, ms, scanned)

    print("\n=== SUMMARY: compare methods at a similar number of documents searched ===")
    print(f"{'method':72s} {'docs searched':>13s} {'% corpus':>8s} {'NDCG@10':>8s} {'R@100':>7s} {'ms/q':>7s}")
    for lab, sc, pc, nd, r1, ms in summary:
        print(f"{lab:72s} {sc:13,.0f} {pc:7.1f}% {nd:8.4f} {r1:7.4f} {ms:7.2f}")

    out_dir = f"cache_{args.dataset}_{n_docs}"
    os.makedirs(out_dir, exist_ok=True)
    s_tag = "-".join(str(x) for x in args.clusters_to_search)
    out = f"{out_dir}/our_results_{args.algo}_m{args.m}_s{s_tag}.json"
    with open(out, "w") as f:
        json.dump(all_out, f, indent=2)
    print("\nSaved", out)


if __name__ == "__main__":
    main()