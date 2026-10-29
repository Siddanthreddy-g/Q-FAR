import argparse
import json
import os
import time

import numpy as np


def timed_search(index, q, k):
    """One query at a time (like your own latency numbers), returns scores, ids, ms/query."""
    ids = np.empty((len(q), k), dtype=np.int64)
    sc = np.empty((len(q), k), dtype=np.float32)
    t0 = time.time()
    for i in range(len(q)):
        s, j = index.search(q[i:i + 1], k)
        sc[i], ids[i] = s[0], j[0]
    return sc, ids, (time.time() - t0) / len(q) * 1000


def overlap_recall(ids, exact_ids, k):
    """Average fraction of the exact top-k that the approximate top-k contains."""
    return float(np.mean([len(set(ids[i, :k]) & set(exact_ids[i, :k])) / k for i in range(len(ids))]))


def nn_in_top(ids, exact_ids, r):
    """Fraction of queries whose exact nearest neighbour is inside the approximate top-r
    (the 'recall@R' used in the product-quantization / Faiss papers)."""
    return float(np.mean([exact_ids[i, 0] in ids[i, :r] for i in range(len(ids))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlist", type=int, default=64, help="IVF cells (same K as your K-Means)")
    ap.add_argument("--nprobe", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--ef", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--target_size", type=int, default=200000)
    ap.add_argument("--skip_hnsw", action="store_true", help="skip the slow HNSW index build")
    args = ap.parse_args()

    import faiss
    from beir.retrieval.evaluation import EvaluateRetrieval
    from cross_test_their_models import load_data, subsample_corpus

    max_threads = os.cpu_count() or 4

    def set_threads(n):
        try:
            faiss.omp_set_num_threads(n)
        except Exception:
            pass

    set_threads(1)      # timed searches use ONE thread, one query at a time (a fair latency)

    corpus, queries, qrels = load_data("nq")
    corpus = subsample_corpus(corpus, qrels, args.target_size)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    d = np.ascontiguousarray(np.load("cache/corpus_emb.npy"), dtype="float32")
    q = np.ascontiguousarray(np.load("cache/query_emb.npy"), dtype="float32")
    assert len(d) == len(doc_ids) and len(q) == len(query_ids), \
        "cache/ does not match this subsample (was it built with --target_size 200000?)"
    dim = d.shape[1]

    flat = faiss.IndexFlatIP(dim)
    flat.add(d)
    print("Step 1/3: exact search, one query at a time (about 2 minutes)...", flush=True)
    sc_e, exact_ids, ms_e = timed_search(flat, q, 100)     # exact result = ground truth for every metric below
    rows = []

    def add_row(method, params, ids, scores, ms, build_s):
        res = {query_ids[i]: {doc_ids[j]: float(s) for j, s in zip(ids[i], scores[i]) if j >= 0}
               for i in range(len(query_ids))}
        ndcg, _map, recall, _p = EvaluateRetrieval().evaluate(qrels, res, [1, 3, 5, 10, 100])
        rows.append({
            "method": method, "params": params,
            "ndcg": ndcg, "recall": recall, "precision": _p, "map": _map,   # the full metric set, saved to the json
            "overlap@10": overlap_recall(ids, exact_ids, 10),
            "NN_in_top1": nn_in_top(ids, exact_ids, 1),
            "NN_in_top10": nn_in_top(ids, exact_ids, 10),
            "NDCG@10": ndcg["NDCG@10"], "Recall@100": recall["Recall@100"],
            "ms_per_query": ms, "build_s": build_s})

    # reference: exact search over everything
    add_row("Flat (exact)", "-", exact_ids, sc_e, ms_e, 0.0)

    print("Step 2/3: FAISS IVF (K-Means partition)...", flush=True)
    # [1] IVF = K-Means partition + nprobe nearest cells
    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, args.nlist, faiss.METRIC_INNER_PRODUCT)
    try:
        ivf.cp.max_points_per_centroid = len(d)       # train on all vectors, like your own K-Means
    except Exception:
        pass
    set_threads(max_threads)        # building may use every core; only the timed searches are single-thread
    t0 = time.time()
    ivf.train(d)
    ivf.add(d)
    build = time.time() - t0
    set_threads(1)
    for npb in args.nprobe:
        ivf.nprobe = npb
        sc, ids, ms = timed_search(ivf, q, 100)
        add_row("FAISS IVF (Johnson et al.)", f"nlist={args.nlist}, nprobe={npb}", ids, sc, ms, build)

    if not args.skip_hnsw:
        print("Step 3/3: HNSW index build (the slowest step, can take 5 to 30 minutes)...", flush=True)
        # [2] HNSW graph index
        hn = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        hn.hnsw.efConstruction = 200
        set_threads(max_threads)
        t0 = time.time()
        hn.add(d)
        build = time.time() - t0
        set_threads(1)
        for ef in args.ef:
            hn.hnsw.efSearch = ef
            sc, ids, ms = timed_search(hn, q, 100)
            add_row("HNSW (Malkov & Yashunin)", f"M={args.hnsw_m}, efSearch={ef}", ids, sc, ms, build)

    hdr = f"{'method':30s} {'params':26s} {'ovl@10':>7s} {'NN@1':>6s} {'NN@10':>6s} {'NDCG@10':>8s} {'R@100':>7s} {'ms/q':>7s}"
    print("\n" + hdr)
    for r in rows:
        print(f"{r['method']:30s} {r['params']:26s} {r['overlap@10']:7.3f} {r['NN_in_top1']:6.3f} "
              f"{r['NN_in_top10']:6.3f} {r['NDCG@10']:8.4f} {r['Recall@100']:7.4f} {r['ms_per_query']:7.2f}")

    print("\nFull metrics (NDCG, Recall, Precision, MAP at 1, 3, 5, 10, 100):")
    for r in rows:
        print(f"\n{r['method']}  {r['params']}")
        for name in ("ndcg", "recall", "precision", "map"):
            print(f"  {name:9s}", {k: round(v, 5) for k, v in r[name].items()})

    os.makedirs("cache_cross", exist_ok=True)
    with open("cache_cross/their_ann_on_nq.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved cache_cross/their_ann_on_nq.json")


if __name__ == "__main__":
    main()