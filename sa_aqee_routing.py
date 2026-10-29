

import argparse
import contextlib
import io
import json
import os

import numpy as np

from sa_aqee_fcm import memberships, sa_aqee_fcm, sqdist


# ------------------------------------------------------------------------------ searching
def build_indices(d_emb, labels, C):
    import faiss
    indices, positions = {}, {}
    for c in range(C):
        pos = np.where(labels == c)[0]
        positions[c] = pos
        if len(pos):
            ix = faiss.IndexFlatIP(d_emb.shape[1])
            ix.add(d_emb[pos].astype("float32"))
            indices[c] = ix
    return indices, positions


def search_chosen(indices, positions, q_emb, chosen, doc_ids, query_ids, top_k=100):
    """chosen[i] = the clusters to search for query i. Returns results and the average documents searched."""
    results, scanned = {}, 0
    for i, qid in enumerate(query_ids):
        qv = q_emb[i:i + 1].astype("float32")
        cands = []
        for c in chosen[i]:
            if c not in indices:
                continue
            scanned += indices[c].ntotal
            s, loc = indices[c].search(qv, min(top_k, indices[c].ntotal))
            cands += [(doc_ids[positions[c][j]], float(sc)) for j, sc in zip(loc[0], s[0])]
        cands.sort(key=lambda x: -x[1])
        results[qid] = dict(cands[:top_k])
    return results, scanned / len(query_ids)


# ---------------------------------------------------------------------------------- routing
def choose_fixed(d2, n):
    order = np.argsort(d2, axis=1)[:, :n]
    return [row for row in order]


def choose_gap(d2, delta, max_clusters):
    order = np.argsort(d2, axis=1)[:, :max_clusters]
    best = d2.min(1)
    return [row[d2[i, row] <= (1.0 + delta) * best[i] + 1e-12] for i, row in enumerate(order)]


def choose_membership(U, tau, max_clusters):
    order = np.argsort(-U, axis=1)[:, :max_clusters]
    out = []
    for i, row in enumerate(order):
        n = int((np.cumsum(U[i, row]) < tau).sum()) + 1
        out.append(row[:n])
    return out


def pareto(points):
    """Upper envelope of (docs, ndcg) points: the best quality reachable at each cost."""
    best, env = -1.0, []
    for d, n in sorted(points):
        if n > best:
            best = n
            env.append((d, n))
    return env


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--pca", type=int, default=0)
    ap.add_argument("--m_min", type=float, default=1.05)
    ap.add_argument("--fit_sample", type=int, default=0)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--emb_dir", default="")
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--taus", type=float, nargs="+", default=[0.6, 0.8, 0.9, 0.95])
    ap.add_argument("--max_clusters", type=int, default=8)
    args = ap.parse_args()

    from run_our_model_on_their_data import load_beir, get_embeddings
    from beir.retrieval.evaluation import EvaluateRetrieval
    from kmean_retrival import KMeansScratch

    corpus, queries, qrels = load_beir(args.dataset)
    if args.subsample:
        from cross_test_their_models import subsample_corpus
        corpus = subsample_corpus(corpus, qrels, args.subsample)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    n_docs = len(doc_ids)
    if args.emb_dir:
        d_emb, q_emb = np.load(f"{args.emb_dir}/corpus_emb.npy"), np.load(f"{args.emb_dir}/query_emb.npy")
    else:
        d_emb, q_emb = get_embeddings(corpus, queries, f"cache_{args.dataset}_{n_docs}")
    C = args.k or int(min(64, max(8, n_docs // 300)))
    X, Q = d_emb.astype(np.float64), q_emb.astype(np.float64)
    if args.pca:
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X[:min(len(X), 5000)] - mu, full_matrices=False)
        P = Vt[:args.pca].T
        Xc, Qc = (X - mu) @ P, (Q - mu) @ P
    else:
        Xc, Qc = X, Q
    print(f"{args.dataset}: {n_docs} docs, C={C}, seeds {args.seeds}, at most {args.max_clusters} clusters per query", flush=True)

    rows = []          # (seed, method, rule, setting, docs, ndcg10, recall100, full metrics)

    def evaluate(seed, method, rule, setting, indices, positions, chosen):
        results, scanned = search_chosen(indices, positions, q_emb, chosen, doc_ids, query_ids)
        ndcg, _map, recall, prec = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
        rows.append(dict(seed=seed, method=method, rule=rule, setting=setting, docs=scanned, ndcg10=ndcg["NDCG@10"],
                         recall100=recall["Recall@100"], ndcg=ndcg, recall=recall, precision=prec, map=_map))

    for seed in args.seeds:
        print(f"\n=== seed {seed}: fitting SA-AQEE-FCM and K-Means ===", flush=True)
        res = sa_aqee_fcm(Xc, C, "V5", np.random.default_rng(seed), sample=2000, fit_sample=args.fit_sample,
                          m_min=args.m_min, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            km = KMeansScratch(n_clusters=C, random_state=seed).fit(Xc)
        sa = dict(name="SA-AQEE-FCM (ours)", labels=res["labels"], V=res["V"], m=res["m"])
        kmeans = dict(name="K-Means (previous)", labels=km.labels_, V=km.cluster_centers_.astype(np.float64), m=None)
        for meth in (sa, kmeans):
            indices, positions = build_indices(d_emb, meth["labels"], C)
            d2 = sqdist(Qc, meth["V"])
            for n in args.n_search:
                evaluate(seed, meth["name"], "fixed", n, indices, positions, choose_fixed(d2, n))
            for dl in args.deltas:
                evaluate(seed, meth["name"], "gap", dl, indices, positions, choose_gap(d2, dl, args.max_clusters))
            if meth is sa:
                U = memberships(Qc, meth["V"], meth["m"])
                for tau in args.taus:
                    evaluate(seed, meth["name"], "membership", tau, indices, positions, choose_membership(U, tau, args.max_clusters))
            print(f"  {meth['name']} done", flush=True)

    # ---------------------------------------------------------------- mean over seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["rule"], r["setting"]), []).append(r)
    agg = []
    for (method, rule, setting), rs in groups.items():
        agg.append(dict(method=method, rule=rule, setting=setting, docs=float(np.mean([r["docs"] for r in rs])),
                        ndcg10=float(np.mean([r["ndcg10"] for r in rs])), recall100=float(np.mean([r["recall100"] for r in rs])),
                        ndcg10_std=float(np.std([r["ndcg10"] for r in rs]))))
    print("\n=== MEAN OVER SEEDS ===")
    print(f"{'method':22s} {'rule':11s} {'setting':>8s} {'docs searched':>14s} {'% corpus':>8s} {'NDCG@10':>8s} {'(std)':>7s} {'R@100':>7s}")
    for a in sorted(agg, key=lambda a: (a["method"], a["docs"])):
        print(f"{a['method']:22s} {a['rule']:11s} {a['setting']:8.2f} {a['docs']:14,.0f} {100 * a['docs'] / n_docs:7.1f}% "
              f"{a['ndcg10']:8.4f} {a['ndcg10_std']:7.3f} {a['recall100']:7.4f}")

    # ------------------------------------------------ SA-AQEE-FCM against the best K-Means curve at equal cost
    km_env = pareto([(a["docs"], a["ndcg10"]) for a in agg if a["method"].startswith("K-Means")])
    ed, en = [p[0] for p in km_env], [p[1] for p in km_env]
    print("\n=== OUR METHOD AGAINST THE BEST K-MEANS CURVE (any rule) AT THE SAME NUMBER OF DOCUMENTS SEARCHED ===")
    print(f"{'rule':11s} {'setting':>8s} {'docs searched':>14s} {'ours NDCG@10':>13s} {'K-Means at same cost':>21s} {'difference':>11s}")
    wins = total = 0
    for a in sorted([a for a in agg if a["method"].startswith("SA-AQEE")], key=lambda a: a["docs"]):
        if a["docs"] < ed[0] or a["docs"] > ed[-1]:
            continue
        ref = float(np.interp(a["docs"], ed, en))
        total += 1
        wins += a["ndcg10"] > ref
        print(f"{a['rule']:11s} {a['setting']:8.2f} {a['docs']:14,.0f} {a['ndcg10']:13.4f} {ref:21.4f} {a['ndcg10'] - ref:+11.4f}")
    print(f"\nour points above the K-Means curve: {wins} of {total}")

    os.makedirs("cache_cross", exist_ok=True)
    name = args.dataset + (str(n_docs) if args.subsample else "")
    path = f"cache_cross/sa_aqee_routing_{name}.json"
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": rows, "mean": agg}, f, indent=1, default=float)
    print("Saved", path)


if __name__ == "__main__":
    main()
