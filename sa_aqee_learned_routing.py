

import argparse
import contextlib
import io
import json
import os

import numpy as np
from scipy.optimize import minimize

from sa_aqee_fcm import sa_aqee_fcm, sqdist
from sa_aqee_routing import build_indices, pareto, search_chosen


# --------------------------------------------------------------------------------- router
def _unpack(theta, C, D):
    return theta[0], theta[1:1 + C * D].reshape(C, D), theta[1 + C * D:]


def fit_router(Q, y, V, lam, maxiter=200):
    N, D = Q.shape
    C = len(V)
    S = -sqdist(Q, V)
    Y = np.zeros((N, C))
    Y[np.arange(N), y] = 1.0

    def f(theta):
        a, W, b = _unpack(theta, C, D)
        L = a * S + Q @ W.T + b
        L -= L.max(1, keepdims=True)
        P = np.exp(L)
        P /= P.sum(1, keepdims=True)
        loss = -np.log(P[np.arange(N), y] + 1e-12).mean() + lam * (W ** 2).sum()
        G = (P - Y) / N
        return loss, np.concatenate([[(G * S).sum()], (G.T @ Q + 2 * lam * W).ravel(), G.sum(0)])

    theta0 = np.concatenate([[10.0], np.zeros(C * D), np.zeros(C)])
    res = minimize(f, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return _unpack(res.x, C, D)


def logits(Q, V, params):
    a, W, b = params
    return a * (-sqdist(Q, V)) + Q @ W.T + b


def val_loss(Q, y, V, params):
    L = logits(Q, V, params)
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())


def choose_lambda(Q, y, qidx, V, rng, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Pick the ridge penalty on a validation part of the TRAINING queries only."""
    uq = np.unique(qidx)
    val_q = set(rng.choice(uq, max(1, len(uq) // 5), replace=False).tolist())
    is_val = np.array([q in val_q for q in qidx])
    best, best_l = np.inf, grid[0]
    for lam in grid:
        p = fit_router(Q[~is_val], y[~is_val], V, lam, maxiter=100)
        l = val_loss(Q[is_val], y[is_val], V, p)
        if l < best:
            best, best_l = l, lam
    return best_l


def top_n(scores, n):
    return [row for row in np.argsort(-scores, axis=1)[:, :n]]


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
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--oracle", action="store_true", help="also show the upper bound that uses the answers")
    args = ap.parse_args()

    from run_our_model_on_their_data import load_beir, get_embeddings
    from beir.retrieval.evaluation import EvaluateRetrieval
    from kmean_retrival import KMeansScratch

    corpus, queries, qrels = load_beir(args.dataset)
    if args.subsample:
        from cross_test_their_models import subsample_corpus
        corpus = subsample_corpus(corpus, qrels, args.subsample)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    n_docs, n_q = len(doc_ids), len(query_ids)
    doc_pos = {d: i for i, d in enumerate(doc_ids)}
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
    rel = [[doc_pos[d] for d, s in qrels.get(qid, {}).items() if s > 0 and d in doc_pos] for qid in query_ids]
    print(f"{args.dataset}: {n_docs} docs, {n_q} queries, C={C}, seeds {args.seeds}; "
          f"{sum(len(r) > 0 for r in rel)} queries have a relevant document in the corpus", flush=True)

    rows = []

    def evaluate(seed, method, router, n, indices, positions, chosen):
        results, scanned = search_chosen(indices, positions, q_emb, chosen, doc_ids, query_ids)
        ndcg, _m, recall, prec = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
        rows.append(dict(seed=seed, method=method, router=router, n=n, docs=scanned, ndcg10=ndcg["NDCG@10"],
                         recall100=recall["Recall@100"], ndcg=ndcg, recall=recall, precision=prec, map=_m))

    for seed in args.seeds:
        print(f"\n=== seed {seed}: fitting SA-AQEE-FCM and K-Means ===", flush=True)
        res = sa_aqee_fcm(Xc, C, "V5", np.random.default_rng(seed), sample=2000, fit_sample=args.fit_sample,
                          m_min=args.m_min, verbose=False)
        with contextlib.redirect_stdout(io.StringIO()):
            km = KMeansScratch(n_clusters=C, random_state=seed).fit(Xc)
        clusterings = [("SA-AQEE-FCM (ours)", res["labels"], res["V"]),
                       ("K-Means (previous)", km.labels_, km.cluster_centers_.astype(np.float64))]
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_q)
        folds = [order[: n_q // 2], order[n_q // 2:]]
        for name, labels, V in clusterings:
            indices, positions = build_indices(d_emb, labels, C)
            # ---- cross-fitted learned router
            learned = np.zeros((n_q, C))
            for k in (0, 1):
                test_q, train_q = folds[k], folds[1 - k]
                qi = np.array([q for q in train_q for _ in rel[q]], dtype=int)
                y = np.array([labels[d] for q in train_q for d in rel[q]], dtype=int)
                lam = choose_lambda(Qc[qi], y, qi, V, rng)
                params = fit_router(Qc[qi], y, V, lam)
                learned[test_q] = logits(Qc[test_q], V, params)
                print(f"  {name}: fold {k + 1}, {len(qi)} training pairs, ridge penalty {lam:g}", flush=True)
            nearest = -sqdist(Qc, V)
            oracle = np.zeros((n_q, C))
            if args.oracle:
                for i, r in enumerate(rel):
                    if r:
                        oracle[i] = np.bincount(labels[r], minlength=C) * 1e6
                oracle += nearest * 1e-3                       # ties broken by the nearest centroid
            for n in args.n_search:
                evaluate(seed, name, "nearest centroid", n, indices, positions, top_n(nearest, n))
                evaluate(seed, name, "learned", n, indices, positions, top_n(learned, n))
                if args.oracle:
                    evaluate(seed, name, "oracle (upper bound)", n, indices, positions, top_n(oracle, n))
            print(f"  {name} done", flush=True)

    # ---------------------------------------------------------------- mean over seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["router"], r["n"]), []).append(r)
    agg = [dict(method=m, router=ro, n=n, docs=float(np.mean([r["docs"] for r in rs])),
                ndcg10=float(np.mean([r["ndcg10"] for r in rs])), std=float(np.std([r["ndcg10"] for r in rs])),
                recall100=float(np.mean([r["recall100"] for r in rs]))) for (m, ro, n), rs in groups.items()]
    print("\n=== MEAN OVER SEEDS (all queries scored out-of-sample) ===")
    print(f"{'method':20s} {'router':22s} {'clusters':>8s} {'docs searched':>14s} {'% corpus':>8s} {'NDCG@10':>8s} {'(std)':>7s} {'R@100':>7s}")
    for a in sorted(agg, key=lambda a: (a["router"], a["method"], a["n"])):
        print(f"{a['method']:20s} {a['router']:22s} {a['n']:8d} {a['docs']:14,.0f} {100 * a['docs'] / n_docs:7.1f}% "
              f"{a['ndcg10']:8.4f} {a['std']:7.3f} {a['recall100']:7.4f}")

    print("\n=== WHAT THE LEARNED ROUTER ADDS (same method, same number of clusters) ===")
    for method in ("SA-AQEE-FCM (ours)", "K-Means (previous)"):
        for n in args.n_search:
            a0 = next(a for a in agg if a["method"] == method and a["router"] == "nearest centroid" and a["n"] == n)
            a1 = next(a for a in agg if a["method"] == method and a["router"] == "learned" and a["n"] == n)
            print(f"{method:20s} {n:2d} clusters: NDCG@10 {a0['ndcg10']:.4f} -> {a1['ndcg10']:.4f} ({a1['ndcg10'] - a0['ndcg10']:+.4f}), "
                  f"docs searched {a0['docs']:,.0f} -> {a1['docs']:,.0f}")

    print("\n=== SA-AQEE-FCM AGAINST K-MEANS UNDER THE SAME ROUTER, AT THE SAME NUMBER OF DOCUMENTS SEARCHED ===")
    for router in ("nearest centroid", "learned"):
        km_pts = pareto([(a["docs"], a["ndcg10"]) for a in agg if a["method"].startswith("K-Means") and a["router"] == router])
        ed, en = [p[0] for p in km_pts], [p[1] for p in km_pts]
        for a in sorted([a for a in agg if a["method"].startswith("SA-AQEE") and a["router"] == router], key=lambda a: a["n"]):
            if ed[0] <= a["docs"] <= ed[-1]:
                ref = float(np.interp(a["docs"], ed, en))
                print(f"{router:18s} {a['n']:2d} clusters, {a['docs']:9,.0f} docs: ours {a['ndcg10']:.4f}, K-Means at same cost {ref:.4f}, difference {a['ndcg10'] - ref:+.4f}")

    os.makedirs("cache_cross", exist_ok=True)
    name = args.dataset + (str(n_docs) if args.subsample else "")
    path = f"cache_cross/sa_aqee_learned_routing_{name}.json"
    with open(path, "w") as f:
        json.dump({"args": vars(args), "runs": rows, "mean": agg}, f, indent=1, default=float)
    print("\nSaved", path)


if __name__ == "__main__":
    main()
