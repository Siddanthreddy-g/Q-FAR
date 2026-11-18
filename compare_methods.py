"""Every retrieval method of the project on the same data, measured the same way.

sa_aqee_soft.py already runs K-Means, SA-AQEE-FCM V5, K-Means with SOAR spilling and the proposed method
(SA-AQEE-FCM-S). This script adds, on the same corpus, queries, relevance labels and bge-small vectors:
  * exhaustive search (the ceiling),
  * Fuzzy C-Means with m = 2 (fuzzycmeans_scratch.py, the earlier method, 20 iterations as in kmeans_retrieval.py),
  * FAISS IVF (Johnson et al.) with as many lists as there are clusters,
  * HNSW (Malkov and Yashunin, M = 32, efConstruction = 200),
and reads the earlier result files for the rest, where they exist: K-Means with the learned router
(sa_aqee_learned_routing.py) and Murali and Damodaram (2015) (pfcm_murali_baseline.py).

Every method is scored with BEIR's evaluator on NDCG, MAP, Recall and Precision at 1, 3, 5, 10 and 100. The cost of a
query is the number of document vectors it is compared with: documents scanned for the cluster methods and IVF (copies
included), distance computations for HNSW. The methods are compared at the proposed method's own costs (1 to 4
clusters searched): every other method is read off its best-quality-per-cost curve at that cost, by linear
interpolation between its measured settings. A method whose cheapest setting already costs more is marked as not
reachable, and one that was not measured that far is left blank.

python compare_methods.py --dataset scifact
python compare_methods.py --dataset nfcorpus --k 64
python compare_methods.py --dataset nq --subsample 200000 --emb_dir cache --k 64
Run sa_aqee_soft.py for the dataset first: its result file supplies K-Means, V5, SOAR and the proposed method.
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

from sa_aqee_soft import build_spilled_index, pareto, search_spilled

CUTS = [1, 3, 5, 10, 100]
KEYS = [f"{p}@{k}" for p in ("NDCG", "MAP", "Recall", "P") for k in CUTS]
HEAD = ["NDCG@10", "MAP@10", "Recall@100", "P@10"]
PROPOSED = "Proposed method (SA-AQEE-FCM-S)"


def flat(m):
    """One dict of all 20 metric values from a row that holds ndcg/map/recall/precision dicts."""
    return {k: float(v) for part in ("ndcg", "map", "recall", "precision") for k, v in m[part].items() if k in KEYS}


def mean_curve(rows):
    """Average per-seed rows that share a setting n: [(n, docs, metrics, ndcg10_std)] sorted by docs."""
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n"], []).append(r)
    out = []
    for n, rs in by_n.items():
        ms = [flat(r) for r in rs]
        out.append(dict(n=n, docs=float(np.mean([r["docs"] for r in rs])),
                        metrics={k: float(np.mean([m[k] for m in ms])) for k in KEYS},
                        ndcg10_std=float(np.std([m["NDCG@10"] for m in ms])), seeds=len(rs)))
    return sorted(out, key=lambda p: p["docs"])


def at_cost(points, budget, key, tol=0.02):
    """Value of `key` on the method's best-quality-per-cost curve at `budget` documents compared.
    Returns (value, status) with status 'ok', 'not reachable' (cheapest setting costs more) or 'not measured'."""
    env = pareto([(p["docs"], p["metrics"][key]) for p in points])
    xs, ys = [e[0] for e in env], [e[1] for e in env]
    if budget < xs[0] * (1 - tol):
        return None, "not reachable"
    if budget > xs[-1] * (1 + tol):
        return None, "not measured"
    return float(np.interp(budget, xs, ys)), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data_path", default="", help="a BEIR-format folder to load instead of downloading")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--emb_dir", default="", help="folder with corpus_emb.npy and query_emb.npy, e.g. cache")
    ap.add_argument("--k", type=int, default=0, help="clusters; must match the sa_aqee_soft.py run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--fcm_m", type=float, default=2.0)
    ap.add_argument("--fcm_iters", type=int, default=20)
    ap.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--hnsw_m", type=int, default=32)
    ap.add_argument("--skip_fcm", action="store_true")
    ap.add_argument("--skip_hnsw", action="store_true")
    args = ap.parse_args()

    import faiss
    from beir.retrieval.evaluation import EvaluateRetrieval
    from fuzzycmeans_scratch import FuzzyCMeansScratch

    t_start = time.time()

    def say(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    # ---- the same data as sa_aqee_soft.py
    if args.data_path:
        from beir.datasets.data_loader import GenericDataLoader
        corpus, queries, qrels = GenericDataLoader(data_folder=args.data_path).load(split="test")
    else:
        from run_our_model_on_their_data import load_beir
        corpus, queries, qrels = load_beir(args.dataset)
    if args.subsample:
        from cross_test_their_models import subsample_corpus
        corpus = subsample_corpus(corpus, qrels, args.subsample)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    if args.emb_dir:
        d_emb, q_emb = np.load(f"{args.emb_dir}/corpus_emb.npy"), np.load(f"{args.emb_dir}/query_emb.npy")
    else:
        from run_our_model_on_their_data import get_embeddings
        d_emb, q_emb = get_embeddings(corpus, queries, f"cache_{args.dataset}_{len(doc_ids)}")
    assert len(d_emb) == len(doc_ids) and len(q_emb) == len(query_ids), "embeddings do not match this corpus"
    name = args.dataset + (str(len(doc_ids)) if args.subsample else "")
    n_docs, dim = len(doc_ids), d_emb.shape[1]
    d32, q32 = np.ascontiguousarray(d_emb, dtype=np.float32), np.ascontiguousarray(q_emb, dtype=np.float32)

    soft_path = f"cache_cross/sa_aqee_soft_{name}.json"
    if not os.path.exists(soft_path):
        raise SystemExit(f"{soft_path} is missing: run sa_aqee_soft.py for this dataset first")
    soft = json.load(open(soft_path))
    assert soft["n_docs"] == n_docs, f"{soft_path} was made on {soft['n_docs']:,} documents, this corpus has {n_docs:,}"
    C = soft["C"]
    if args.k and args.k != C:
        raise SystemExit(f"--k {args.k} does not match the {C} clusters of {soft_path}")
    sa = soft["args"]
    rho, rule = sa["default_rho"], sa["default_rule"]
    say(f"{name}: {n_docs:,} docs, {len(query_ids):,} queries, {C} clusters; proposed setting rho {rho}, {rule}, "
        f"lambda {sa['soar_lambda']} (from {soft_path})")

    def evaluate(results):
        with contextlib.redirect_stdout(io.StringIO()):
            ndcg, _map, recall, prec = EvaluateRetrieval().evaluate(qrels, results, CUTS)
        return dict(ndcg=ndcg, map=_map, recall=recall, precision=prec)

    def to_results(D, I):
        return {qid: {doc_ids[j]: float(s) for j, s in zip(I[i], D[i]) if j >= 0} for i, qid in enumerate(query_ids)}

    methods, notes = {}, {}

    # ---- from the sa_aqee_soft.py run (identical data, seeds and evaluation)
    runs = soft["runs"]
    pick = lambda f: [r for r in runs if f(r)]                                         # noqa: E731
    methods["K-Means"] = mean_curve(pick(lambda r: r["method"] == "K-Means (previous)" and r["route"] == "ip")) + \
        mean_curve(pick(lambda r: r["method"] == "K-Means (previous)" and r["route"] == "l2"))
    methods["K-Means"].sort(key=lambda p: p["docs"])
    notes["K-Means"] = "both routings (inner product and L2) kept; the better one counts at each cost"
    v5 = pick(lambda r: r["method"].startswith("SA-AQEE-FCM V5"))
    if v5:
        methods["SA-AQEE-FCM V5 (the plan as written)"] = mean_curve(v5)
    soar = pick(lambda r: r["method"] == "K-Means + same spill" and r["rho"] == 1.0 and r["rule"] == "soar")
    if soar:
        methods["SOAR (K-Means + a second copy of every document)"] = mean_curve(soar)
        notes["SOAR (K-Means + a second copy of every document)"] = \
            f"Sun et al. (2023) spilling on K-Means clusters, lambda {sa['soar_lambda']}, index 2x"
    methods["K-Means + the same fuzzy-gated soft assignment"] = mean_curve(
        pick(lambda r: r["method"] == "K-Means + same spill" and r["rho"] == rho and r["rule"] == rule))
    noq = pick(lambda r: r["method"] == "SA-AQEE-FCM-S without QIEA" and r["rho"] == rho and r["rule"] == rule)
    if noq:
        methods["Proposed method without the evolution"] = mean_curve(noq)
    methods[PROPOSED] = mean_curve(
        pick(lambda r: r["method"].startswith("SA-AQEE-FCM-S (") and r["rho"] == rho and r["rule"] == rule))
    idx_x = float(np.mean([r["index_x"] for r in runs if r["method"].startswith("SA-AQEE-FCM-S (")
                           and r["rho"] == rho and r["rule"] == rule]))

    # ---- learned router (stage 6), where it was run on this dataset
    lr_path = f"cache_cross/sa_aqee_learned_routing_{name}.json"
    if os.path.exists(lr_path):
        lr = json.load(open(lr_path))
        rows = [r for r in lr["runs"] if r["method"] == "K-Means (previous)" and r["router"] == "learned"]
        if rows:
            methods["K-Means + learned router"] = mean_curve(rows)
            notes["K-Means + learned router"] = f"from {lr_path} (seeds {lr['args']['seeds']}, router cross-fitted on " \
                                                "half of the queries at a time)"

    # ---- exhaustive search
    flat_ix = faiss.IndexFlatIP(dim)
    flat_ix.add(d32)
    D, I = flat_ix.search(q32, 100)
    m = evaluate(to_results(D, I))
    methods["Exhaustive search (ceiling)"] = [dict(n=0, docs=float(n_docs), metrics=flat(m), ndcg10_std=0.0, seeds=1)]
    say(f"exhaustive search: NDCG@10 {m['ndcg']['NDCG@10']:.4f}")

    # ---- Fuzzy C-Means, m = 2 (the earlier method): documents go to their strongest membership, queries to the
    #      clusters of their strongest memberships (the nearest centroids)
    if not args.skip_fcm:
        rows, collapse = [], []
        for seed in args.seeds:
            say(f"seed {seed}: fitting Fuzzy C-Means (k={C}, m={args.fcm_m}, {args.fcm_iters} iterations)")
            with contextlib.redirect_stdout(io.StringIO()):
                fcm = FuzzyCMeansScratch(n_clusters=C, m=args.fcm_m, max_iters=args.fcm_iters, chunk_size=2000,
                                         random_state=seed).fit(d32)
            V = fcm.cluster_centers_.astype(np.float32)
            sizes = np.bincount(fcm.labels_, minlength=C)
            collapse.append(dict(seed=seed, mean_top_membership=float(fcm.u_.max(1).mean()),
                                 flat_split=1.0 / C, largest_cluster=int(sizes.max()), empty=int((sizes == 0).sum())))
            built = build_spilled_index(d32, np.arange(n_docs), fcm.labels_, C)
            d2 = (q32 ** 2).sum(1)[:, None] - 2 * q32 @ V.T + (V ** 2).sum(1)[None]
            for n in args.n_search:
                res, scanned = search_spilled(built, q32, np.argsort(d2, 1)[:, :n], doc_ids, query_ids)
                rows.append(dict(seed=seed, n=n, docs=scanned, **evaluate(res)))
            say(f"  mean top membership {collapse[-1]['mean_top_membership']:.3f} (flat split {1 / C:.3f}), "
                f"largest cluster {sizes.max():,} of {n_docs:,}")
            del built, fcm
        methods["Fuzzy C-Means (m = 2)"] = mean_curve(rows)
        notes["Fuzzy C-Means (m = 2)"] = dict(collapse=collapse)

    # ---- FAISS IVF with as many lists as clusters (its own k-means training, default settings, one run per seed)
    rows = []
    for seed in args.seeds:
        quant = faiss.IndexFlatIP(dim)
        ivf = faiss.IndexIVFFlat(quant, dim, C, faiss.METRIC_INNER_PRODUCT)
        ivf.cp.seed = seed
        ivf.train(d32)
        ivf.add(d32)
        sizes = np.array([ivf.invlists.list_size(c) for c in range(C)])
        for n in args.n_search:
            ivf.nprobe = n
            D, I = ivf.search(q32, 100)
            _, L = quant.search(q32, n)
            rows.append(dict(seed=seed, n=n, docs=float(sizes[L].sum(1).mean()), **evaluate(to_results(D, I))))
    methods["FAISS IVF"] = mean_curve(rows)
    notes["FAISS IVF"] = f"IndexIVFFlat, inner product, nlist = {C}, nprobe = {args.n_search}"
    say("FAISS IVF searched")

    # ---- HNSW: cost = distance computations per query (one build)
    if not args.skip_hnsw:
        say(f"building HNSW (M={args.hnsw_m}, efConstruction=200)")
        hn = faiss.IndexHNSWFlat(dim, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        hn.hnsw.efConstruction = 200
        hn.add(d32)
        rows = []
        for ef in args.ef:
            hn.hnsw.efSearch = ef
            faiss.cvar.hnsw_stats.reset()
            D, I = hn.search(q32, 100)
            ndis = faiss.cvar.hnsw_stats.ndis / len(q32)
            rows.append(dict(seed=0, n=ef, docs=float(ndis), **evaluate(to_results(D, I))))
        methods["HNSW"] = mean_curve(rows)
        notes["HNSW"] = f"IndexHNSWFlat, M = {args.hnsw_m}, efSearch = {args.ef}; cost = distance computations; " \
                        f"the graph adds about {2 * args.hnsw_m * 4} bytes of links per document"
        say("HNSW searched")

    # ---- Murali and Damodaram (2015), where it was run on this corpus (a word-feature method; no cost axis)
    mu_path = f"cache_cross/pfcm_murali_{args.dataset}_{n_docs}.json"
    murali = None
    if os.path.exists(mu_path):
        murali = flat(json.load(open(mu_path)))

    # ---- every method at the proposed method's costs (1 to 4 clusters searched)
    budgets = [p for p in methods[PROPOSED] if p["n"] <= 4]
    table, wins = [], {}
    for p in budgets:
        for mname, pts in methods.items():
            if mname == PROPOSED:
                continue
            vals, status = {}, set()
            for key in KEYS:
                v, s = at_cost(pts, p["docs"], key) if mname != "Exhaustive search (ceiling)" else \
                    (pts[0]["metrics"][key], "ok")
                vals[key] = v
                status.add(s)
            st = "ok" if status == {"ok"} else sorted(status - {"ok"})[0]
            table.append(dict(n=p["n"], docs=p["docs"], method=mname, status=st, metrics=vals,
                              proposed=p["metrics"]))
            if st == "ok" and mname != "Exhaustive search (ceiling)":
                w = wins.setdefault(mname, [0, 0])
                w[0] += sum(p["metrics"][k] > vals[k] for k in KEYS)
                w[1] += len(KEYS)

    # ---- print
    print(f"\n{name}: every method at the proposed method's costs (index x{idx_x:.2f} for the proposed method)")
    for p in budgets:
        print(f"\n{p['n']} cluster(s) searched by the proposed method = {p['docs']:,.0f} documents compared "
              f"({100 * p['docs'] / n_docs:.1f}% of the corpus)")
        print(f"  {'method':52s} " + " ".join(f"{h:>10s}" for h in HEAD))
        print(f"  {PROPOSED:52s} " + " ".join(f"{p['metrics'][h]:10.4f}" for h in HEAD))
        for t in table:
            if t["n"] != p["n"]:
                continue
            if t["status"] != "ok":
                print(f"  {t['method']:52s} {t['status']}")
                continue
            print(f"  {t['method']:52s} " + " ".join(f"{t['metrics'][h]:10.4f}" for h in HEAD))
    if murali:
        print(f"\n  Murali and Damodaram (2015), whole collection: " +
              " ".join(f"{h} {murali[h]:.4f}" for h in HEAD))
    print("\nProposed method ahead, over all four metrics at cutoffs 1, 3, 5, 10, 100 and the compared costs:")
    for mname, (w, n) in wins.items():
        print(f"  vs {mname:52s} {w}/{n}")
    if "Fuzzy C-Means (m = 2)" in notes:
        c = notes["Fuzzy C-Means (m = 2)"]["collapse"]
        print(f"\nFuzzy C-Means: mean top membership {np.mean([x['mean_top_membership'] for x in c]):.3f} "
              f"(flat split {1 / C:.3f}); largest cluster {np.mean([x['largest_cluster'] for x in c]):,.0f} documents")
    print("\nEach method's own measured settings (mean over seeds):")
    for mname, pts in methods.items():
        print(f"  {mname}")
        for q in pts:
            print(f"    setting {q['n']:>4}  docs {q['docs']:>10,.0f}  " +
                  "  ".join(f"{h} {q['metrics'][h]:.4f}" for h in HEAD))

    os.makedirs("cache_cross", exist_ok=True)
    out = f"cache_cross/compare_methods_{name}.json"
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), name=name, n_docs=n_docs, C=C, proposed_setting=dict(rho=rho, rule=rule,
                       soar_lambda=sa["soar_lambda"], index_x=idx_x), curves=methods, notes=notes,
                       murali=murali, at_proposed_costs=table, wins=wins), f, indent=1)
    say(f"saved {out}")


if __name__ == "__main__":
    main()
