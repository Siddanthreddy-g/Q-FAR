"""Paired significance tests over queries for every comparison in the paper.

The other scripts save only mean scores. This one re-runs, with the same data, seeds and settings, the methods the paper
compares with the proposed method (BAFC) and keeps every query's NDCG@10, MAP@10, Recall@100 and P@10. The random draws
of sa_aqee_soft.py are replayed in the same order, so the re-run reproduces its numbers; the script checks this against
cache_cross/compare_methods_<name>.json and stops with a message if they differ.

For each budget of the proposed method (1 to 4 clusters searched) every other method is read at the same number of
documents compared, as in compare_methods.py, but query by query: each query's value is interpolated between the two
settings of the other method that bracket that cost, with the same weight for every query, so the mean over queries is
the number in the paper's tables. Each mean difference then gets
  * a 95% confidence interval and a two-sided p-value from a paired bootstrap over queries (10,000 resamples),
  * a Wilcoxon signed-rank p-value,
  * Holm's correction over each method's four budgets, and over all comparisons of the dataset (NDCG@10).
Per-query scores are kept in cache_cross/per_query_<name>.npz, so later tests need no re-run.

Two additions answer reviewer questions:
  * "BAFC, random border documents": the same partition and the same number of copies, but the documents that get a
    copy are drawn at random instead of by the fuzzy gate. It tells whether the gate picks better documents than chance.
  * Index size (MB) and build time for every method.

python significance.py --dataset scifact
python significance.py --dataset nfcorpus --k 64
python significance.py --dataset nq --subsample 200000 --emb_dir cache --k 64
Run after sa_aqee_soft.py and compare_methods.py for the same dataset.
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

from sa_aqee_soft import (build_spilled_index, d2_to, fit_sa_aqee_soft, make_probe, pareto, route_scores, search_spilled,
                          soar_cluster, spill_entries, unit)
from sa_aqee_fcm import sqdist

MEAS = ("ndcg_cut_10", "map_cut_10", "recall_100", "P_10")
NAMES = ("NDCG@10", "MAP@10", "Recall@100", "P@10")
P = "BAFC"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def bootstrap(d, rng, B=10000, chunk=1000):
    """Paired bootstrap of the mean of d (n,) or (n, k): 95% CI and two-sided p-value (null-centred)."""
    d = np.asarray(d, float)
    n = len(d)
    means = []
    for s in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - s), n))
        means.append(d[idx].mean(1))
    means = np.concatenate(means)
    obs = d.mean(0)
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    p = ((np.abs(means - obs) >= np.abs(obs)).sum(0) + 1) / (len(means) + 1)
    return obs, lo, hi, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data_path", default="")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--emb_dir", default="")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--with_fcm", action="store_true", help="also FCM (m = 2); slow on NQ and never reachable there")
    ap.add_argument("--skip_hnsw", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--no_check", action="store_true", help="do not stop if the re-run differs from the paper's numbers")
    args = ap.parse_args()

    import faiss
    import pytrec_eval
    from kmean_retrival import KMeansScratch

    t0 = time.time()

    def say(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # ---- the same data as sa_aqee_soft.py and compare_methods.py
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
    N, D = d_emb.shape
    X, Q = d_emb.astype(np.float64), q_emb.astype(np.float64)
    Xu, Qu = unit(X), unit(Q)
    d32, q32 = np.ascontiguousarray(d_emb, dtype=np.float32), np.ascontiguousarray(q_emb, dtype=np.float32)

    soft = json.load(open(f"cache_cross/sa_aqee_soft_{name}.json"))
    ref = json.load(open(f"cache_cross/compare_methods_{name}.json"))
    a = soft["args"]
    C = soft["C"]
    assert soft["n_docs"] == N and (not args.k or args.k == C), "data or --k differ from the sa_aqee_soft.py run"
    rho, rule, lam, tau = a["default_rho"], a["default_rule"], a["soar_lambda"], a["tau"]
    configs = [(0.0, "none")] + [(r, ru) for r in a["rhos"] for ru in a["rules"]]
    NS = a["n_search"]
    say(f"{name}: {N:,} docs, {len(query_ids):,} queries, C={C}; BAFC setting rho {rho}, {rule}, lambda {lam}")

    # ---- per-query evaluation (pytrec_eval, as inside BEIR's evaluator)
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "map_cut.10", "recall.100", "P.10"})
    qeval = [q for q in query_ids if q in qrels]

    def per_query(results):
        results = {q: {d: s for d, s in r.items() if d != q} for q, r in results.items()}   # BEIR ignores identical ids
        sc = evaluator.evaluate(results)
        return np.array([[sc[q][m] if q in sc else 0.0 for m in MEAS] for q in qeval], dtype=np.float32)

    def spilled_at(V, seed, target):
        """spill_entries for config `target`, with the rng advanced exactly as sa_aqee_soft.py's run_spills does."""
        rng = np.random.default_rng(seed)
        for r_, ru in configs:
            if (r_, ru) == target:
                return spill_entries(Xu, V, "cos", r_, ru if r_ > 0 else "membership", tau, lam, rng)
            rng.choice(N, min(20000, N), replace=False)
        raise ValueError(f"{target} is not among {configs}")

    curves = {}           # method -> list of (setting label, seed, docs, per-query array)
    build = {}            # method -> list of (seconds, megabytes)

    def search_all(method, built, scores, seed, label=""):
        for n in NS:
            res, scanned = search_spilled(built, q32, np.argsort(-scores, 1)[:, :n], doc_ids, query_ids)
            curves.setdefault(method, []).append((f"{label}{n}", seed, scanned, per_query(res)))

    def mb(entries):
        return (entries * (D * 4 + 8) + C * D * 4) / 2 ** 20

    for seed in args.seeds:
        say(f"=== seed {seed} ===")
        probe = make_probe(Xu, np.random.default_rng(seed), "cos", a["fit_sample"], a["n_pseudo"])
        # K-Means, both routings
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()):
            km = KMeansScratch(n_clusters=C, max_iters=a["km_iters"], random_state=seed).fit(d_emb)
        Vk = km.cluster_centers_.astype(np.float64)
        built = build_spilled_index(d32, np.arange(N), km.labels_, C)
        km_time = time.time() - t
        build.setdefault("K-Means", []).append((km_time, mb(N)))
        search_all("K-Means", built, route_scores(Q, Vk, "cos"), seed, "ip")
        search_all("K-Means", built, -sqdist(Q, Vk), seed, "l2")
        for meth, target in (("K-Means + same soft assignment", (rho, rule)), ("SOAR", (1.0, "soar"))):
            t = time.time()
            ed, ec, _, info = spilled_at(Vk, seed, target)
            built = build_spilled_index(d32, ed, ec, C)
            build.setdefault(meth, []).append((km_time + time.time() - t, mb(len(ed))))
            search_all(meth, built, route_scores(Qu, Vk, "cos"), seed)
        say("K-Means, K-Means + same soft assignment and SOAR searched")
        # BAFC and BAFC without the evolution
        for meth, use_qiea in ((P, True), ("BAFC w/o evolution", False)):
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                fit = fit_sa_aqee_soft(Xu, C, np.random.default_rng(seed), probe, "cos", use_qiea, a["m_fit"],
                                       pop=a["pop"], gens=a["gens"], refine_iters=a["refine_iters"], verbose=False)
            V = fit["V"]
            ed, ec, (c1, c2), info = spilled_at(V, seed, (rho, rule))
            built = build_spilled_index(d32, ed, ec, C)
            build.setdefault(meth, []).append((time.time() - t, mb(len(ed))))
            search_all(meth, built, route_scores(Qu, V, "cos"), seed)
            if use_qiea:
                # same partition and number of copies, border documents drawn at random instead of by the gate
                n_copy = int((c2 >= 0).sum())
                pick = np.sort(np.random.default_rng(seed + 1000).choice(N, n_copy, replace=False))
                c2r = np.full(N, -1)
                for s in range(0, len(pick), 20000):
                    rows = pick[s:s + 20000]
                    c2r[rows] = soar_cluster(Xu[rows], V, d2_to(Xu[rows], V, "cos"), c1[rows], "cos", lam)
                sp = c2r >= 0
                built = build_spilled_index(d32, np.concatenate([np.arange(N), np.where(sp)[0]]),
                                            np.concatenate([c1, c2r[sp]]), C)
                search_all("BAFC, random border documents", built, route_scores(Qu, V, "cos"), seed)
            say(f"{meth} searched (index x{info['index_x']:.2f})")
        # FAISS IVF
        t = time.time()
        quant = faiss.IndexFlatIP(D)
        ivf = faiss.IndexIVFFlat(quant, D, C, faiss.METRIC_INNER_PRODUCT)
        ivf.cp.seed = seed
        ivf.train(d32)
        ivf.add(d32)
        build.setdefault("FAISS IVF", []).append((time.time() - t, mb(N)))
        sizes = np.array([ivf.invlists.list_size(c) for c in range(C)])
        for n in NS:
            ivf.nprobe = n
            Dd, Ii = ivf.search(q32, 100)
            _, L = quant.search(q32, n)
            res = {qid: {doc_ids[j]: float(s) for j, s in zip(Ii[i], Dd[i]) if j >= 0} for i, qid in enumerate(query_ids)}
            curves.setdefault("FAISS IVF", []).append((str(n), seed, float(sizes[L].sum(1).mean()), per_query(res)))
        if args.with_fcm:
            from fuzzycmeans_scratch import FuzzyCMeansScratch
            t = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                fcm = FuzzyCMeansScratch(n_clusters=C, m=2.0, max_iters=20, chunk_size=2000, random_state=seed).fit(d32)
            Vf = fcm.cluster_centers_.astype(np.float32)
            built = build_spilled_index(d32, np.arange(N), fcm.labels_, C)
            build.setdefault("Fuzzy C-Means (m = 2)", []).append((time.time() - t, mb(N)))
            d2q = (q32 ** 2).sum(1)[:, None] - 2 * q32 @ Vf.T + (Vf ** 2).sum(1)[None]
            search_all("Fuzzy C-Means (m = 2)", built, -d2q, seed)
        say("FAISS IVF" + (" and FCM" if args.with_fcm else "") + " searched")

    if not args.skip_hnsw:
        t = time.time()
        hn = faiss.IndexHNSWFlat(D, 32, faiss.METRIC_INNER_PRODUCT)
        hn.hnsw.efConstruction = 200
        hn.add(d32)
        build["HNSW"] = [(time.time() - t, faiss.serialize_index(hn).size / 2 ** 20)]
        for ef in (16, 32, 64, 128, 256, 512, 1024, 2048):
            hn.hnsw.efSearch = ef
            faiss.cvar.hnsw_stats.reset()
            Dd, Ii = hn.search(q32, 100)
            ndis = faiss.cvar.hnsw_stats.ndis / len(q32)
            res = {qid: {doc_ids[j]: float(s) for j, s in zip(Ii[i], Dd[i]) if j >= 0} for i, qid in enumerate(query_ids)}
            curves.setdefault("HNSW", []).append((str(ef), 0, float(ndis), per_query(res)))
        say("HNSW searched")

    # ---- mean over seeds per setting: (docs, per-query mean over seeds)
    settings = {}
    for meth, rows in curves.items():
        by = {}
        for label, seed, docs, pq in rows:
            by.setdefault(label, []).append((docs, pq))
        settings[meth] = sorted([(float(np.mean([d for d, _ in v])), np.mean([p for _, p in v], 0), lab)
                                 for lab, v in by.items()], key=lambda s: s[0])
    prop = {int(lab): (docs, pq) for docs, pq, lab in settings[P]}

    # reproduction check against the paper's numbers (compare_methods JSON)
    ref_p = {q["n"]: q["metrics"] for q in ref["curves"]["Proposed method (SA-AQEE-FCM-S)"]}
    worst = max(abs(float(prop[n][1][:, 0].mean()) - ref_p[n]["NDCG@10"]) for n in (1, 2, 3, 4))
    say(f"reproduction check: largest NDCG@10 difference from the paper's BAFC numbers {worst:.5f}")
    if worst > 0.002 and not args.no_check:
        raise SystemExit("the re-run does not reproduce the paper's numbers (data or settings differ); "
                         "pass --no_check to test it anyway")

    def read_at(meth, docs, tol=0.02):
        """Per-query values of `meth` at `docs` documents, on its NDCG@10 envelope (None if not reachable)."""
        pts = settings[meth]
        env = pareto([(d, pq[:, 0].mean()) for d, pq, _ in pts])
        keep = [next(p for p in pts if p[0] == e[0] and abs(p[1][:, 0].mean() - e[1]) < 1e-12) for e in env]
        xs = [p[0] for p in keep]
        if docs < xs[0] * (1 - tol) or docs > xs[-1] * (1 + tol):
            return None
        if docs <= xs[0]:
            return keep[0][1]
        if docs >= xs[-1]:
            return keep[-1][1]
        j = int(np.searchsorted(xs, docs))
        w = (docs - xs[j - 1]) / (xs[j] - xs[j - 1])
        return (1 - w) * keep[j - 1][1] + w * keep[j][1]

    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)
    tests = []
    for meth in settings:
        if meth == P:
            continue
        for n in (1, 2, 3, 4):
            docs, pq = prop[n]
            other = read_at(meth, docs)
            if other is None:
                continue
            obs, lo, hi, p = bootstrap(pq - other, rng, args.bootstrap)
            wil = []
            for j in range(4):
                dj = pq[:, j] - other[:, j]
                try:
                    wil.append(float(wilcoxon(dj, zero_method="zsplit").pvalue) if np.any(dj != 0) else 1.0)
                except ValueError:
                    wil.append(1.0)
            tests.append(dict(method=meth, n=n, docs=docs, bafc=pq.mean(0).tolist(), other=other.mean(0).tolist(),
                              diff=obs.tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist(), p_boot=p.tolist(),
                              p_wilcoxon=wil))
    adj = holm([t["p_boot"][0] for t in tests])                     # over every row of the dataset
    for t, h in zip(tests, adj):
        t["p_holm_all_ndcg10"] = float(h)
    for meth in {t["method"] for t in tests}:                          # within each method (its 1-4 cluster budgets)
        fam = [t for t in tests if t["method"] == meth]
        for t, h in zip(fam, holm([t["p_boot"][0] for t in fam])):
            t["p_holm_ndcg10"] = float(h)

    print(f"\n{name}: BAFC minus each method at equal documents compared, NDCG@10 (mean over {len(qeval)} queries)")
    print(f"  {'method':32s} {'n':>2s} {'docs':>8s} {'diff':>8s} {'95% CI':>19s} {'p boot':>8s} {'p Wilc.':>8s} "
          f"{'Holm/m':>8s} {'Holm/all':>8s}")
    for t in sorted(tests, key=lambda t: (t["method"], t["n"])):
        sig = "*" if t["p_holm_ndcg10"] < 0.05 else ""
        print(f"  {t['method']:32s} {t['n']:2d} {t['docs']:8,.0f} {t['diff'][0]:+8.4f} "
              f"[{t['ci_lo'][0]:+.4f}, {t['ci_hi'][0]:+.4f}] {t['p_boot'][0]:8.4f} {t['p_wilcoxon'][0]:8.4f} "
              f"{t['p_holm_ndcg10']:8.4f} {t['p_holm_all_ndcg10']:8.4f} {sig}")
    print("  * significant at 0.05 after Holm's correction over the method's budgets (Holm/m); "
          "Holm/all corrects over every row")
    print("\nIndex size and build time (mean over seeds):")
    for meth, v in build.items():
        print(f"  {meth:32s} {np.mean([x for x, _ in v]):8.1f} s  {np.mean([m for _, m in v]):8.1f} MB")

    os.makedirs("cache_cross", exist_ok=True)
    out = f"cache_cross/significance_{name}.json"
    json.dump(dict(args=vars(args), name=name, n_queries=len(qeval), metrics=NAMES, setting=dict(rho=rho, rule=rule,
                   soar_lambda=lam), reproduction_max_abs_diff=worst, tests=tests,
                   build={m: dict(seconds=float(np.mean([x for x, _ in v])), megabytes=float(np.mean([y for _, y in v])))
                          for m, v in build.items()}), open(out, "w"), indent=1)
    np.savez_compressed(f"cache_cross/per_query_{name}.npz", queries=np.array(qeval),
                        **{f"{m}|{lab}": pq for m, pts in settings.items() for _, pq, lab in pts},
                        **{f"{m}|{lab}|docs": np.array(d) for m, pts in settings.items() for d, _, lab in pts})
    say(f"saved {out} and cache_cross/per_query_{name}.npz")


if __name__ == "__main__":
    main()
