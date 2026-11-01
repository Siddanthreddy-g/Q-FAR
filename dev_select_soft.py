"""Choose the SA-AQEE-FCM-S spill setting on development queries only.

The queries are SciFact's train split, which no reported result uses, and the vectors come from all-MiniLM-L6-v2, a
different model from the bge-small vectors behind the reported results. Every setting is compared with plain K-Means at
the same number of documents searched (1 to 4 clusters searched; 17 and 64 clusters; seeds 42, 43, 44), and ranked by
the mean NDCG@10 difference over the cluster counts (--criterion ndcg10, the default), or by the mean difference over
all four metrics, NDCG, MAP, Recall and Precision at cutoffs 1, 3, 5, 10 and 100 (--criterion all). Both are printed.
  stage 1: spill budget rho in {0.25, 0.5, 0.75} x rule in {membership, soar (lambda 1)}; a tie within 0.001 goes to the
           smaller rho (smaller index)
  stage 2: SOAR lambda in {0.25, 0.5, 1, 2, 4} at the stage-1 rho, if stage 1 picked the SOAR rule

python dev_select_soft.py                                   # downloads SciFact, embeds it with sentence-transformers
python dev_select_soft.py --data_path D --split test --emb_dir E   # a BEIR folder on disk and precomputed vectors
python dev_select_soft.py --dataset nfcorpus --split dev --ks 64 --criterion all   # NFCorpus's own dev queries
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np

from sa_aqee_soft import (build_spilled_index, fit_sa_aqee_soft, make_probe, pareto, route_scores, search_spilled,
                          spill_entries, unit)

NS = [1, 2, 3, 4, 6, 8]
CUTS = [1, 3, 5, 10, 100]
KEYS = [f"{p}@{k}" for p in ("NDCG", "MAP", "Recall", "P") for k in CUTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--split", default="train", help="the development split (never the test split of a reported run)")
    ap.add_argument("--data_path", default="")
    ap.add_argument("--emb_dir", default="", help="folder with corpus_emb.npy and query_emb.npy for this split")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[17, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--criterion", choices=["ndcg10", "all"], default="ndcg10",
                    help="rank settings by NDCG@10, or by the mean over all four metrics at all cutoffs")
    args = ap.parse_args()

    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from kmean_retrival import KMeansScratch

    if args.data_path:
        path = args.data_path
    else:
        from beir import util
        path = util.download_and_unzip(f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip",
                                       "datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=args.split)
    doc_ids, query_ids = list(corpus), list(queries)
    if args.emb_dir:
        d, q = np.load(f"{args.emb_dir}/corpus_emb.npy"), np.load(f"{args.emb_dir}/query_emb.npy")
    else:
        cache = f"cache_{args.dataset}_{args.split}_{args.model.split('/')[-1]}"
        os.makedirs(cache, exist_ok=True)
        if os.path.exists(f"{cache}/corpus_emb.npy"):
            d, q = np.load(f"{cache}/corpus_emb.npy"), np.load(f"{cache}/query_emb.npy")
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(args.model)
            texts = [(corpus[x].get("title", "") + " " + corpus[x]["text"]).strip() for x in doc_ids]
            d = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
            q = model.encode([queries[x] for x in query_ids], batch_size=64, normalize_embeddings=True)
            np.save(f"{cache}/corpus_emb.npy", d)
            np.save(f"{cache}/query_emb.npy", q)
    assert len(d) == len(doc_ids) and len(q) == len(query_ids), "embeddings do not match this split"
    Xu, Qu = unit(np.asarray(d, np.float64)), unit(np.asarray(q, np.float64))
    d32, q32 = np.ascontiguousarray(d, np.float32), np.ascontiguousarray(q, np.float32)
    print(f"{args.dataset} {args.split}: {len(doc_ids):,} docs, {len(query_ids):,} development queries", flush=True)

    def curve(scores, built):
        out = []
        for n in NS:
            res, scanned = search_spilled(built, q32, np.argsort(-scores, 1)[:, :n], doc_ids, query_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                ndcg, _map, recall, prec = EvaluateRetrieval().evaluate(qrels, res, CUTS)
            m = {**ndcg, **_map, **recall, **prec}
            out.append((n, scanned, [m[k] for k in KEYS]))
        return out

    fits = {}                                                        # (C, seed) -> (K-Means curve, our centroids)
    for C in args.ks:
        for seed in args.seeds:
            with contextlib.redirect_stdout(io.StringIO()):
                km = KMeansScratch(n_clusters=C, max_iters=50, random_state=seed).fit(d32)
            V = km.cluster_centers_.astype(np.float64)
            kmc = curve(route_scores(Qu, V, "cos"), build_spilled_index(d32, np.arange(len(doc_ids)), km.labels_, C))
            probe = make_probe(Xu, np.random.default_rng(seed), "cos")
            fit = fit_sa_aqee_soft(Xu, C, np.random.default_rng(seed), probe, "cos", True, verbose=False)
            fits[(C, seed)] = (kmc, fit["V"])
            print(f"  C={C} seed {seed}: K-Means and SA-AQEE-FCM-S fitted", flush=True)

    def score(rho, rule, lam):
        """Differences against the plain K-Means curve at the same documents searched (n = 1..4), per cluster count:
        NDCG@10 alone, and the mean over all four metrics at all cutoffs (each metric against its own K-Means curve)."""
        per_c_ndcg, per_c_all, wins, total = [], [], 0, 0
        for C in args.ks:
            ours, ref = {}, {}
            for seed in args.seeds:
                kmc, V = fits[(C, seed)]
                ed, ec, _, _ = spill_entries(Xu, V, "cos", rho, rule, lam=lam, rng=np.random.default_rng(seed))
                for n, docs, vals in curve(route_scores(Qu, V, "cos"), build_spilled_index(d32, ed, ec, C)):
                    ours.setdefault(n, []).append([docs] + vals)
                for n, docs, vals in kmc:
                    ref.setdefault(n, []).append([docs] + vals)
            ours = {n: np.mean(v, 0) for n, v in ours.items()}
            ref = [np.mean(v, 0) for v in ref.values()]
            diffs = {}
            for j, key in enumerate(KEYS, start=1):
                env = pareto([(r[0], r[j]) for r in ref])
                ex, ev = [e[0] for e in env], [e[1] for e in env]
                diffs[key] = [o[j] - np.interp(o[0], ex, ev) for n, o in ours.items() if n <= 4 and ex[0] <= o[0] <= ex[-1]]
            per_c_ndcg.append(float(np.mean(diffs["NDCG@10"])))
            per_c_all.append(float(np.mean([np.mean(v) for v in diffs.values()])))
            wins += sum(x > 0 for v in diffs.values() for x in v)
            total += sum(len(v) for v in diffs.values())
        return dict(ndcg10=float(np.mean(per_c_ndcg)), all=float(np.mean(per_c_all)), wins_all=f"{wins}/{total}",
                    per_c_ndcg10=per_c_ndcg, per_c_all=per_c_all)

    crit = args.criterion
    show = lambda r: (f"NDCG@10 {r['ndcg10']:+.4f}  all metrics {r['all']:+.4f}  "        # noqa: E731
                      f"(ahead in {r['wins_all']} metric points)")
    print(f"\nstage 1: spill budget x rule (ranked by {'NDCG@10' if crit == 'ndcg10' else 'all four metrics'})")
    stage1 = []
    for rho in (0.25, 0.5, 0.75):
        for rule in ("membership", "soar"):
            r = score(rho, rule, 1.0)
            stage1.append(dict(rho=rho, rule=rule, lam=1.0, mean=r[crit], **r))
            print(f"  rho {rho:.2f} {rule:10s}  {show(r)}")
    top = max(s["mean"] for s in stage1)
    best = min([s for s in stage1 if s["mean"] >= top - 0.001], key=lambda s: (s["rho"], -s["mean"]))
    stage2 = []
    if best["rule"] == "soar":
        print(f"\nstage 2: SOAR lambda at rho {best['rho']:.2f}")
        for lam in (0.25, 0.5, 1.0, 2.0, 4.0):
            r = score(best["rho"], "soar", lam)
            stage2.append(dict(rho=best["rho"], rule="soar", lam=lam, mean=r[crit], **r))
            print(f"  lambda {lam:<4g}  {show(r)}")
        best = max(stage2, key=lambda s: s["mean"])
    print(f"\nchosen: rho {best['rho']:.2f}, rule {best['rule']}" + (f", lambda {best['lam']:g}" if best["rule"] == "soar" else ""))
    os.makedirs("cache_cross", exist_ok=True)
    out = f"cache_cross/dev_select_soft_{args.dataset}_{args.split}" + ("" if crit == "ndcg10" else f"_{crit}") + ".json"
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), stage1=stage1, stage2=stage2, chosen=best), f, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()
