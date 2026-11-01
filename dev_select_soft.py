"""Choose the SA-AQEE-FCM-S spill setting on development queries only.

The queries are SciFact's train split, which no reported result uses, and the vectors come from all-MiniLM-L6-v2, a
different model from the bge-small vectors behind the reported results. Every setting is compared with plain K-Means at
the same number of documents searched (1 to 4 clusters searched; 17 and 64 clusters; seeds 42, 43, 44), and ranked by
the mean NDCG@10 difference over both cluster counts.
  stage 1: spill budget rho in {0.25, 0.5, 0.75} x rule in {membership, soar (lambda 1)}; a tie within 0.001 goes to the
           smaller rho (smaller index)
  stage 2: SOAR lambda in {0.25, 0.5, 1, 2, 4} at the stage-1 rho, if stage 1 picked the SOAR rule

python dev_select_soft.py                                   # downloads SciFact, embeds it with sentence-transformers
python dev_select_soft.py --data_path D --split test --emb_dir E   # a BEIR folder on disk and precomputed vectors
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--split", default="train", help="the development split (never the test split of a reported run)")
    ap.add_argument("--data_path", default="")
    ap.add_argument("--emb_dir", default="", help="folder with corpus_emb.npy and query_emb.npy for this split")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ks", type=int, nargs="+", default=[17, 64])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
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
                ndcg = EvaluateRetrieval().evaluate(qrels, res, [10])[0]["NDCG@10"]
            out.append((n, scanned, ndcg))
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
        """Mean NDCG@10 difference against the plain K-Means curve, over both cluster counts (n = 1..4)."""
        per_c = []
        for C in args.ks:
            ours, ref = {}, {}
            for seed in args.seeds:
                kmc, V = fits[(C, seed)]
                ed, ec, _, _ = spill_entries(Xu, V, "cos", rho, rule, lam=lam, rng=np.random.default_rng(seed))
                for n, docs, nd in curve(route_scores(Qu, V, "cos"), build_spilled_index(d32, ed, ec, C)):
                    ours.setdefault(n, []).append((docs, nd))
                for n, docs, nd in kmc:
                    ref.setdefault(n, []).append((docs, nd))
            env = pareto([tuple(np.mean(v, 0)) for v in ref.values()])
            ex, en = [p[0] for p in env], [p[1] for p in env]
            diffs = [nd - np.interp(dc, ex, en) for n, v in ours.items() if n <= 4
                     for dc, nd in [tuple(np.mean(v, 0))] if ex[0] <= dc <= ex[-1]]
            per_c.append(float(np.mean(diffs)))
        return float(np.mean(per_c)), per_c

    print("\nstage 1: spill budget x rule")
    stage1 = []
    for rho in (0.25, 0.5, 0.75):
        for rule in ("membership", "soar"):
            m, per_c = score(rho, rule, 1.0)
            stage1.append(dict(rho=rho, rule=rule, lam=1.0, mean=m, per_c=per_c))
            print(f"  rho {rho:.2f} {rule:10s}  mean NDCG@10 difference {m:+.4f}  per cluster count {np.round(per_c, 4)}")
    top = max(s["mean"] for s in stage1)
    best = min([s for s in stage1 if s["mean"] >= top - 0.001], key=lambda s: (s["rho"], -s["mean"]))
    stage2 = []
    if best["rule"] == "soar":
        print(f"\nstage 2: SOAR lambda at rho {best['rho']:.2f}")
        for lam in (0.25, 0.5, 1.0, 2.0, 4.0):
            m, per_c = score(best["rho"], "soar", lam)
            stage2.append(dict(rho=best["rho"], rule="soar", lam=lam, mean=m, per_c=per_c))
            print(f"  lambda {lam:<4g}  mean NDCG@10 difference {m:+.4f}  per cluster count {np.round(per_c, 4)}")
        best = max(stage2, key=lambda s: s["mean"])
    print(f"\nchosen: rho {best['rho']:.2f}, rule {best['rule']}" + (f", lambda {best['lam']:g}" if best["rule"] == "soar" else ""))
    os.makedirs("cache_cross", exist_ok=True)
    out = f"cache_cross/dev_select_soft_{args.dataset}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(dict(args=vars(args), stage1=stage1, stage2=stage2, chosen=best), f, indent=1)
    print("Saved", out)


if __name__ == "__main__":
    main()
