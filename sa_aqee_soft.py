"""SA-AQEE-FCM-S: SA-AQEE-FCM with soft (multi-cluster) assignment of boundary documents.

What changes against SA-AQEE-FCM V5 (sa_aqee_fcm.py), and why:
  1. Spherical FCM on the full corpus (chunked). Text embeddings are unit vectors, so centroids are re-normalised after
     every update and queries are routed by inner product, the same rule as the K-Means baseline. V5 fitted a 30k
     sample and routed by Euclidean distance to unnormalised centroids.
  2. The quantum-inspired evolution searches data-anchored seeds: every cluster slot is an index into a k-means++
     candidate pool, so a chromosome has C * log2(pool) qubits (512 for C = 64) instead of C * D * bits (147,456).
     The fitness is label-free and retrieval-aligned: the share of each pseudo-query's exact neighbours that lie in
     the clusters it would search, at a fixed number of documents searched. The first BEST is the k-means++ seeding,
     so evolution can only improve on the K-Means start.
  3. The self-adaptive fuzzifier (Eq. 9 form) now sets how fuzzy the document assignment is. m_assign is adapted
     until the share of documents whose second membership is at least tau times their first equals the budget rho.
  4. Those boundary documents get a second copy: in their second-highest-membership cluster ("membership"), or in
     the cluster chosen by the SOAR loss of Sun et al. (NeurIPS 2023), "soar". At most two copies per document.

Cost is counted fairly: a copy scanned twice counts twice in "documents searched", and the index size is reported.

Default setting (rho = 0.5, SOAR rule, lambda = 0.5) was chosen on development queries only, with dev_select_soft.py:
the SciFact train queries, which no reported result uses, embedded with all-MiniLM-L6-v2.
"""
import argparse
import contextlib
import io
import json
import os
import time

import numpy as np

from sa_aqee_fcm import QuantumPopulation, rotation_angles, sqdist

EPS = 1e-12


# ------------------------------------------------------------------------------ geometry
def unit(V):
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), EPS)


def d2_to(X, V, geometry):
    """Squared distances (n, C). 'cos' assumes unit-norm X and uses the unit centroids."""
    if geometry == "cos":
        return np.maximum(2.0 - 2.0 * X @ unit(V).T, EPS)
    return sqdist(X, V)


def route_scores(Q, V, geometry):
    """Higher = better cluster for the query (IP to unit centroids, or minus the squared L2 distance)."""
    return Q @ unit(V).T if geometry == "cos" else -sqdist(Q, V)


def memberships_from_d2(d2, m):
    a = -np.log(d2) / (m - 1.0)
    a -= a.max(1, keepdims=True)
    w = np.exp(a)
    return w / w.sum(1, keepdims=True)


# ------------------------------------------------------------------------------ FCM
def run_fcm_chunked(X, V0, m, geometry="cos", max_iters=50, tol=1e-5, chunk=20000):
    """FCM over the whole of X in chunks; spherical (unit centroids) when geometry == 'cos'."""
    V = V0.astype(np.float64).copy()
    if geometry == "cos":
        V = unit(V)
    for _ in range(max_iters):
        num, den = np.zeros_like(V), np.zeros(len(V))
        for s in range(0, len(X), chunk):
            xb = X[s:s + chunk]
            W = memberships_from_d2(d2_to(xb, V, geometry), m) ** m
            num += W.T @ xb
            den += W.sum(0)
        Vn = np.where(den[:, None] > 1e-9, num / np.maximum(den, EPS)[:, None], V)   # keep an empty centre
        if geometry == "cos":
            Vn = unit(Vn)
        shift = np.abs(Vn - V).max()
        V = Vn
        if shift < tol:
            break
    return V


def hard_labels(X, V, geometry, chunk=20000):
    return np.concatenate([d2_to(X[s:s + chunk], V, geometry).argmin(1) for s in range(0, len(X), chunk)])


# ------------------------------------------------------------------- soft assignment (spill)
def log_ratio_and_top2(d2):
    """For each row: nearest cluster, second-nearest cluster and log(d1^2 / d2^2) <= 0."""
    top = np.argpartition(d2, 1, axis=1)[:, :2]
    a, b = top[:, 0], top[:, 1]
    r = np.arange(len(d2))
    swap = d2[r, a] > d2[r, b]
    c1, c2 = np.where(swap, b, a), np.where(swap, a, b)
    return c1, c2, np.log(d2[r, c1] / d2[r, c2])


def spill_gate(log_r, m, tau):
    """Second membership >= tau * first membership. With u_j proportional to d_j^(-2/(m-1)) this is
    log(d1^2/d2^2) / (m - 1) >= log(tau), which admits more documents as m grows."""
    if not np.isfinite(m):
        return np.ones(len(log_r), dtype=bool)
    return log_r / (m - 1.0) >= np.log(tau)


def adapt_m_assign(log_r, rho, tau=0.5, m0=2.0, eta=1.0, m_lo=1.001, m_hi=50.0, steps=300):
    """Eq. (9)-style self-adaptive fuzzifier: m <- clip(m + eta * (rho - spill share)), eta halved on a sign change."""
    if rho <= 0:
        return 1.0, 0.0
    if rho >= 1:
        return float("inf"), 1.0
    m, prev = m0, None
    for _ in range(steps):
        frac = float(spill_gate(log_r, m, tau).mean())
        err = rho - frac
        if abs(err) < 0.002:
            break
        if prev is not None and np.sign(err) != np.sign(prev):
            eta *= 0.5
        m, prev = float(np.clip(m + eta * err, m_lo, m_hi)), err
    return m, float(spill_gate(log_r, m, tau).mean())


def soar_cluster(X, V, d2, c1, geometry, lam=1.0):
    """argmin over j != c1 of ||x - v_j||^2 + lam * <x - v_c1, x - v_j>^2 / ||x - v_c1||^2 (Sun et al., 2023)."""
    Vg = unit(V) if geometry == "cos" else V
    xv = X @ Vg.T
    xx = (X * X).sum(1)
    G = Vg @ Vg.T
    r = np.arange(len(X))
    dot = xx[:, None] - xv - xv[r, c1][:, None] + G[c1]              # <x - v_c1, x - v_j>
    loss = d2 + lam * dot ** 2 / np.maximum(d2[r, c1], EPS)[:, None]
    loss[r, c1] = np.inf
    return loss.argmin(1)


def assign(X, V, geometry, m_assign, tau=0.5, rule="membership", lam=1.0, chunk=20000):
    """Primary cluster c1 for every document, and a second cluster c2 (-1 if the document is not spilled)."""
    c1s, c2s = [], []
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        d2 = d2_to(xb, V, geometry)
        c1, c2, log_r = log_ratio_and_top2(d2)
        if m_assign <= 1.0:
            c2 = np.full(len(xb), -1)
        else:
            if rule == "soar":
                c2 = soar_cluster(xb, V, d2, c1, geometry, lam)
            c2 = np.where(spill_gate(log_r, m_assign, tau), c2, -1)
        c1s.append(c1)
        c2s.append(c2)
    return np.concatenate(c1s), np.concatenate(c2s)


def spill_entries(X, V, geometry, rho, rule="membership", tau=0.5, lam=1.0, rng=None, adapt_sample=20000):
    """Adapt m_assign to the budget rho on a sample, then assign every document. Returns entries and info."""
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(X), min(adapt_sample, len(X)), replace=False)
    _, _, log_r = log_ratio_and_top2(d2_to(X[idx], V, geometry))
    m_assign, _ = adapt_m_assign(log_r, rho, tau)
    c1, c2 = assign(X, V, geometry, m_assign, tau, rule, lam)
    spilled = c2 >= 0
    entry_doc = np.concatenate([np.arange(len(X)), np.where(spilled)[0]])
    entry_cl = np.concatenate([c1, c2[spilled]])
    info = dict(rho=rho, rule=rule, m_assign=m_assign, spill_share=float(spilled.mean()),
                index_x=len(entry_doc) / len(X))
    return entry_doc, entry_cl, (c1, c2), info


# --------------------------------------------------------------- label-free retrieval proxy
def make_probe(X, rng, geometry, fit_sample=20000, n_pseudo=1000, n_nbrs=10):
    """A fit sample, pseudo-queries drawn from it and each pseudo-query's exact neighbours inside it (self excluded)."""
    s_idx = rng.choice(len(X), min(fit_sample, len(X)), replace=False)
    Xs = X[s_idx]
    pq = rng.choice(len(Xs), min(n_pseudo, len(Xs)), replace=False)
    S = route_scores(Xs[pq], Xs, geometry)                           # similarity of pseudo-queries to sample docs
    S[np.arange(len(pq)), pq] = -np.inf
    nbrs = np.argpartition(-S, n_nbrs, axis=1)[:, :n_nbrs]
    return dict(sample_idx=s_idx, Xs=Xs, pq=pq, nbrs=nbrs)


def capture(probe, V, c1, c2, geometry, budgets=(1.0, 2.0)):
    """Mean share of each pseudo-query's neighbours found when clusters are scanned in routing order until
    b * n / C entries have been scanned (the last cluster counted fractionally). Higher is better."""
    Xs, pq, nb = probe["Xs"], probe["pq"], probe["nbrs"]
    C = len(V)
    order = np.argsort(-route_scores(Xs[pq], V, geometry), axis=1)
    sizes = np.bincount(c1, minlength=C) + np.bincount(c2[c2 >= 0], minlength=C)
    sz = sizes[order].astype(np.float64)
    before = np.cumsum(sz, 1) - sz
    rows = np.arange(len(pq))[:, None]
    n1, n2 = c1[nb], c2[nb]
    caps = []
    for b in budgets:
        w_rank = np.clip((b * len(Xs) / C - before) / np.maximum(sz, 1.0), 0.0, 1.0)
        w = np.empty_like(w_rank)
        w[rows, order] = w_rank                                       # weight of every cluster for every query
        g = np.maximum(w[rows, n1], np.where(n2 >= 0, w[rows, np.maximum(n2, 0)], 0.0))
        caps.append(g.mean())
    return float(np.mean(caps))


# ------------------------------------------------------------------------ the algorithm
def kpp_pool(Xs, n, rng, geometry):
    """k-means++ D^2 sampling of n points; the first C of them are an ordinary k-means++ seeding."""
    chosen = [int(rng.integers(len(Xs)))]
    closest = np.full(len(Xs), np.inf)
    for _ in range(1, n):
        closest = np.minimum(closest, d2_to(Xs, Xs[chosen[-1]][None], geometry)[:, 0])
        p = np.maximum(closest, 0.0)
        p[chosen] = 0.0
        chosen.append(int(rng.choice(len(Xs), p=p / p.sum())))
    return np.array(chosen)


def decode_slots(bits, C, b, P):
    """Every b-bit segment is an index into the candidate pool; duplicate picks move to the next unused candidate."""
    z = (bits.reshape(len(bits), C, b) * (2 ** np.arange(b - 1, -1, -1))).sum(-1) % P
    for row in z:
        used = set()
        for j in range(C):
            while int(row[j]) in used:
                row[j] = (row[j] + 1) % P
            used.add(int(row[j]))
    return z


def encode_slots(idx, b):
    return ((np.asarray(idx)[:, None] >> np.arange(b - 1, -1, -1)) & 1).astype(np.uint8).ravel()


def fit_sa_aqee_soft(X, C, rng, probe, geometry="cos", use_qiea=True, m_fit=1.05, pool_factor=4, pop=12, gens=25,
                     patience=8, refine_iters=3, final_iters=50, theta0=0.01 * np.pi, theta_max=0.05 * np.pi,
                     verbose=True):
    """Returns the final centroids (fitted on all of X), the seed centroids and the evolution log."""
    t0 = time.time()
    Xs = probe["Xs"]
    P = pool_factor * C
    pool = kpp_pool(Xs, P, rng, geometry)
    b = int(np.ceil(np.log2(P)))

    def fitness(slot_idx):
        V = run_fcm_chunked(Xs, Xs[pool[slot_idx]], m_fit, geometry, max_iters=refine_iters)
        c1 = hard_labels(Xs, V, geometry)
        return 1.0 - capture(probe, V, c1, np.full(len(Xs), -1), geometry)

    best_idx = np.arange(C)                                          # the plain k-means++ seeding
    best_bits = encode_slots(best_idx, b)
    best_fit = fitness(best_idx)
    log = [(-1, best_fit)]
    if use_qiea:
        Q = QuantumPopulation(pop, C * b)
        stall = 0
        for g in range(gens):
            B_ = Q.measure(rng)
            idx = decode_slots(B_.copy(), C, b, P)
            fits = np.array([fitness(idx[i]) for i in range(pop)])
            i_best = int(fits.argmin())
            improved = fits[i_best] < best_fit - 1e-6
            if improved:
                best_idx, best_bits, best_fit = idx[i_best].copy(), encode_slots(idx[i_best], b), float(fits[i_best])
            stall = 0 if improved else stall + 1
            log.append((g, best_fit))
            Q.rotate(rotation_angles(Q, B_, best_bits, fits, best_fit, g, gens, "sa", theta0, theta_max))
            if verbose and (g % 5 == 0 or g == gens - 1):
                print(f"    gen {g:3d}  best fitness {best_fit:.4f} (k-means++ seed {log[0][1]:.4f})", flush=True)
            if stall >= patience:
                if verbose:
                    print(f"    stopped at generation {g}: no improvement for {patience} generations", flush=True)
                break
    evo_time = time.time() - t0
    V0 = Xs[pool[best_idx]]
    V = run_fcm_chunked(X, V0, m_fit, geometry, max_iters=final_iters)
    return dict(V=V, V0=V0, log=log, evo_time=evo_time, seed_fitness=log[0][1], best_fitness=best_fit)


# ------------------------------------------------------------------------------ searching
def build_spilled_index(d32, entry_doc, entry_cl, C, metric="ip"):
    import faiss
    order = np.argsort(entry_cl, kind="stable")
    bounds = np.searchsorted(entry_cl[order], np.arange(C + 1))
    members, indices = [], []
    for c in range(C):
        mem = entry_doc[order[bounds[c]:bounds[c + 1]]]
        members.append(mem)
        ix = None
        if len(mem):
            ix = faiss.IndexFlatIP(d32.shape[1]) if metric == "ip" else faiss.IndexFlatL2(d32.shape[1])
            ix.add(d32[mem])
        indices.append(ix)
    return members, indices


def search_spilled(built, q32, chosen, doc_ids, query_ids, top_k=100, metric="ip"):
    """chosen: (n_queries, n) clusters to search. One faiss call per cluster for all queries that probe it.
    Duplicate copies are merged BEFORE the top-k cut. Returns results and the mean entries scanned per query."""
    members, indices = built
    nq = len(q32)
    cand_d, cand_s = [[] for _ in range(nq)], [[] for _ in range(nq)]
    scanned = np.zeros(nq)
    for c, (mem, ix) in enumerate(zip(members, indices)):
        qs = np.where((chosen == c).any(1))[0]
        if not len(qs) or ix is None:
            continue
        scanned[qs] += len(mem)
        S, L = ix.search(q32[qs], min(top_k, len(mem)))
        if metric != "ip":
            S = -S
        for row, qi in enumerate(qs):
            cand_d[qi].append(mem[L[row]])
            cand_s[qi].append(S[row])
    results = {}
    for qi, qid in enumerate(query_ids):
        if not cand_d[qi]:
            results[qid] = {}
            continue
        dd, ss = np.concatenate(cand_d[qi]), np.concatenate(cand_s[qi])
        o = np.argsort(-ss, kind="stable")
        dd, ss = dd[o], ss[o]
        _, first = np.unique(dd, return_index=True)
        keep = np.sort(first)[:top_k]
        results[qid] = {doc_ids[j]: float(s) for j, s in zip(dd[keep], ss[keep])}
    return results, float(scanned.mean())


def pareto(points):
    """Upper envelope of (docs, quality) points: the best quality reachable at each cost."""
    best, env = -np.inf, []
    for d, n in sorted(points):
        if n > best:
            best = n
            env.append((d, n))
    return env


# --------------------------------------------------------------------------- synthetic data
def synthetic_data(n_docs=20000, n_queries=1000, dim=64, n_topics=48, seed=0):
    """Pipeline check only: an anisotropic mixture on the sphere; each query is a shifted, noisy copy of one document
    (its single relevant document). Numbers from this mode are NOT evidence about real retrieval."""
    rng = np.random.default_rng(seed)
    centers = unit(rng.normal(size=(n_topics, dim)))
    counts = rng.multinomial(n_docs, rng.dirichlet(np.full(n_topics, 2.0)))
    X = []
    for t in range(n_topics):
        A = rng.normal(size=(dim, 6)) * 0.25                         # a few strong directions per topic
        X.append(centers[t] + rng.normal(size=(counts[t], 6)) @ A.T + 0.08 * rng.normal(size=(counts[t], dim)))
    X = unit(np.vstack(X))
    X = X[rng.permutation(len(X))]
    shift = unit(rng.normal(size=(1, dim))) * 0.25                   # a common query-document gap
    tgt = rng.choice(len(X), n_queries, replace=False)
    Q = unit(X[tgt] + shift + 0.25 * rng.normal(size=(n_queries, dim)) / np.sqrt(dim) * 4)
    doc_ids = [f"d{i}" for i in range(len(X))]
    query_ids = [f"q{i}" for i in range(n_queries)]
    qrels = {query_ids[i]: {doc_ids[t]: 1} for i, t in enumerate(tgt)}
    return doc_ids, query_ids, qrels, X.astype(np.float32), Q.astype(np.float32)


# ------------------------------------------------------------------------------------ main
OURS, NOQ, KM, KMS, V5 = ("SA-AQEE-FCM-S (ours)", "SA-AQEE-FCM-S without QIEA", "K-Means (previous)",
                          "K-Means + same spill", "SA-AQEE-FCM V5 (previous)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--data_path", default="", help="a BEIR-format folder to load instead of downloading")
    ap.add_argument("--synthetic", action="store_true", help="pipeline check on synthetic data (not evidence)")
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--emb_dir", default="", help="folder with corpus_emb.npy and query_emb.npy, e.g. cache")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="spill budgets (share of docs)")
    ap.add_argument("--rules", nargs="+", default=["membership", "soar"], choices=["membership", "soar"])
    ap.add_argument("--default_rho", type=float, default=0.5, help="the method's spill budget, fixed in advance")
    ap.add_argument("--default_rule", default="soar", choices=["membership", "soar"])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--soar_lambda", type=float, default=0.5, help="SOAR weight (0.5 chosen on development queries)")
    ap.add_argument("--m_fit", type=float, default=1.05)
    ap.add_argument("--fit_sample", type=int, default=20000, help="sample for the evolution and the proxy")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--refine_iters", type=int, default=3, help="FCM iterations before each fitness evaluation")
    ap.add_argument("--n_pseudo", type=int, default=1000, help="pseudo-queries for the evolution's fitness")
    ap.add_argument("--km_iters", type=int, default=50, help="K-Means iterations (the previous scripts used 20; 50 is a stronger baseline)")
    ap.add_argument("--skip_v5", action="store_true")
    ap.add_argument("--v5_fit_sample", type=int, default=0, help="30000 reproduces the published NQ run")
    ap.add_argument("--m_min", type=float, default=1.05, help="V5 only: floor of its fuzzifier")
    ap.add_argument("--skip_noqiea", action="store_true", help="skip the without-QIEA ablation")
    args = ap.parse_args()

    from beir.retrieval.evaluation import EvaluateRetrieval
    from kmean_retrival import KMeansScratch

    if args.synthetic:
        doc_ids, query_ids, qrels, d_emb, q_emb = synthetic_data()
        name = "synthetic"
        print("SYNTHETIC DATA: pipeline check only, the numbers are not evidence about real retrieval", flush=True)
    else:
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
    n_docs = len(doc_ids)
    C = args.k or int(min(64, max(8, n_docs // 300)))
    X, Q = d_emb.astype(np.float64), q_emb.astype(np.float64)
    Xu, Qu = unit(X), unit(Q)                                        # bge vectors are unit already: a no-op there
    d32, q32 = np.ascontiguousarray(d_emb, dtype=np.float32), np.ascontiguousarray(q_emb, dtype=np.float32)
    print(f"{name}: {n_docs:,} docs, {len(query_ids):,} queries, C={C}, seeds {args.seeds}, "
          f"spill budgets {args.rhos} x {args.rules}", flush=True)

    rows, infos = [], []
    t_start = time.time()

    def say(msg):
        print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)

    def evaluate(seed, method, route, rho, rule, built, scores, index_x):
        for n in args.n_search:
            chosen = np.argsort(-scores, axis=1)[:, :n]
            results, scanned = search_spilled(built, q32, chosen, doc_ids, query_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                ndcg, _m, recall, prec = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
            rows.append(dict(seed=seed, method=method, route=route, rho=rho, rule=rule, n=n, docs=scanned,
                             index_x=index_x, ndcg10=ndcg["NDCG@10"], recall100=recall["Recall@100"],
                             ndcg=ndcg, recall=recall, precision=prec, map=_m))

    def run_spills(seed, method, V, rng):
        """rho = 0 plus every (rho, rule) for one set of centroids."""
        scores = route_scores(Qu, V, "cos")
        configs = [(0.0, "none")] + [(r, ru) for r in args.rhos for ru in args.rules]
        say(f"{method}: searching {len(configs)} spill settings x {len(args.n_search)} cluster counts")
        for rho, rule in configs:
            ed, ec, _, info = spill_entries(Xu, V, "cos", rho, rule if rho > 0 else "membership", args.tau,
                                            args.soar_lambda, rng)
            info.update(seed=seed, method=method, rule=rule)
            infos.append(info)
            built = build_spilled_index(d32, ed, ec, C)
            evaluate(seed, method, "ip", rho, rule, built, scores, info["index_x"])
            del built
            say(f"  {method:28s} rho {rho:4.2f} {rule:10s} m_assign {info['m_assign']:7.3f} "
                f"spilled {info['spill_share']:.3f} index x{info['index_x']:.2f}")

    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        rng = np.random.default_rng(seed)
        probe = make_probe(Xu, rng, "cos", args.fit_sample, args.n_pseudo)
        # ---- K-Means, as in the previous scripts, routed both ways; its spill uses the identical rule
        say(f"seed {seed}: fitting K-Means (k={C}, up to {args.km_iters} iterations, one line per iteration)")
        km = KMeansScratch(n_clusters=C, max_iters=args.km_iters, random_state=seed).fit(d_emb)
        Vk = km.cluster_centers_.astype(np.float64)
        lab = km.labels_
        built = build_spilled_index(d32, np.arange(n_docs), lab, C)
        evaluate(seed, KM, "ip", 0.0, "none", built, route_scores(Q, Vk, "cos"), 1.0)
        evaluate(seed, KM, "l2", 0.0, "none", built, -sqdist(Q, Vk), 1.0)
        del built
        say(f"K-Means fitted and searched (sizes min/max {np.bincount(lab, minlength=C).min()}/"
            f"{np.bincount(lab, minlength=C).max()})")
        run_spills(seed, KMS, Vk, np.random.default_rng(seed))
        # ---- the previous SA-AQEE-FCM, for continuity
        if not args.skip_v5:
            from sa_aqee_fcm import sa_aqee_fcm
            say(f"seed {seed}: fitting SA-AQEE-FCM V5 (the previous method)")
            res = sa_aqee_fcm(X, C, "V5", np.random.default_rng(seed), sample=2000, fit_sample=args.v5_fit_sample,
                              m_min=args.m_min, verbose=False)
            built = build_spilled_index(d32, np.arange(n_docs), res["labels"], C)
            evaluate(seed, V5, "l2", 0.0, "none", built, -sqdist(Q, res["V"]), 1.0)
            del built
            say(f"V5 fitted and searched (m* {res['m']:.3f})")
        # ---- ours, and ours without the evolution
        for method, use_qiea in ((OURS, True), (NOQ, False)):
            if method == NOQ and args.skip_noqiea:
                continue
            say(f"seed {seed}: fitting {method}" + (" (evolution, then FCM on all documents)" if use_qiea else ""))
            fit = fit_sa_aqee_soft(Xu, C, np.random.default_rng(seed), probe, "cos", use_qiea, args.m_fit,
                                   pop=args.pop, gens=args.gens, refine_iters=args.refine_iters, verbose=use_qiea)
            say(f"{method} fitted: seed fitness {fit['seed_fitness']:.4f} -> {fit['best_fitness']:.4f} "
                f"(evolution {fit['evo_time']:.0f}s)")
            run_spills(seed, method, fit["V"], np.random.default_rng(seed))

    # ---------------------------------------------------------------- mean over seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["method"], r["route"], r["rho"], r["rule"], r["n"]), []).append(r)
    agg = []
    for (method, route, rho, rule, n), rs in groups.items():
        agg.append(dict(method=method, route=route, rho=rho, rule=rule, n=n,
                        docs=float(np.mean([r["docs"] for r in rs])), index_x=float(np.mean([r["index_x"] for r in rs])),
                        ndcg10=float(np.mean([r["ndcg10"] for r in rs])), ndcg10_std=float(np.std([r["ndcg10"] for r in rs])),
                        recall100=float(np.mean([r["recall100"] for r in rs]))))
    order_m = {KM: 0, V5: 1, KMS: 2, NOQ: 3, OURS: 4}
    print("\n=== MEAN OVER SEEDS ===")
    print(f"{'method':28s} {'route':5s} {'rho':>4s} {'rule':10s} {'n':>2s} {'docs searched':>13s} {'% corpus':>8s} "
          f"{'index':>5s} {'NDCG@10':>8s} {'(std)':>6s} {'R@100':>7s}")
    for a in sorted(agg, key=lambda a: (order_m[a["method"]], a["route"], a["rho"], a["rule"], a["n"])):
        print(f"{a['method']:28s} {a['route']:5s} {a['rho']:4.2f} {a['rule']:10s} {a['n']:2d} {a['docs']:13,.0f} "
              f"{100 * a['docs'] / n_docs:7.1f}% x{a['index_x']:4.2f} {a['ndcg10']:8.4f} {a['ndcg10_std']:6.3f} "
              f"{a['recall100']:7.4f}")

    def envelope(sel, key):
        env = pareto([(a["docs"], a[key]) for a in agg if sel(a)])
        return [p[0] for p in env], [p[1] for p in env]

    def versus(title, ours_sel, ref_sel_for):
        """Each point of ours against the reference curve at the same documents searched (inside its range only)."""
        print(f"\n=== {title} ===")
        print(f"{'method':28s} {'rho':>4s} {'rule':10s} {'n':>2s} {'docs':>9s} {'NDCG@10':>8s} {'ref':>7s} {'diff':>8s} "
              f"{'R@100':>7s} {'ref':>7s} {'diff':>8s}")
        out, wins, total = [], 0, 0
        for a in sorted([a for a in agg if ours_sel(a)], key=lambda a: (order_m[a["method"]], a["rho"], a["rule"], a["n"])):
            ref_sel = ref_sel_for(a)
            ed, en = envelope(ref_sel, "ndcg10")
            rd, rr = envelope(ref_sel, "recall100")
            if not ed or not (ed[0] <= a["docs"] <= ed[-1]):
                continue
            n_ref, r_ref = float(np.interp(a["docs"], ed, en)), float(np.interp(a["docs"], rd, rr))
            total += 1
            wins += a["ndcg10"] > n_ref
            out.append(dict(method=a["method"], rho=a["rho"], rule=a["rule"], n=a["n"], docs=a["docs"],
                            ndcg10=a["ndcg10"], ref_ndcg10=n_ref, recall100=a["recall100"], ref_recall100=r_ref))
            print(f"{a['method']:28s} {a['rho']:4.2f} {a['rule']:10s} {a['n']:2d} {a['docs']:9,.0f} {a['ndcg10']:8.4f} "
                  f"{n_ref:7.4f} {a['ndcg10'] - n_ref:+8.4f} {a['recall100']:7.4f} {r_ref:7.4f} {a['recall100'] - r_ref:+8.4f}")
        print(f"points above the reference curve (NDCG@10): {wins} of {total}")
        return dict(points=out, wins=wins, total=total)

    is_km = lambda a: a["method"] == KM                               # noqa: E731  (both routings)
    h2h = {"vs_kmeans": versus("OURS AGAINST THE BEST K-MEANS CURVE (either routing) AT THE SAME DOCS SEARCHED",
                               lambda a: a["method"] == OURS, lambda a: is_km)}
    h2h["kmeans_spill_vs_kmeans"] = versus("K-MEANS + SAME SPILL AGAINST PLAIN K-MEANS (how much the spill alone gives)",
                                           lambda a: a["method"] == KMS and a["rho"] > 0, lambda a: is_km)
    h2h["vs_kmeans_spill"] = versus(
        "ATTRIBUTION: OURS AGAINST K-MEANS WITH THE IDENTICAL SPILL (same rho and rule)",
        lambda a: a["method"] == OURS,
        lambda a: (lambda b, a=a: b["method"] == KMS and b["rho"] == a["rho"] and b["rule"] == a["rule"]))
    if not args.skip_noqiea:
        h2h["vs_no_qiea"] = versus(
            "ATTRIBUTION: OURS AGAINST OURS WITHOUT THE QIEA (same rho and rule)",
            lambda a: a["method"] == OURS,
            lambda a: (lambda b, a=a: b["method"] == NOQ and b["rho"] == a["rho"] and b["rule"] == a["rule"]))

    # ------------- headline: the default spill setting, fixed in advance (the sweep above is the sensitivity analysis)
    dr, du = args.default_rho, args.default_rule
    if any(a["method"] == OURS and a["rho"] == dr and a["rule"] == du for a in agg):
        h2h["default_vs_kmeans"] = versus(
            f"HEADLINE: OURS WITH THE PRE-SET DEFAULT (rho {dr:.2f}, {du}) AGAINST THE BEST K-MEANS CURVE",
            lambda a: a["method"] == OURS and a["rho"] == dr and a["rule"] == du, lambda a: is_km)

    os.makedirs("cache_cross", exist_ok=True)
    path = f"cache_cross/sa_aqee_soft_{name}.json"
    with open(path, "w") as f:
        json.dump({"args": vars(args), "n_docs": n_docs, "C": C, "runs": rows, "mean": agg, "spill_info": infos,
                   "head_to_head": h2h}, f, indent=1, default=float)
    print("\nSaved", path)


if __name__ == "__main__":
    main()
