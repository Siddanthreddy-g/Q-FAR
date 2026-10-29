import argparse
import json
import os
import time

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------------- FCM pieces
def sqdist(X, V):
    d2 = (X * X).sum(1)[:, None] - 2.0 * X @ V.T + (V * V).sum(1)[None, :]
    return np.maximum(d2, EPS)


def memberships(X, V, m):
    """Eq. (6), written as u_ij proportional to d2_ij^(-1/(m-1)) and computed in the log domain."""
    a = -np.log(sqdist(X, V)) / (m - 1.0)
    a -= a.max(1, keepdims=True)
    w = np.exp(a)
    return w / w.sum(1, keepdims=True)


def norm_entropy(U):
    """Eq. (7) and (8): mean membership entropy divided by log C."""
    return float(-(U * np.log(U + EPS)).sum(1).mean() / np.log(U.shape[1]))


def fcm_j(X, V, U, m):
    return float(((U ** m) * sqdist(X, V)).sum())


def xie_beni(X, V, U, m):
    d = ((V[:, None] - V[None]) ** 2).sum(2)
    dmin = d[np.triu_indices(len(V), 1)].min()
    return fcm_j(X, V, U, m) / (len(X) * max(dmin, EPS))


def run_fcm(X, V0, m, max_iters=100, tol=1e-5, adapt=None):
    """Standard FCM from the given centres. With a fixed m this is the plan's final FCM.
    adapt = (eta, h_target, rule, m_min, m_max) keeps updating m with Eq. (9) at every iteration."""
    V = V0.copy()
    for _ in range(max_iters):
        U = memberships(X, V, m)
        W = U ** m
        den = W.sum(0)[:, None]
        Vn = np.where(den > 1e-9, (W.T @ X) / np.maximum(den, EPS), V)     # keep a centre if its cluster is empty
        shift = np.abs(Vn - V).max()
        V = Vn
        if adapt is not None:
            eta, h_target, rule, m_min, m_max = adapt
            m = float(np.clip(m + (1.0 if rule == "paper" else -1.0) * eta * (norm_entropy(U) - h_target), m_min, m_max))
        elif shift < tol:
            break
        if adapt is not None and shift < tol:
            break
    return V, memberships(X, V, m), m


# --------------------------------------------------------------------- quantum chromosomes
class QuantumPopulation:
    """P chromosomes of L qubits. Real amplitudes, alpha^2 + beta^2 = 1 (Eq. 3)."""

    def __init__(self, P, L):
        self.alpha = np.full((P, L), 1.0 / np.sqrt(2.0))
        self.beta = np.full((P, L), 1.0 / np.sqrt(2.0))

    def measure(self, rng):                       # Eq. (4): bit 0 if r <= alpha^2, else 1
        return (rng.random(self.alpha.shape) > self.alpha ** 2).astype(np.uint8)

    def rotate(self, angles):                     # Eq. (14): rotation gate, angles per qubit
        c, s = np.cos(angles), np.sin(angles)
        a, b = self.alpha.copy(), self.beta.copy()
        self.alpha = c * a - s * b
        self.beta = s * a + c * b


def decode(bits, C, D, B, vmin, vmax):
    """Eq. (5): every B-bit segment becomes one coordinate in [vmin, vmax]."""
    P = bits.shape[0]
    z = (bits.reshape(P, C, D, B) * (2 ** np.arange(B - 1, -1, -1))).sum(-1)
    return vmin + z / (2 ** B - 1.0) * (vmax - vmin)


def rotation_angles(pop, bits, best_bits, fit, best_fit, t, G, mode, theta0, theta_max):
    """Angles that turn every qubit that disagrees with the BEST bit towards the BEST bit."""
    P, L = bits.shape
    angles = np.zeros((P, L))
    phi = np.arctan2(pop.beta, pop.alpha)                          # current angle of each qubit
    target = np.where(best_bits[None, :] == 1, np.pi / 2, 0.0)     # angle of |1> or |0>
    direction = np.sign(target - phi)
    differs = bits != best_bits[None, :]
    worse = (fit > best_fit + 1e-12)[:, None]                      # BEST itself is not rotated
    if mode == "fixed":                                            # QEE-FCM: constant step
        mag = np.full((P, L), theta0)
    else:                                                          # self-adaptive step
        gamma = np.abs(pop.beta) / (np.abs(pop.alpha) + EPS)       # Eq. (12)
        p1 = gamma ** 2 / (1.0 + gamma ** 2)                       # probability of measuring 1
        p_best = np.where(best_bits[None, :] == 1, p1, 1.0 - p1)   # probability of measuring the BEST bit
        rng_f = fit.max() - fit.min()
        gap = (fit - best_fit) / (rng_f + EPS)                     # 0 for the best, 1 for the worst
        mag = theta_max * (1.0 - p_best) * (0.2 + 0.8 * gap[:, None]) * (1.0 - 0.9 * t / max(G, 1))
    angles = np.where(differs & worse, direction * np.minimum(mag, np.abs(target - phi)), 0.0)
    return angles


# ------------------------------------------------------------------------- the algorithm
VARIANTS = {
    "V1": dict(evolve=False, rotation=None, adaptive_m=False, label="FCM baseline"),
    "V2": dict(evolve=True, rotation="fixed", adaptive_m=False, label="QEE-FCM"),
    "V3": dict(evolve=True, rotation="sa", adaptive_m=False, label="SA-QIEA rotation"),
    "V4": dict(evolve=True, rotation="fixed", adaptive_m=True, label="QEE-FCM + adaptive m"),
    "V5": dict(evolve=True, rotation="sa", adaptive_m=True, label="SA-AQEE-FCM (full)"),
}


def sa_aqee_fcm(X, C, variant, rng, sample=2000, pop=10, gens=40, bits=6, patience=10,
                m0=2.0, m_min=1.05, m_max=3.0, eta=0.5, h_target=0.5, m_rule="stabilizing",
                fitness="fe", lam=1.0, theta0=0.01 * np.pi, theta_max=0.05 * np.pi, final_adapt=False,
                fit_sample=0, verbose=True):
    cfg = VARIANTS[variant]
    Xall = X                                                       # every document gets labelled at the end
    if fit_sample and fit_sample < len(X):                         # scalability: fit on a sample, label everything
        X = X[rng.choice(len(X), fit_sample, replace=False)]
    N, D = X.shape
    m = m0
    log = []
    t0 = time.time()
    if not cfg["evolve"]:                                          # V1: random data points as start centres
        V0 = X[rng.choice(N, C, replace=False)].copy()
    else:
        Xs = X[rng.choice(N, min(sample, N), replace=False)]
        vmin, vmax = X.min(0), X.max(0)
        total_scatter = ((Xs - Xs.mean(0)) ** 2).sum()
        L = C * D * bits                                           # Eq. (2)
        Q = QuantumPopulation(pop, L)

        def evaluate(V, mm):
            U = memberships(Xs, V, mm)
            f = norm_entropy(U)
            if fitness == "fe_plus_j":
                f += lam * fcm_j(Xs, V, U, mm) / total_scatter
            return f, U

        best_bits, best_V, best_fit, stall = None, None, np.inf, 0
        for g in range(gens):
            B_ = Q.measure(rng)
            Vs = decode(B_, C, D, bits, vmin, vmax)
            fits = np.empty(pop)
            Hn = np.empty(pop)
            for i in range(pop):
                fits[i], U = evaluate(Vs[i], m)
                Hn[i] = norm_entropy(U)
            if best_V is not None:                                 # re-evaluate BEST at the current m
                best_fit, _ = evaluate(best_V, m)
            i_best = int(fits.argmin())
            improved = fits[i_best] < best_fit - 1e-6
            if best_V is None or fits[i_best] < best_fit:
                best_bits, best_V, best_fit = B_[i_best].copy(), Vs[i_best].copy(), float(fits[i_best])
            stall = 0 if improved else stall + 1
            if cfg["adaptive_m"]:                                  # Eq. (9)
                sign = 1.0 if m_rule == "paper" else -1.0
                m = float(np.clip(m + sign * eta * (Hn[i_best] - h_target), m_min, m_max))
            log.append((g, best_fit, m, float(Hn[i_best])))
            angles = rotation_angles(Q, B_, best_bits, fits, best_fit, g, gens, cfg["rotation"], theta0, theta_max)
            Q.rotate(angles)
            if verbose and (g % 10 == 0 or g == gens - 1):
                print(f"    gen {g:3d}  best fitness {best_fit:.4f}  m {m:.3f}  H_norm(best) {Hn[i_best]:.3f}", flush=True)
            if stall >= patience:
                if verbose:
                    print(f"    stopped at generation {g}: no improvement for {patience} generations", flush=True)
                break
        V0 = best_V
    evo_time = time.time() - t0
    adapt = (eta, h_target, m_rule, m_min, m_max) if (final_adapt and cfg["adaptive_m"]) else None
    V, U, m = run_fcm(X, V0, m, adapt=adapt)                       # final FCM with the final fuzzifier m*
    if Xall is not X:                                              # memberships of every document, in chunks
        U = np.vstack([memberships(Xall[i:i + 20000], V, m) for i in range(0, len(Xall), 20000)])
    return dict(V=V, U=U, m=m, labels=U.argmax(1), evo_time=evo_time, log=log, V0=V0)


# ------------------------------------------------------------------------- retrieval test
def cluster_report(X, res, C):
    V, U, m = res["V"], res["U"], res["m"]
    sizes = np.bincount(res["labels"], minlength=C)
    spread = np.linalg.norm(V[:, None] - V[None], axis=2)[np.triu_indices(C, 1)].mean()
    return dict(m=m, H_norm=norm_entropy(U), avg_highest_membership=float(U.max(1).mean()),
                equal_membership=1.0 / C, centroid_spread=float(spread), size_min=int(sizes.min()),
                size_max=int(sizes.max()), xie_beni=xie_beni(X, V, U, m), evo_seconds=res["evo_time"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--pca", type=int, default=0, help="cluster in this many PCA dimensions (0 = full embeddings)")
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--sample", type=int, default=2000, help="documents used for the fitness evaluations")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--bits", type=int, default=6)
    ap.add_argument("--m0", type=float, default=2.0)
    ap.add_argument("--m_min", type=float, default=1.05, help="floor of the fuzzifier; use about 1.02 for 64 clusters")
    ap.add_argument("--m_max", type=float, default=3.0)
    ap.add_argument("--eta", type=float, default=0.5, help="update rate of Eq. 9")
    ap.add_argument("--h_target", type=float, default=0.5, help="target normalised entropy of Eq. 9")
    ap.add_argument("--m_rule", choices=["paper", "stabilizing"], default="stabilizing")
    ap.add_argument("--fitness", choices=["fe", "fe_plus_j"], default="fe")
    ap.add_argument("--final_adapt", action="store_true", help="keep adapting m during the final FCM run (adaptive variants)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="several seeds give mean and std over runs")
    ap.add_argument("--fit_sample", type=int, default=0, help="fit on this many documents, then label all (for large corpora)")
    ap.add_argument("--subsample", type=int, default=0, help="keep this many documents, same selection as the NQ runs")
    ap.add_argument("--emb_dir", default="", help="folder with corpus_emb.npy and query_emb.npy to re-use, e.g. cache")
    args = ap.parse_args()

    from run_our_model_on_their_data import load_beir, get_embeddings, search_clustered
    from beir.retrieval.evaluation import EvaluateRetrieval

    corpus, queries, qrels = load_beir(args.dataset)
    if args.subsample:
        from cross_test_their_models import subsample_corpus
        corpus = subsample_corpus(corpus, qrels, args.subsample)
    doc_ids, query_ids = list(corpus.keys()), list(queries.keys())
    n_docs = len(doc_ids)
    if args.emb_dir:
        d_emb, q_emb = np.load(f"{args.emb_dir}/corpus_emb.npy"), np.load(f"{args.emb_dir}/query_emb.npy")
        assert len(d_emb) == n_docs and len(q_emb) == len(query_ids), "embeddings do not match this corpus"
    else:
        d_emb, q_emb = get_embeddings(corpus, queries, f"cache_{args.dataset}_{n_docs}")
    C = args.k or int(min(64, max(8, n_docs // 300)))
    X, Q = d_emb.astype(np.float64), q_emb.astype(np.float64)
    if args.pca:                                                   # cluster in a smaller space
        mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X[:min(len(X), 5000)] - mu, full_matrices=False)
        P = Vt[:args.pca].T
        Xc, Qc = (X - mu) @ P, (Q - mu) @ P
    else:
        Xc, Qc = X, Q
    print(f"{args.dataset}: {n_docs} docs, clustering space {Xc.shape[1]}-d, C={C}, m rule = {args.m_rule}, "
          f"seeds {args.seeds}", flush=True)

    out, rows = {"args": vars(args)}, []
    for seed in args.seeds:
        for v in args.variants:
            print(f"\n=== seed {seed} | {v}: {VARIANTS[v]['label']} ===", flush=True)
            rng = np.random.default_rng(seed)
            res = sa_aqee_fcm(Xc, C, v, rng, sample=args.sample, pop=args.pop, gens=args.gens, bits=args.bits,
                              m0=args.m0, m_min=args.m_min, m_max=args.m_max, eta=args.eta, h_target=args.h_target,
                              m_rule=args.m_rule, fitness=args.fitness, final_adapt=args.final_adapt,
                              fit_sample=args.fit_sample)
            rep = cluster_report(Xc, res, C)
            print("  cluster check: m*={m:.3f} | H_norm {H_norm:.3f} | avg highest membership {avg_highest_membership:.4f} "
                  "(equal = {equal_membership:.4f}) | sizes min/max {size_min}/{size_max} (balanced = {bal}) | "
                  "Xie-Beni {xie_beni:.4f}".format(bal=n_docs // C, **rep), flush=True)
            route = memberships(Qc, res["V"], res["m"])
            out[f"seed{seed}_{v}"] = {"cluster": rep, "retrieval": {}}
            for ns in args.n_search:
                results, ms, scanned = search_clustered(d_emb, q_emb, res["labels"], route, C, doc_ids, query_ids, n_search=ns)
                ndcg, _m, recall, prec = EvaluateRetrieval().evaluate(qrels, results, [1, 3, 5, 10, 100])
                out[f"seed{seed}_{v}"]["retrieval"][ns] = {"ndcg": ndcg, "recall": recall, "precision": prec, "map": _m,
                                                           "docs_searched": scanned, "pct": 100 * scanned / n_docs}
                rows.append((seed, v, ns, scanned, ndcg["NDCG@10"], recall["Recall@100"]))
    print("\n=== SUMMARY: mean (std) over seeds; compare at a similar number of documents searched ===")
    print(f"{'variant':8s} {'description':24s} {'clusters':>8s} {'docs searched':>14s} {'% corpus':>8s} {'NDCG@10':>15s} {'R@100':>15s}")
    for v in args.variants:
        for ns in args.n_search:
            sel = [r for r in rows if r[1] == v and r[2] == ns]
            sc = np.mean([r[3] for r in sel]); nd = [r[4] for r in sel]; rc = [r[5] for r in sel]
            print(f"{v:8s} {VARIANTS[v]['label']:24s} {ns:8d} {sc:14,.0f} {100 * sc / n_docs:7.1f}% "
                  f"{np.mean(nd):8.4f} ({np.std(nd):.3f}) {np.mean(rc):8.4f} ({np.std(rc):.3f})")
    os.makedirs("cache_cross", exist_ok=True)
    tag = f"{args.m_rule}_pca{args.pca}" + ("_finaladapt" if args.final_adapt else "") + ("_fej" if args.fitness == "fe_plus_j" else "") + (f"_mmin{args.m_min}" if args.m_min != 1.05 else "")
    name = args.dataset + (str(n_docs) if args.subsample else "")
    path = f"cache_cross/sa_aqee_{name}_{tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved", path)


if __name__ == "__main__":
    main()
