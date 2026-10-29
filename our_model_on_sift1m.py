import argparse
import json
import os
import tarfile
import time
import urllib.request

import numpy as np

SIFT_URLS = ["http://corpus-texmex.irisa.fr/sift.tar.gz",
             "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"]
MIN_TGZ_BYTES = 100_000_000          # the real file is about 168 MB
BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

MANUAL = """
Could not download SIFT1M automatically. Do it by hand (about 168 MB), then run this script again:

  Option A (PowerShell, in the 4-1 folder):
      mkdir datasets\\sift1m
      curl.exe -L -C - -A "Mozilla/5.0" -o datasets\\sift1m\\sift.tar.gz http://corpus-texmex.irisa.fr/sift.tar.gz

  Option B: paste  http://corpus-texmex.irisa.fr/sift.tar.gz  into your browser, save the file as
      datasets\\sift1m\\sift.tar.gz   (inside the 4-1 folder)

The script then unpacks the file itself. (-C - lets curl resume if the connection drops.)
"""


# --------------------------------------------------------------------------- data
def read_fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def read_ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = int(a[0])
    return a.reshape(-1, d + 1)[:, 1:].copy()


def find_file(root, name):
    for dp, _, files in os.walk(root):
        if name in files:
            return os.path.join(dp, name)
    return None


def download(url, dest):
    """Download with a browser-like User-Agent (some servers answer 403 to the default Python one)."""
    req = urllib.request.Request(url, headers=BROWSER_HEADERS)
    part = dest + ".part"
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        total = 0
        try:
            total = int(r.headers.get("Content-Length", 0) or 0)
        except Exception:
            pass
        done, last = 0, time.time()
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 10:
                print(f"   downloaded {done / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else ""), flush=True)
                last = time.time()
    os.replace(part, dest)


def get_sift(data_dir, urls=None):
    if find_file(data_dir, "sift_base.fvecs") is None:
        os.makedirs(data_dir, exist_ok=True)
        tgz = os.path.join(data_dir, "sift.tar.gz")
        if os.path.exists(tgz) and os.path.getsize(tgz) >= MIN_TGZ_BYTES:
            print("Found", tgz, "- unpacking it", flush=True)
        else:
            ok = False
            for url in (urls or SIFT_URLS):
                print(f"Downloading SIFT1M from {url} (about 168 MB, can take a while)...", flush=True)
                try:
                    download(url, tgz)
                    ok = True
                    break
                except Exception as e:
                    print(f"   failed: {e}", flush=True)
            if not ok:
                raise SystemExit(MANUAL)
        with tarfile.open(tgz) as t:
            try:
                t.extractall(data_dir, filter="data")
            except TypeError:
                t.extractall(data_dir)
    return (read_fvecs(find_file(data_dir, "sift_base.fvecs")),
            read_fvecs(find_file(data_dir, "sift_query.fvecs")),
            read_ivecs(find_file(data_dir, "sift_groundtruth.ivecs")))


def exact_neighbours(base, queries, k=100, chunk=100):
    """Exact L2 nearest neighbours (needed when the base set is a subset, so the supplied
    ground truth no longer applies)."""
    b2 = (base * base).sum(1)[None, :]
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        qq = queries[s:s + chunk]
        d2 = (qq * qq).sum(1)[:, None] - 2.0 * qq @ base.T + b2
        top = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, top, 1), axis=1)
        out[s:s + chunk] = np.take_along_axis(top, order, 1)
    return out


# ------------------------------------------------------------------------ metrics
def overlap_recall(ids, exact_ids, k):
    return float(np.mean([len(set(ids[i, :k]) & set(exact_ids[i, :k])) / k for i in range(len(ids))]))


def nn_in_top(ids, exact_ids, r):
    return float(np.mean([exact_ids[i, 0] in ids[i, :r] for i in range(len(ids))]))


# ------------------------------------------------------------------- our search
def search_ours(base, queries, labels, route_order, n_search, n_clusters, k=100):
    """Search the n_search best clusters of every query. Returns the ids, ms per query and
    the average number of vectors searched per query."""
    members = [np.where(labels == c)[0] for c in range(n_clusters)]
    out = np.full((len(queries), k), -1, dtype=np.int64)
    t_all, scanned = 0.0, 0
    for i in range(len(queries)):
        t0 = time.time()
        cand = np.concatenate([members[c] for c in route_order[i, :n_search]])
        scanned += len(cand)
        diff = base[cand] - queries[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        kk = min(k, len(cand))
        top = np.argpartition(d2, kk - 1)[:kk]
        out[i, :kk] = cand[top[np.argsort(d2[top])]]
        t_all += time.time() - t0
    return out, t_all / len(queries) * 1000, scanned / len(queries)


def l2_order(queries, centroids):
    d2 = (queries ** 2).sum(1)[:, None] - 2.0 * queries @ centroids.T + (centroids ** 2).sum(1)[None, :]
    return np.argsort(d2, axis=1)


def make_row(method, params, ids, exact_ids, ms, scanned, n_base):
    return {"method": method, "params": params,
            "overlap@10": overlap_recall(ids, exact_ids, 10),
            "NN_in_top1": nn_in_top(ids, exact_ids, 1),
            "NN_in_top10": nn_in_top(ids, exact_ids, 10),
            "ms_per_query": ms,
            "avg_vectors_searched": scanned,
            "pct_of_base": (100.0 * scanned / n_base) if scanned is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets/sift1m")
    ap.add_argument("--url", default="", help="another place to download sift.tar.gz from")
    ap.add_argument("--n_base", type=int, default=200000)
    ap.add_argument("--n_queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=64, help="number of clusters")
    ap.add_argument("--n_search", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--m", type=float, default=2.0, help="Fuzzy C-Means fuzziness")
    ap.add_argument("--fcm_iters", type=int, default=20)
    ap.add_argument("--with_faiss", action="store_true")
    ap.add_argument("--sa_aqee", action="store_true", help="also run our method, SA-AQEE-FCM (V5)")
    ap.add_argument("--m_min", type=float, default=1.02, help="floor of the fuzzifier for SA-AQEE-FCM")
    ap.add_argument("--fit_sample", type=int, default=30000, help="documents used to fit SA-AQEE-FCM")
    ap.add_argument("--skip_hnsw", action="store_true", help="skip the slow HNSW index build")
    args = ap.parse_args()

    base, queries, gt = get_sift(args.data_dir, [args.url] if args.url else None)
    queries = queries[:args.n_queries]
    if args.n_base < len(base):
        base = base[:args.n_base]
        print(f"Using the first {len(base):,} base vectors -> computing exact neighbours", flush=True)
        exact_ids = exact_neighbours(base, queries)
    else:
        exact_ids = gt[:args.n_queries, :100]
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    n_base = len(base)
    print(f"base={base.shape} queries={queries.shape}", flush=True)
    rows = []

    from kmean_retrival import KMeansScratch
    from fuzzycmeans_scratch import FuzzyCMeansScratch

    print("\nFitting our K-Means...", flush=True)
    km = KMeansScratch(n_clusters=args.k, max_iters=20, chunk_size=2000, random_state=42).fit(base)
    labels = km.predict(base)
    sizes = np.bincount(labels, minlength=args.k)
    print(f"K-Means cluster sizes min/max {sizes.min()}/{sizes.max()} (balanced = {n_base // args.k})", flush=True)
    order = l2_order(queries, km.cluster_centers_)
    for ns in args.n_search:
        ids, ms, scanned = search_ours(base, queries, labels, order, ns, args.k)
        rows.append(make_row("previous: K-Means (scratch)", f"K={args.k}, search={ns}", ids, exact_ids, ms, scanned, n_base))

    print(f"\nFitting our Fuzzy C-Means (m={args.m})...", flush=True)
    fcm = FuzzyCMeansScratch(n_clusters=args.k, m=args.m, max_iters=args.fcm_iters, chunk_size=2000,
                             random_state=42).fit(base)
    C = fcm.cluster_centers_
    spread = np.linalg.norm(C[:, None] - C[None], axis=2)[np.triu_indices(args.k, 1)].mean()
    fsizes = np.bincount(fcm.labels_, minlength=args.k)
    print(f"FCM check (m={args.m}): avg highest membership {fcm.u_.max(1).mean():.4f} "
          f"(collapsed = {1 / args.k:.4f}) | centroid spread {spread:.2e} | "
          f"cluster sizes min/max {fsizes.min()}/{fsizes.max()} (balanced = {n_base // args.k})", flush=True)
    order = np.argsort(-fcm.predict_proba(queries), axis=1)
    for ns in args.n_search:
        ids, ms, scanned = search_ours(base, queries, fcm.labels_, order, ns, args.k)
        rows.append(make_row("previous: Fuzzy C-Means (scratch)", f"K={args.k}, m={args.m}, search={ns}",
                             ids, exact_ids, ms, scanned, n_base))


    if args.sa_aqee:
        from sa_aqee_fcm import sa_aqee_fcm, memberships as sa_memberships, cluster_report
        print("\nFitting SA-AQEE-FCM (V5, our method)...", flush=True)
        res = sa_aqee_fcm(base.astype(np.float64), args.k, "V5", np.random.default_rng(42), sample=2000,
                          fit_sample=args.fit_sample, m_min=args.m_min, verbose=True)
        rep = cluster_report(base.astype(np.float64), res, args.k)
        print(f"SA-AQEE check: m*={rep['m']:.3f} | H_norm {rep['H_norm']:.3f} | avg highest membership "
              f"{rep['avg_highest_membership']:.4f} (equal = {rep['equal_membership']:.4f}) | "
              f"cluster sizes min/max {rep['size_min']}/{rep['size_max']} (balanced = {n_base // args.k})", flush=True)
        order = np.argsort(-sa_memberships(queries.astype(np.float64), res["V"], res["m"]), axis=1)
        for ns in args.n_search:
            ids, ms, scanned = search_ours(base, queries, res["labels"], order, ns, args.k)
            rows.append(make_row("ours: SA-AQEE-FCM (V5)", f"K={args.k}, m*={res['m']:.2f}, search={ns}",
                                 ids, exact_ids, ms, scanned, n_base))

    if args.with_faiss:
        import faiss
        max_threads = os.cpu_count() or 4

        def set_threads(n):
            try:
                faiss.omp_set_num_threads(n)
            except Exception:
                pass

        d = base.shape[1]
        set_threads(1)      # timed searches: one query at a time on one thread

        def run(index, label, params, scanned=None):
            t0 = time.time()
            ids = np.empty((len(queries), 100), dtype=np.int64)
            for i in range(len(queries)):
                ids[i] = index.search(queries[i:i + 1], 100)[1][0]
            rows.append(make_row(label, params, ids, exact_ids, (time.time() - t0) / len(queries) * 1000,
                                 scanned, n_base))

        print("\nFAISS step 1/3: exhaustive search...", flush=True)
        flat = faiss.IndexFlatL2(d)
        flat.add(base)
        run(flat, "Exhaustive search (FAISS flat)", "all vectors", float(n_base))

        print("FAISS step 2/3: IVF (K-Means partition)...", flush=True)
        ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, args.k)
        try:
            ivf.cp.max_points_per_centroid = n_base
        except Exception:
            pass
        set_threads(max_threads)
        ivf.train(base)
        ivf.add(base)
        set_threads(1)
        for ns in args.n_search:
            ivf.nprobe = ns
            run(ivf, "FAISS IVF (Johnson et al.)", f"nlist={args.k}, nprobe={ns}")

        if not args.skip_hnsw:
            print("FAISS step 3/3: HNSW build (the slowest step, can take several minutes)...", flush=True)
            hn = faiss.IndexHNSWFlat(d, 32)
            hn.hnsw.efConstruction = 200
            set_threads(max_threads)
            hn.add(base)
            set_threads(1)
            for ef in (16, 64, 128):
                hn.hnsw.efSearch = ef
                run(hn, "HNSW (Malkov and Yashunin)", f"M=32, efSearch={ef}")

    print(f"\n{'method':32s} {'params':28s} {'ovl@10':>7s} {'NN@1':>6s} {'NN@10':>6s} {'vectors searched':>17s} {'ms/q':>7s}")
    for r in rows:
        vs = f"{r['avg_vectors_searched']:,.0f} ({r['pct_of_base']:.1f}%)" if r["avg_vectors_searched"] is not None else "n/a"
        print(f"{r['method']:32s} {r['params']:28s} {r['overlap@10']:7.3f} {r['NN_in_top1']:6.3f} "
              f"{r['NN_in_top10']:6.3f} {vs:>17s} {r['ms_per_query']:7.2f}")
    os.makedirs("cache_cross", exist_ok=True)
    out = f"cache_cross/our_model_sift_{n_base}_m{args.m}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved", out)


if __name__ == "__main__":
    main()
