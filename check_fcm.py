import numpy as np
from kmean_retrival import KMeansScratch
from fuzzycmeans_scratch import FuzzyCMeansScratch
d = np.load("cache_nfcorpus_3633/docs.npy"); K = 12
f = FuzzyCMeansScratch(n_clusters=K, m=2.0, max_iters=20, chunk_size=2000, random_state=42).fit(d)
k = KMeansScratch(n_clusters=K, max_iters=20, chunk_size=2000, random_state=42).fit(d)
def spread(C): D = np.linalg.norm(C[:, None] - C[None], axis=2); return D[np.triu_indices(len(C), 1)].mean()
print("avg highest membership:", f.u_.max(1).mean(), "(flat would be %.3f)" % (1 / K))
print("FCM cluster sizes:", np.bincount(f.labels_, minlength=K))
print("K-Means cluster sizes:", np.bincount(k.predict(d), minlength=K))
print("centroid spread, FCM:", spread(f.cluster_centers_), " K-Means:", spread(k.cluster_centers_))