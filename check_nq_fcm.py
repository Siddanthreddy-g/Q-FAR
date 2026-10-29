import pickle, numpy as np
fcm = pickle.load(open("cache/fuzzy_cmeans_k64.pkl", "rb"))
C = fcm.cluster_centers_
print("avg highest membership:", fcm.u_.max(1).mean(), "(collapsed = %.4f)" % (1 / 64))
print("centroid spread:", np.linalg.norm(C[:, None] - C[None], axis=2)[np.triu_indices(64, 1)].mean())
s = np.bincount(fcm.labels_, minlength=64); print("largest cluster:", s.max(), " empty:", (s == 0).sum(), " (balanced = 3125)")