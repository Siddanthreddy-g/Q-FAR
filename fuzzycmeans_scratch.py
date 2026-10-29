
import numpy as np
import time
import importlib


class FuzzyCMeansScratch:
    def __init__(self, n_clusters, m=2.0, max_iters=20, tol=1e-4,
                 chunk_size=2000, random_state=42, eps=1e-10):
        self.n_clusters = n_clusters
        self.m = m                      # fuzziness exponent (sir said use 2)
        self.max_iters = max_iters
        self.tol = tol
        self.chunk_size = chunk_size
        self.random_state = random_state
        self.eps = eps                  # tiny number, avoids divide-by-zero

        self.cluster_centers_ = None    # (n_clusters, n_features)
        self.u_ = None                  # (n_samples, n_clusters) membership matrix
        self.labels_ = None             # hard labels = argmax(u_, axis=1)

    # ---------------------------------------------------------------
    # Step 0: initial centroids (same K-Means++ idea as KMeansScratch,
    # so the fuzzy version starts from an equally sensible position)
    # ---------------------------------------------------------------
    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        first_idx = rng.integers(n_samples)
        centroids = [X[first_idx]]

        closest_dist_sq = np.full(n_samples, np.inf, dtype=np.float32)

        for _ in range(1, self.n_clusters):
            last_centroid = centroids[-1]
            for start in range(0, n_samples, self.chunk_size):
                end = min(start + self.chunk_size, n_samples)
                chunk = X[start:end]
                d = np.linalg.norm(chunk - last_centroid, axis=1) ** 2
                closest_dist_sq[start:end] = np.minimum(closest_dist_sq[start:end], d)

            probs = closest_dist_sq / closest_dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids, dtype=np.float32)

    # ---------------------------------------------------------------
    # Step 1: membership update, done in chunks
    #   u_ij = 1 / sum_k ( d_ij / d_ik )^(2/(m-1))
    # ---------------------------------------------------------------
    def _update_membership_chunked(self, X, centroids):
        n_samples = X.shape[0]
        power = 2.0 / (self.m - 1.0)
        U = np.empty((n_samples, self.n_clusters), dtype=np.float32)

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            chunk = X[start:end]                                     # (chunk, 384)

            # distance from this chunk to every centroid -> (chunk, n_clusters)
            dists = np.linalg.norm(chunk[:, None, :] - centroids[None, :, :], axis=2)
            dists = np.maximum(dists, self.eps).astype(np.float64)   # avoid /0; float64 so small m (big exponent) cannot overflow

            # ratio[i, j, k] = (d_ij / d_ik)^power, built via broadcasting:
            # shape (chunk, n_clusters, n_clusters) -- only small because
            # n_clusters (64) is small, chunk is only 2000 at a time.
            with np.errstate(over="ignore"):      # an inf here just means "membership 0" (the correct limit)
                ratio = (dists[:, :, None] / dists[:, None, :]) ** power
            denom = ratio.sum(axis=2)             # sum over k -> (chunk, n_clusters)
            U[start:end] = 1.0 / denom

        return U

    # ---------------------------------------------------------------
    # Step 2: centroid update -- weighted mean using u_ij^m as weights
    #   c_j = sum_i (u_ij^m * x_i) / sum_i (u_ij^m)
    # ---------------------------------------------------------------
    def _update_centroids_chunked(self, X, U):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        um = U ** self.m                                   # (n_samples, n_clusters)

        weighted_sum = np.zeros((self.n_clusters, n_features), dtype=np.float64)
        weight_total = np.zeros(self.n_clusters, dtype=np.float64)

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            chunk_X = X[start:end]                          # (chunk, 384)
            chunk_um = um[start:end]                         # (chunk, n_clusters)

            # (n_clusters, chunk) @ (chunk, 384) -> (n_clusters, 384)
            weighted_sum += chunk_um.T @ chunk_X
            weight_total += chunk_um.sum(axis=0)

        weight_total = np.maximum(weight_total, self.eps)[:, None]   # avoid /0
        new_centroids = (weighted_sum / weight_total).astype(np.float32)
        return new_centroids

    # ---------------------------------------------------------------
    # fit: the main Lloyd's-style loop, fuzzy version
    # ---------------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            t0 = time.time()

            # ---- Membership step ----
            U = self._update_membership_chunked(X, centroids)

            # ---- Centroid step ----
            new_centroids = self._update_centroids_chunked(X, U)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            print(f"  iter {iteration+1:2d}/{self.max_iters}  "
                  f"centroid shift={shift:.6f}  ({time.time()-t0:.1f}s)")

            if shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break

        self.cluster_centers_ = centroids
        self.u_ = U
        self.labels_ = np.argmax(U, axis=1)    # hard label = strongest membership
        return self

    # ---------------------------------------------------------------
    # predict: membership (and hard labels) for new/unseen data
    # ---------------------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        U = self._update_membership_chunked(X, self.cluster_centers_)
        return np.argmax(U, axis=1)      # hard labels, for drop-in use like KMeansScratch.predict()

    def predict_proba(self, X):
        """Full membership matrix instead of hard labels -- this is what
        Q-FAR's adaptive routing actually needs later (multi-cluster search,
        entropy-based ambiguity), not just a single hard label."""
        X = np.asarray(X, dtype=np.float32)
        return self._update_membership_chunked(X, self.cluster_centers_)


# -------------------------------------------------------------------------
# Correctness check: compare against scikit-fuzzy's cmeans on a small,
# well-separated synthetic dataset (same style as the K-Means check)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(300, 2)),
        rng.normal(loc=[6, 6], scale=0.5, size=(300, 2)),
        rng.normal(loc=[0, 6], scale=0.5, size=(300, 2)),
    ]).astype(np.float32)

    print("Fitting from-scratch Fuzzy C-Means (m=2)...")
    mine = FuzzyCMeansScratch(n_clusters=3, m=2.0, max_iters=50,
                               chunk_size=200, random_state=42).fit(X)
    print("\nMy centroids:\n", mine.cluster_centers_)
    print("\nMembership matrix shape:", mine.u_.shape, " (rows sum to 1:",
          np.allclose(mine.u_.sum(axis=1), 1.0), ")")
    print("First 5 points' memberships:\n", mine.u_[:5])

    # Optional comparison against the reference library implementation,
    # if scikit-fuzzy is installed (pip install scikit-fuzzy)
    try:
        fuzz = importlib.import_module("skfuzzy")
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X.T, c=3, m=2.0, error=1e-4, maxiter=50, seed=42
        )
        print("\nscikit-fuzzy centroids:\n", cntr)
        print("(Order may differ, but the 3 points found should match)")
    except ImportError:
        print("\n(scikit-fuzzy not installed -- skipping library comparison. "
              "pip install scikit-fuzzy to enable it.)")
