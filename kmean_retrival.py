import numpy as np
import time


class KMeansScratch:
    def __init__(self, n_clusters, max_iters=20, tol=1e-4, chunk_size=2000, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.chunk_size = chunk_size      # how many points to compare at once
        self.random_state = random_state
        self.cluster_centers_ = None      # same attribute name as sklearn, for drop-in use
        self.labels_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        first_idx = rng.integers(n_samples)
        centroids = [X[first_idx]]

        # distance from every point to its nearest chosen centroid so far
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

    def _assign_chunked(self, X, centroids):
        """
        Assignment step, done in chunks so we never build a full
        (n_samples x n_clusters) distance matrix in one go.
        """
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int64)

        for start in range(0, n_samples, self.chunk_size):
            end = min(start + self.chunk_size, n_samples)
            chunk = X[start:end]                         # (chunk, 384)
            dists = np.linalg.norm(chunk[:, None, :] - centroids[None, :, :], axis=2)
            labels[start:end] = np.argmin(dists, axis=1)

        return labels

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            t0 = time.time()

            # ---- Assignment step (chunked) ----
            labels = self._assign_chunked(X, centroids)

            # ---- Update step ----
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                members = X[labels == k]
                new_centroids[k] = members.mean(axis=0) if len(members) > 0 else centroids[k]

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            print(f"  iter {iteration+1:2d}/{self.max_iters}  "
                  f"centroid shift={shift:.6f}  ({time.time()-t0:.1f}s)")

            if shift < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self._assign_chunked(X, self.cluster_centers_)


if __name__ == "__main__":
    from sklearn.cluster import KMeans as SklearnKMeans

    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(500, 2)),
        rng.normal(loc=[5, 5], scale=0.5, size=(500, 2)),
        rng.normal(loc=[0, 5], scale=0.5, size=(500, 2)),
    ]).astype(np.float32)

    print("Fitting chunked K-Means with K-Means++ init...")
    mine = KMeansScratch(n_clusters=3, max_iters=20, chunk_size=200, random_state=42).fit(X)

    theirs = SklearnKMeans(n_clusters=3, random_state=42, n_init=10).fit(X)

    print("\nMy centroids:\n", mine.cluster_centers_)
    print("\nsklearn centroids:\n", theirs.cluster_centers_)