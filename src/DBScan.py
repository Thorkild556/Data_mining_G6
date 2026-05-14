import numpy as np
import sklearn

class DBScan:
    def __init__(self, radius: float, min_dense: int):
        self.radius = radius
        self.min_dense = min_dense

    def calculate_distances(self, v: np.array, m: np.array):
        return np.linalg.norm(v - m,
                              axis=1)  # calc the diff and then the length of the diff between v and all vectors in m
    
    def cosine_distances(self, v: np.array, m: np.array):
        return (1 - sklearn.metrics.pairwise.cosine_similarity(v.reshape(1, -1), m)).flatten()
    
    def get_neighbors(self, idx: int, X: np.array, cosine: bool = True):
        if cosine:
            dists = self.cosine_distances(X[idx], X)
        else:
            dists = self.calculate_distances(X[idx], X)
        neighbors = np.where(dists <= self.radius)[0]
        return neighbors[neighbors != idx]  # exclude self
    
    def make_cluster(self, X, start_idx, cosine=True):
        neighbor_stack = [start_idx]
        cluster_formed = False
        while neighbor_stack:
            idx = neighbor_stack.pop()
            if self.visited[idx] == 1:
                continue
            self.visited[idx] = 1
            neigh_idxs = self.get_neighbors(idx, X, cosine)

            if len(neigh_idxs) >= self.min_dense:
                self.type[idx] = 1
                self.clusters[idx] = self.n_cluster
                neighbor_stack.extend(neigh_idxs.tolist())
                cluster_formed = True
            else:
                self.type[idx] = 3

        if cluster_formed:
            self.n_cluster += 1
    
    def make_clusters(self, X, cosine: bool = True):
        self.visited = np.zeros(shape=(X.shape[0]))
        self.type = np.zeros(shape=len(X))
        self.clusters = np.full(shape=len(X), fill_value=-1)  # fix: -1 for noise
        self.n_cluster = 0  # fix: start at 0

        for idx in range(X.shape[0]):
            if self.visited[idx] == 0:
                self.make_cluster(X, start_idx=idx, cosine=cosine)

        for idx in range(X.shape[0]):
            if self.type[idx] == 3:
                neigh_idxs = self.get_neighbors(idx, X, cosine)
                for n in neigh_idxs:
                    if self.type[n] == 1:
                        self.type[idx] = 2
                        self.clusters[idx] = self.clusters[n]
                        break  # still arbitrary, but at least noise=-1 is now distinct

        return self.clusters


