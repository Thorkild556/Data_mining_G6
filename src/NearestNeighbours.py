import numpy as np
import sklearn.metrics.pairwise

class NearestNeighbours:
    def __init__(self, k_neighbours: int):
        self.k = k_neighbours
        self.X = None

    def fit(self, X: np.array):
        self.X = np.asarray(X, dtype=np.float32)
        return self

    def calculate_euclidean_distances(self, v: np.array, m: np.array):
        return np.sqrt(np.sum((m - v) * (m - v), axis=1))

    def calculate_cosine_distances(self, v: np.array, m: np.array):
        return (1 - sklearn.metrics.pairwise.cosine_similarity(v.reshape(1, -1), m)).flatten()

    def _get_distances(self, idx: int, cosine: bool):
        if cosine:
            dists = self.calculate_cosine_distances(self.X[idx], self.X)
        else:
            dists = self.calculate_euclidean_distances(self.X[idx], self.X)
        dists[idx] = np.inf  # exclude itself
        return dists

    def get_k_nearest_neighbors(self, idx: int, cosine: bool = False):
        dists = self._get_distances(idx, cosine)
        return np.argsort(dists)[:self.k]

    def get_kth_nearest_neighbor_distance(self, idx: int, cosine: bool = False):
        dists = self._get_distances(idx, cosine)
        return np.partition(dists, self.k - 1)[self.k - 1]

    def k_neighbours(self, X: np.array, cosine: bool = False):
        for i in range(self.X.shape[0]):
            yield self.get_k_nearest_neighbors(i, cosine)

    def kth_distances(self, cosine: bool = False):
        for i in range(self.X.shape[0]):
            yield self.get_kth_nearest_neighbor_distance(i, cosine)