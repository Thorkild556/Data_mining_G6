import numpy as np

class NearestNeighbours:

    def __init__(self, k_neighbours: int):
        self.k = k_neighbours
        self.X = None

    def fit(self, X: np.array):
        self.X = np.asarray(X, dtype=np.float32)
        return self
        
    def calculate_euclidean_distances(self, v: np.array, m: np.array):
        euclidean_distance = np.sqrt(np.sum((m-v) * (m-v), axis=1))
        return euclidean_distance

    def get_k_nearest_neighbors(self, idx: int):
        dists = self.calculate_euclidean_distances(self.X[idx], self.X)
        dists[idx] = np.inf  # exclude itself
        return np.argsort(dists)[:self.k]
    
    def get_kth_nearest_neighbor_distance(self, idx: int):
        dists = self.calculate_euclidean_distances(self.X[idx], self.X)
        dists[idx] = np.inf  # exclude itself
        return np.partition(dists, self.k-1)[self.k-1]
    
    def k_neighbours(self, X: np.array):
        for i in range(self.X.shape[0]):
            yield self.get_k_nearest_neighbors(i)
        
    
    
