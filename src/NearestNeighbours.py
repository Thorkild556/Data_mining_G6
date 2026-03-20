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

    def get_distances(self, idx: int, cosine: bool):
        if cosine:
            dists = self.calculate_cosine_distances(self.X[idx], self.X)
        else:
            dists = self.calculate_euclidean_distances(self.X[idx], self.X)
        dists[idx] = np.inf  # exclude itself
        return dists

    def get_k_nearest_neighbors(self, idx: int, cosine: bool = False):
        dists = self.get_distances(idx, cosine)
        return np.argsort(dists)[:self.k]

    def get_kth_nearest_neighbor_distance(self, idx: int, cosine: bool = False):
        dists = self.get_distances(idx, cosine)
        return np.partition(dists, self.k - 1)[self.k - 1]

    def k_neighbours(self, X: np.array, cosine: bool = False):
        for i in range(self.X.shape[0]):
            yield self.get_k_nearest_neighbors(i, cosine)

    def kth_distances(self, cosine: bool = False):
        for i in range(self.X.shape[0]):
            yield self.get_kth_nearest_neighbor_distance(i, cosine)

    def build_mutual_knn_graph(self, cosine: bool = False, weighted: bool = False, sigma: float = 1.0) -> np.ndarray:
        n_samples = self.X.shape[0] # nodes in the graph

        neighbor_sets = []

        for i in range(n_samples):
            neighbors = self.get_k_nearest_neighbors(i, cosine) # get k nearest neighbors for node i with cosine distance
            neighbor_sets.append(set(neighbors.tolist()))

        adjacency = np.zeros((n_samples, n_samples), dtype=np.float32) # initialize adjacency matrix with zeros (no edges)

        for i in range(n_samples):
            for j in neighbor_sets[i]:

                if i not in neighbor_sets[j]:
                    continue    # we only want mutual neighbors

                if weighted:
                    cosine_distance = self.get_distances(i, cosine)[j] #cosine distance between node i and j
                    weight = np.exp(-(cosine_distance ** 2) / (sigma ** 2)) # convert distance to similarity using Gaussian kernel (RBF)
                else:
                    weight = 1.0 # unweighted graph: all edges have weight 1.0

                adjacency[i, j] = weight
                adjacency[j, i] = weight # fill the adjencency matrics

        return adjacency

    def mutual_knn_edges(self, cosine: bool = False, weighted: bool = False, sigma: float = 1.0) -> list[tuple[int, int, float]]:
        adjacency = self.build_mutual_knn_graph(cosine=cosine, weighted=weighted, sigma=sigma) # get adjacency matrix

        edges = []

        rows, cols = np.where(np.triu(adjacency, k=1) > 0) # get indices of upper triangular part of the adjacency matrix where there are edges (weight > 0) to avoid duplicates in undirected graph

        for i, j in zip(rows, cols):
            edges.append((int(i), int(j), float(adjacency[i, j])))
            # transfrom the matrix into list of edges

        return edges
