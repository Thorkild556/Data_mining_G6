import numpy as np
from scipy.spatial.distance import cdist


class DBScan:
    def __init__(self, radius: float, min_dense: int):
        self.radius = radius
        self.min_dense = min_dense

    def calculate_distances(self, v: np.array, m: np.array):
        return np.linalg.norm(v - m,
                              axis=1)  # calc the diff and then the length of the diff between v and all vectors in m

    def get_neighbors(self, idx: int, X: np.array):
        dists = self.calculate_distances(X[idx], X)
        return np.where(dists <= self.radius)[0]  # returns an array where the distance is less than the radius

    def make_cluster(self, X, idx_list=np.array([0]), root=True):
        for idx in idx_list:
            if self.visited[idx] == 1:  # has already been visited
                continue
            self.visited[idx] = 1  # it has now been visited
            self.clusters[idx] = self.n_cluster  # put it in the current cluster.
            neigh_idxs = self.get_neighbors(idx, X)

            if neigh_idxs.shape[0] < 1:
                self.type[idx] = 3  # noise
                return
            elif neigh_idxs.shape[0] >= self.min_dense:
                self.type[idx] = 1  # core object, we continue
                self.make_cluster(X, neigh_idxs, root=False)  # we continue on its children since its a core object
            else:  # we have an edge/border
                self.type[idx] = 2

        if root == True:  # bc we are done making the clustering
            self.n_cluster += 1

    def make_clusters(self, X):
        self.visited = np.zeros(shape=len(X))
        self.type = np.zeros(shape=len(X))  # 1 = core, 2 = edge, 3 = noise
        self.clusters = np.zeros(shape=len(X))
        self.n_cluster = 1
        for idx in range(X.shape[0]):
            if self.visited[idx] == 0:
                self.make_cluster(X, idx_list=np.array([idx]))
        return self.clusters

    def calculate_distance_matrix(self, X):
        return cdist(X, X)

    def get_(self):
        return
