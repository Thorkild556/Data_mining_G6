import numpy as np

class DBScan:
    def __init__(self, radius: float, min_dense: int):
        self.radius = radius
        self.min_dense = min_dense


    def calculate_distances(self, v: np.array, m: np.array):
        return np.linalg.norm(v - m, axis=1) # calc the diff and then the length of the diff between v and all vectors in m
    
    def get_neighbors(self, idx: int, X: np.array):
        dists = self.calculate_distances(X[idx], X)
        return np.where(dists <= self.radius)[0] # returns an array where the distance is less than the radius
        
    def make_cluster(self, X, start_idx):
        neighbor_stack = [start_idx]
        start = True
        while neighbor_stack:
            idx = neighbor_stack.pop()
            if self.visited[idx] == 1: #has already been visited
                continue
            self.visited[idx] = 1 # it has now been visited
            neigh_idxs = self.get_neighbors(idx, X)

            if neigh_idxs.shape[0] < 1:
                self.type[idx] = 3 # noise for sure.. it has no neighbors
                return
            elif neigh_idxs.shape[0] >= self.min_dense: # if a core object
                self.type[idx] = 1 # core object, we continue
                neighbor_stack.extend(neigh_idxs.tolist()) # we continue on its children since its a core object
                start = False
                self.clusters[idx] = self.n_cluster # put it in the current cluster.
            else: # we either have an edge/border or a noise point but we will check all later
                if start == False: # we know now it definitely came from a core object 
                    self.type[idx] = 2
                    self.clusters[idx] = self.n_cluster
                else: # in the case that this is the first point we look at, we do not know yet if we have any core neighbors
                    self.visited[idx] = 0  #so if there is a core neighbor we will later discover it
                    self.type[idx] = 3 #leave it as noise for now, but might be overwritten.
                    
        self.n_cluster += 1
    
    def make_clusters(self, X):
        self.visited = np.zeros(shape=(X.shape[0]))
        self.type = np.zeros(shape=len(X)) # 1 = core, 2 = edge, 3 = noise
        self.clusters = np.zeros(shape=len(X))
        self.n_cluster = 1
        for idx in range(X.shape[0]):
            if self.visited[idx] == 0:
                self.make_cluster(X, start_idx=idx)

        return self.clusters


   
   
        
        
            
            
