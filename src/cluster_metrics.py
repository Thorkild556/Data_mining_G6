from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from src.DBScan import DBScan
import numpy as np
from itertools import chain
from typing import List, Tuple, Optional
from datasets import load_dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class EvalClustering:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pca = PCA(n_components=int(2 ** 6))

    def __init__(self, _dataset: DatasetDict, force_embeddings: Optional = None, grp=0, radius=0.6, min_dense=10):
        self.scanner = DBScan(radius=radius, min_dense=min_dense)
        if force_embeddings is not None:
            self.tests = force_embeddings
        else:
            self.tests = self.embed(list(chain.from_iterable(_dataset["test"]["sentences"][:grp + 1])))
        self.test_results = list(chain.from_iterable(_dataset["test"]["labels"][:grp + 1]))

    def embed(self, samples):
        embedded = self.embedder.encode(samples)
        print(embedded.shape)
        return self.dim_red(embedded)

    def dim_red(self, sample_embedded):
        print("BEFORE", sample_embedded.shape)
        transformed_samples = self.pca.fit_transform(sample_embedded)
        print("AFTER", transformed_samples.shape)
        return transformed_samples

    def db_scan(self):
        cluster_of_samples = self.scanner.make_clusters(self.tests)
        noisy = np.where(cluster_of_samples == 0)
        print("Number of clusters produced: ", len(np.unique(cluster_of_samples)) - int(bool(noisy)))
        return {
            "noise": len(np.where(cluster_of_samples == 0)),
            "clusters": cluster_of_samples,
            "cluster_num": len(np.unique(cluster_of_samples)) - int(bool(noisy)),
            "raw_clusters": np.unique(cluster_of_samples).size
        }

    def eval_db_scan(self):
        db_scan_results = self.db_scan()
        table, cluster_map = self.contingency_tab(db_scan_results["cluster_num"], db_scan_results["clusters"])

        return {
            "Noise %": self.percent(db_scan_results["noise"] / db_scan_results["cluster_num"]),
            "purity_score": self.purity_score(table),
            "nml_score": self.nml_score(table),
            "clusters_found": db_scan_results["cluster_num"],
            "noise": db_scan_results["noise"],
            "silhouette_score": self.silhouette(db_scan_results["clusters"])
        }

    @staticmethod
    def percent(num_):
        return round(num_ * 1e2, 2)

    def contingency_tab(self, got_this_many_clusters: int, clustered):
        table = np.zeros(20 * got_this_many_clusters).reshape((20, got_this_many_clusters))
        cluster_map = {clustered_to: cluster_index for cluster_index, clustered_to in enumerate(sorted(set(clustered)))}
        for cluster_index, clustered_to in enumerate(clustered):
            if clustered_to == 0:
                continue
            table[self.test_results[cluster_index], cluster_map[clustered_to] - 1] += 1

        return table, cluster_map

    def prob_contingency(self, table):
        total = np.sum(table)
        _new_table = table.copy()
        for row in range(_new_table.shape[0]):
            for col in range(_new_table.shape[1]):
                actual_prob = _new_table[row][col] / total
                if actual_prob == 0:
                    continue
                from_actual_cluster = np.sum(_new_table[row, :]) / total
                to_predicted_cluster = np.sum(_new_table[:, col]) / total

                _new_table[row][col] = actual_prob * np.log2(actual_prob / (from_actual_cluster * to_predicted_cluster))
        return _new_table

    def entropy(self, table, true_labels=False):
        total = np.sum(table)
        val = 0
        for row in range(table.shape[0] if true_labels else table.shape[1]):
            if true_labels:
                prob = np.sum(table[row, :]) / total
            else:
                prob = np.sum(table[:, row]) / total
            if prob == 0:
                continue
            val += -prob * np.log2(prob)
        return val

    def purity_score(self, table):
        return np.sum(np.max(table, axis=0)) / np.sum(table)

    def nml_score(self, table):
        top = np.sum(self.prob_contingency(table))
        bottom = np.sqrt(self.entropy(table) * self.entropy(table, true_labels=True))
        if bottom == 0:
            return 1
        return top / bottom

    def get_k_nearest_neighbors(self, idx: int, k: int = 10, data_items: List = None) -> Tuple[List[int], List[float]]:
        if data_items is None:
            data_items = self.tests
        dists = [
            (
                self.calc_distance(self.tests[idx], _) if self.tests[idx] != self.tests[idx] else np.inf
            )
            for _ in data_items
        ]
        points = np.argsort(dists)[:k].tolist()
        return points, [dists[_] for _ in points]

    def single_link_coefficient(self, clustered: List[int]):
        ...

    def silhouette(self, clustered: List[int], use_cosine: bool = False):
        # cluster_id (by db scan) = list of points in that cluster
        clustered_map = {
            clustered_to: [self.tests[idx] for idx, _ in enumerate(clustered) if _ != 0]
            for clustered_to in
            sorted(set(clustered)) if clustered_to != 0
        }
        dist_func = cosine_similarity if use_cosine else euclidean_distances
        # point to its avg distance with all its points in teh cluster
        internal_cluster_distance = {}
        for cluster in clustered_map:
            dists = dist_func(clustered_map[cluster], clustered_map[cluster])
            for point, _ in enumerate(clustered_map[cluster]):
                internal_cluster_distance[point] = np.mean(
                    np.concatenate((dists[point, :point], dists[point, point + 1:]))
                )

        external_cluster_distance = {}
        for point_index, point in enumerate(clustered):
            avg_distance = np.inf
            if point == 0:
                continue
            for cluster in clustered_map:
                if cluster == point:
                    continue

                avg_distance = min(
                    avg_distance,
                    np.mean(
                        dist_func(clustered_map[point], [self.tests[point_index]])
                    )
                )
            external_cluster_distance[point_index] = avg_distance

        return {
            ((internal_cluster_distance[point] - external_cluster_distance[point]) / max(
                internal_cluster_distance[point], external_cluster_distance[point]))
            for point, _ in enumerate(clustered)
            if point in external_cluster_distance and point in internal_cluster_distance
        }


if __name__ == "__main__":
    dataset = load_dataset("mteb/twentynewsgroups-clustering", revision="6125ec4e24fa026cec8a478383ee943acfbd5449")
    print(dataset["test"])

    # Extracting variables
    test = dataset["test"]
    texts_sep = list(test["sentences"])
    labels_sep = list(test["labels"])

    error = EvalClustering(dataset)
    print(error.eval_db_scan())
