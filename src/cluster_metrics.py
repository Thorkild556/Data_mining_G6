from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from src.DBScan import DBScan
import numpy as np
from itertools import chain
from typing import List, Optional
from datasets import DatasetDict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from pathlib import Path
import json


class EvalClustering:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pca = PCA(n_components=int(2 ** 6))

    def __init__(self, _dataset: Optional[DatasetDict] = None, force_embeddings: Optional = None, grp=0, radius=0.6,
                 min_dense=10):
        self.scanner = DBScan(radius=radius, min_dense=min_dense)
        if force_embeddings is not None:
            self.tests = force_embeddings
        else:
            if _dataset is None:
                self.tests = np.load("embeddings.npy")
            else:
                self.tests = self.embed(list(chain.from_iterable(_dataset["test"]["sentences"][:grp + 1])))
                np.save("embeddings.npy", self.tests)
        if _dataset is not None:
            self.test_results = list(chain.from_iterable(_dataset["test"]["labels"][:grp + 1]))
            Path("labels.json").write_text(json.dumps(self.test_results))
        else:
            self.test_results = json.loads(Path("labels.json").read_text())

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
        unique_clusters = set(cluster_of_samples)
        print("Number of clusters produced: ", len(unique_clusters) - int(0 in unique_clusters))
        return {
            "noise": len(np.where(cluster_of_samples == 0)) if 0 in unique_clusters else 0,
            "clusters": list(cluster_of_samples),
            "cluster_num": len(unique_clusters) - int(0 in unique_clusters),
            "raw_clusters": np.unique(cluster_of_samples).size
        }

    def eval_db_scan(self, force_read=False):
        if not force_read:
            db_scan_results = self.db_scan()
            Path("db_scan_results.json").write_text(json.dumps(db_scan_results))
        else:
            db_scan_results = json.loads(Path("db_scan_results.json").read_text())
        table, unique_clusters = self.contingency_tab(
            db_scan_results["cluster_num"],
            db_scan_results["clusters"]
        )
        print(table.shape)

        return {
            "Noise %": self.percent(db_scan_results["noise"] / db_scan_results["cluster_num"]),
            "purity_score": self.purity_score(table),
            "nml_score": self.nml_score(table),
            "clusters_found": db_scan_results["cluster_num"],
            "noise": db_scan_results["noise"],
            "silhouette_score": self.silhouette(db_scan_results["clusters"], unique_clusters)
        }

    @staticmethod
    def percent(num_):
        return round(num_ * 1e2, 2)

    def contingency_tab(self, got_this_many_clusters: int, clustered: List[int]):
        table = np.zeros((20, got_this_many_clusters), dtype=int)
        db_scan_clusters, db_scan_clusters_mapped = np.unique(clustered, return_inverse=True)
        np.add.at(table, (self.test_results, db_scan_clusters_mapped), 1)
        return table, db_scan_clusters

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

    def silhouette(self, clustered: List[int], unique_clusters: List[int], use_cosine: bool = True):
        dist_func = (lambda x, y: (1 - cosine_similarity(x, y))) if use_cosine else euclidean_distances

        internal_cluster_distance = np.zeros(len(clustered))
        external_cluster_distance = np.zeros(len(clustered))


        distance_of_every_point_with_all_other_points = dist_func(self.tests, self.tests)
        np.fill_diagonal(distance_of_every_point_with_all_other_points, np.nan)  # we don't want diagonal distances as they are same point

        for cluster in unique_clusters:
            cluster_points = np.where(clustered == cluster)
            other_cluster_points = np.where(clustered != cluster)

            mini_matrix = distance_of_every_point_with_all_other_points[cluster_points, cluster_points]
            internal_cluster_distance[cluster_points] += np.nanmean(mini_matrix, axis=1)

            avg_distance = None

            for cluster_to in unique_clusters:
                if cluster_to == cluster:
                    continue

                mini_matrix_to = distance_of_every_point_with_all_other_points[
                    cluster_points, other_cluster_points
                ]
                _avg = np.nanmean(mini_matrix_to, axis=1)

                if avg_distance is None:
                    avg_distance = _avg
                else:
                    # https://stackoverflow.com/questions/39277638/element-wise-minimum-of-multiple-vectors-in-numpy
                    avg_distance = np.minimum.reduce([avg_distance, _avg])

            external_cluster_distance[cluster_points] += avg_distance

        return (
                external_cluster_distance - internal_cluster_distance
        ) / np.maximum(internal_cluster_distance, external_cluster_distance)


# debugging script to make sure script can be modified to run fast
if __name__ == "__main__":
    error = EvalClustering(grp=10)
    results = error.eval_db_scan(True)
    print(results)
