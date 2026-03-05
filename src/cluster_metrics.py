from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from src.DBScan import DBScan
import numpy as np
from itertools import chain


class TrialAndError:
    seed = 66
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pca = PCA(n_components=int(2 ** 6))

    def __init__(self, dataset, force_embeddings=None, grp=0, radius=0.6, min_dense=10):
        self.scanner = DBScan(radius=radius, min_dense=min_dense)
        if force_embeddings is not None:
            self.tests = force_embeddings
        else:
            self.tests = self.embed(list(chain.from_iterable(dataset["test"]["sentences"][:grp + 1])))
        self.test_results = list(chain.from_iterable(dataset["test"]["labels"][:grp + 1]))

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

    def knn(self, samples):
        ...
        return {}

    def calc_distance(self, point_from, point_to):
        return

    def eval_db_scan(self):
        db_scan_results = self.db_scan()
        table, cluster_map = self.contingency_tab(db_scan_results["cluster_num"], db_scan_results["clusters"])

        return {
            "Noise %": self.percent(db_scan_results["noise"] / db_scan_results["cluster_num"]),
            "purity_score": self.percent(self.purity_score(table)),
            "nml_score": self.percent(self.nml_score(table)),
            "clusters_found": db_scan_results["cluster_num"],
            "noise": db_scan_results["noise"]
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

        table.sum(axis=1)
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
