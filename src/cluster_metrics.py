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
import plotly.graph_objects as go
import plotly.subplots as sub
from loguru import logger
import gc


class ClusterMetrics:
    @classmethod
    def silhouette(cls, clustered: np.ndarray, unique_clusters: List[int], dist_matrix: np.ndarray):
        # please note we have the dist_matrix where the embeddings * embeddings
        # where each cell would be the distance between them
        internal_cluster_distance = np.zeros(len(clustered))
        external_cluster_distance = np.zeros(len(clustered))

        # for every cluster we have predicted
        for cluster in unique_clusters:
            cluster_points = np.where(clustered == cluster)[0]

            # we would be creating the matrix where it would have those embeddings inside that cluster
            mini_matrix = dist_matrix[np.ix_(cluster_points, cluster_points)]
            # saving those results inside the internal_cluster_distance for every point
            internal_cluster_distance[cluster_points] = np.nanmean(mini_matrix, axis=1)

            avg_distance = None

            # now for every other cluster
            for cluster_to in unique_clusters:
                if cluster_to == cluster:
                    continue

                # we would be calcuating the distance between ever point of one cluster to every other points in
                # other cluster (pair wise)

                other_cluster_points = np.where(clustered == cluster_to)[0]
                mini_matrix_to = dist_matrix[
                    np.ix_(cluster_points, other_cluster_points)
                ]
                _avg = np.nanmean(mini_matrix_to, axis=1)

                if avg_distance is None:
                    avg_distance = _avg
                else:
                    # and then we would be picking (for every point) min. distance to its cluster
                    # https://stackoverflow.com/questions/39277638/element-wise-minimum-of-multiple-vectors-in-numpy
                    avg_distance = np.minimum.reduce([avg_distance, _avg])

            external_cluster_distance[cluster_points] = avg_distance

        # for every point we have the (external - internal) distance
        # and we would be dividing it by the max of internal and external distance
        return (
                external_cluster_distance - internal_cluster_distance
        ) / np.maximum(internal_cluster_distance, external_cluster_distance)

    @staticmethod
    def percent(num_):
        # just a helper function to convert a number to a percentage
        return round(num_ * 1e2, 2)

    @classmethod
    def contingency_tab(cls, got_this_many_clusters: int, clustered: List[int], test_results: List[int]):
        # builds the contingency table for the clustering results
        table = np.zeros((got_this_many_clusters, len(set(test_results))), dtype=int)
        db_scan_clusters, db_scan_clusters_mapped = np.unique(clustered, return_inverse=True)
        np.add.at(table, (db_scan_clusters_mapped, test_results), 1)
        return table, db_scan_clusters

    @classmethod
    def normalized_mutual_info(cls, table):
        total = np.sum(table)

        # we first calculate the probablity of element in the contingency table
        prob = table / total

        # and among our clusters
        among_our_clusters = np.sum(table, axis=0) / total
        among_actual_clusters = np.sum(table, axis=1) / total
        entropy_of_our_clusters = -np.sum(among_our_clusters * np.log2(among_our_clusters))
        entropy_from_actual_clusters = -np.sum(among_actual_clusters * np.log2(among_actual_clusters))

        mutual_info = prob * np.log2(prob / (among_our_clusters * among_actual_clusters))
        return mutual_info / np.sqrt(entropy_of_our_clusters * entropy_from_actual_clusters)

    @classmethod
    def purity_score(cls, table):
        cluster_number = np.sum(table, axis=0)
        scores_for_clusters = np.max(table, axis=0) / cluster_number
        return scores_for_clusters, np.average(scores_for_clusters, axis=0, weights=cluster_number / np.sum(table))
    
    @classmethod
    def ps(cls, clusters_we_got, test_results):
        return cls.purity_score(
            cls.contingency_tab(
                len(set(clusters_we_got)),
                clusters_we_got, test_results
            )
        )


class EvalClustering(ClusterMetrics):
    root = Path(__file__).parent
    embedder = None
    pca = None

    def __init__(self, _dataset: Optional[DatasetDict] = None, force_embeddings: Optional = None, grp=0, radius=0.6,
                 min_dense=10, force_dist_matrix: Optional[np.ndarray] = None, use_cosine: bool = True):
        # all u need to understand from this constructor is that we use the embeddings and the model object here only when needed
        # I wanted to make sure all the checkpoints
        # are saved to save time to test the changes
        self.scanner = DBScan(radius=radius, min_dense=min_dense)
        if force_embeddings is not None:
            self.tests = force_embeddings
        else:
            if _dataset is None:
                self.tests = np.load(self.root / "embeddings.npy")
                logger.info("Loaded embeddings.npy")
            else:
                if (self.root / "embeddings.npy").exists():
                    self.tests = np.load(self.root / "embeddings.npy")
                    logger.info("Loaded embeddings.npy")
                else:
                    self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                    self.pca = PCA(n_components=int(2 ** 6))

                    self.tests = self.embed(list(chain.from_iterable(_dataset["test"]["sentences"][:grp + 1])))
                    np.save(self.root / "embeddings.npy", self.tests)
                    logger.info("Saved embeddings.npy")

        if _dataset is not None:
            if (self.root / "labels.json").exists():
                self.test_results = json.loads((self.root / "labels.json").read_text())
            else:
                self.test_results = list(chain.from_iterable(_dataset["test"]["labels"][:grp + 1]))
                (self.root / "labels.json").write_text(json.dumps(self.test_results))
                logger.info("Saved labels.json")
        else:
            self.test_results = json.loads((self.root / "labels.json").read_text())

        if force_dist_matrix is not None:
            self.dist_matrix = force_dist_matrix
        else:
            if not (self.root / "dists.npy").exists():
                logger.info("Calculating distance matrix")
                if use_cosine:
                    self.dist_matrix = (1 - cosine_similarity(self.tests, self.tests))
                else:
                    self.dist_matrix = euclidean_distances(self.tests, self.tests)

                # we would be making sure the distance between the same embeddings are NAN
                # so they can be ignored in NAN
                np.fill_diagonal(self.dist_matrix, np.nan)

                np.save(self.root / "dists.npy", self.dist_matrix)
                del self.dist_matrix
                gc.collect()

            # memmap because its 60k x 60k and we don't want to load it all into ram
            self.dist_matrix = np.memmap(self.root / "dists.npy", dtype='float32', mode='r',
                                         shape=(len(self.tests), len(self.tests)))
            logger.info("Loaded dists.npy")

    def embed(self, samples):
        embedded = self.embedder.encode(samples)
        print(embedded.shape)
        return self.dim_red(embedded)

    def dim_red(self, sample_embedded):
        print("BEFORE", sample_embedded.shape)
        transformed_samples = self.pca.fit_transform(sample_embedded)
        print("AFTER", transformed_samples.shape)
        return transformed_samples

    def db_scan(self, remove_noise: bool = True):
        cluster_of_samples = self.scanner.make_clusters(self.tests)
        unique_clusters = set(cluster_of_samples)
        print("Number of clusters produced: ", len(unique_clusters) - int(0 in unique_clusters))
        if remove_noise:
            focus = np.where(cluster_of_samples != 0)[0]
            cluster_of_samples = cluster_of_samples[focus]
            self.test_results = np.array(self.test_results)[focus]
            logger.info("Number of clusters produced after removing noise: {}", (len(unique_clusters) - 1,))
        return {
            "noise": len(np.where(cluster_of_samples == 0)) if 0 in unique_clusters else 0,
            "clusters": list(cluster_of_samples),
            "cluster_num": len(unique_clusters) - int(0 in unique_clusters),
            "raw_clusters": np.unique(cluster_of_samples).size
        }

    def eval_db_scan(self, force_read=False, save_results=False, remove_noise=True):
        if not force_read:
            logger.info("Running DBSCAN")
            db_scan_results = self.db_scan(remove_noise)
            logger.info("DBSCAN Completed")
            if save_results:
                (self.root / "db_scan_results.json").write_text(json.dumps(db_scan_results))
        else:
            logger.info("Loading DBSCAN results")
            db_scan_results = json.loads((self.root / "db_scan_results.json").read_text())
        table, unique_clusters = self.contingency_tab(
            db_scan_results["cluster_num"],
            db_scan_results["clusters"],
            self.test_results
        )
        logger.info("Loaded contingency table and validating the results")
        score = self.silhouette(db_scan_results["clusters"], list(unique_clusters), self.dist_matrix)
        logger.info("Silhouette score: {}", np.mean(score))
        return {
            "Noise %": self.percent(db_scan_results["noise"] / db_scan_results["cluster_num"]),
            "purity_score": self.purity_score(table),
            "nml_score": self.nml_score(table),
            "clusters_found": db_scan_results["cluster_num"],
            "noise": db_scan_results["noise"],
            "mean_silhouette_score": np.mean(score),
            "silhouette_score": score
        }

    @classmethod
    def plot_single_result(cls, sil_scores: List[float]):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=np.arange(len(sil_scores)), y=sil_scores, mode="lines+markers", name="Silhouette Scores"))
        avg_score = np.mean(sil_scores)
        fig.add_trace(
            go.Scatter(x=[0, len(sil_scores) - 1], y=[avg_score, avg_score], mode="lines",
                       line=dict(color="red", dash="dash"),
                       name=f"Average: {avg_score:.2f}"))
        fig.update_layout(title="Silhouette Scores", template="plotly_dark")
        fig.update_xaxes(title_text="Points")
        fig.update_yaxes(title_text="Silhouette Score")
        fig.update_layout(width=900, height=600)
        return fig

    @classmethod
    def plot_multiple_results(cls, run_results: List[dict], run_results_title: List[str]):
        fig = sub.make_subplots(
            rows=2, cols=3,
            specs=[[{}, {}, {}],
                   [{}, {"colspan": 1}, {}]],
            subplot_titles=["Purity Score", "NML Score", "Clusters Found", "Mean Silhouette Score", "Silhouette Score",
                            "Noise %"])
        purity_scores = [result["purity_score"] for result in run_results]
        nml_scores = [result["nml_score"] for result in run_results]
        clusters_found = [result["clusters_found"] for result in run_results]
        noise_percentages = [result["Noise %"] for result in run_results]
        mean_silhouette_scores = [np.mean(result["silhouette_score"]) for result in run_results]

        fig.add_trace(go.Bar(x=run_results_title, y=purity_scores), row=1, col=1)
        fig.add_trace(go.Bar(x=run_results_title, y=nml_scores), row=1, col=2)
        fig.add_trace(go.Bar(x=run_results_title, y=clusters_found), row=1, col=3)
        fig.add_trace(go.Bar(x=run_results_title, y=mean_silhouette_scores), row=2, col=1)
        fig.add_trace(go.Bar(x=run_results_title, y=noise_percentages), row=2, col=3)

        for i, run in enumerate(run_results):
            fig.add_trace(
                go.Scatter(x=np.arange(len(run["silhouette_score"])), y=run["silhouette_score"], mode="lines+markers",
                           name=run_results_title[i]), row=2, col=2)
        fig.update_layout(showlegend=False, width=1000, height=600)
        fig.update_layout(title="Comparison of Clustering Results", template="plotly_dark")
        fig.update_xaxes(title_text="Run")
        fig.update_yaxes(title_text="Score")
        return fig

    @classmethod
    def plot_multiple_results_with_selected_scores(cls, run_results, run_results_title):
        fig = sub.make_subplots(
            rows=2, cols=3,
            specs=[[{}, {}, {}],
                   [{}, {"colspan": 2}, None]],
            subplot_titles=["Purity Score", "NML Score", "Clusters Found", "Mean Silhouette Score", "Silhouette Score",
                            "Noise %"])
        purity_scores = [result["purity_score"] for result in run_results]
        clusters_found = [result["clusters_found"] for result in run_results]
        noise_percentages = [result["Noise %"] for result in run_results]
        mean_silhouette_scores = [np.mean(result["silhouette_score"]) for result in run_results]

        fig.add_trace(go.Bar(x=run_results_title, y=purity_scores), row=1, col=1)
        fig.add_trace(go.Bar(x=run_results_title, y=clusters_found), row=1, col=2)
        fig.add_trace(go.Bar(x=run_results_title, y=noise_percentages), row=1, col=3)
        fig.add_trace(go.Bar(x=run_results_title, y=mean_silhouette_scores), row=2, col=1)

        for i, run in enumerate(run_results):
            fig.add_trace(
                go.Scatter(fill='tozeroy', x=np.arange(len(run["silhouette_score"])), y=run["silhouette_score"],
                           name=run_results_title[i]), row=2, col=2)
        fig.update_layout(showlegend=False, width=1000, height=600)
        fig.update_layout(title="Comparison of Clustering Results", template="plotly_dark")
        fig.update_xaxes(title_text="Run")
        fig.update_yaxes(title_text="Score")
        return fig

    def __del__(self):
        if self.dist_matrix:
            del self.dist_matrix
        gc.collect()


# debugging script to make sure script can be modified to run fast
if __name__ == "__main__":
    from scipy.stats.contingency import crosstab
    from sklearn.metrics.pairwise import euclidean_distances

    a = [1, 2, 2, 3, 4, 5, 5, 5, 7, 7, 7, 9, 10, 10, 11, 19, 18, 17, 16, 6, 8, 12, 13, 14, 15, 0]
    b = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
    assert len(a) == len(b)
    res = crosstab(b, a)
    ours, _ = ClusterMetrics.contingency_tab(len(set(b)), np.array(b), np.array(a))
    print(res)
    print(ours)
    assert np.all(res.count == ours)

    from sklearn.metrics import normalized_mutual_info_score, homogeneity_score

    expected = normalized_mutual_info_score(a, b)
    actual = ClusterMetrics.normalized_mutual_info(ours)
    assert expected == actual

    expected = homogeneity_score(a, b)
    actual = ClusterMetrics.purity_score(ours)
    assert expected == actual

# if __name__ == "__main__":
#     error = EvalClustering(grp=9, radius=0.00001, min_dense=10)
#     results = error.eval_db_scan(False)
#     print(results)
#     EvalClustering.plot_single_result(
#         results["silhouette_score"]
#     ).show()

# #
# EvalClustering.plot_multiple_results(
#     [{"Noise %": 10, "purity_score": 0.5, "nml_score": 0.6, "clusters_found": 7,
#       "mean_silhouette_score": np.mean([0.7, 0.8, 0.9]),
#       "silhouette_score": [0.7, 0.8, 0.9]},
#      {"Noise %": 5, "purity_score": 0.8, "nml_score": 0.9, "clusters_found": 10,
#       "mean_silhouette_score": np.mean([0.85, 0.9, 0.95]), "silhouette_score": [0.85, 0.9, 0.95]},
#      {"Noise %": 15, "purity_score": 0.7, "nml_score": 0.8, "clusters_found": 9,
#       "mean_silhouette_score": np.mean([0.75, 0.8, 0.85]), "silhouette_score": [0.75, 0.8, 0.85]}],
#     ["Run 1", "Run 2", "Run 3"]
# ).show()
#
# EvalClustering.plot_single_result(
#     [0.7, 0.8, 0.9]
# ).show()

# print(ClusterMetrics.silhouette(
#     np.array([0, 0, 1, 1, 0, 1, 2, 2, 2, 2]),
#     [0, 1, 2],
#     np.random.rand(10, 10)
# ))
