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


class EvalClustering:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    pca = PCA(n_components=int(2 ** 6))
    root = Path(__file__).parent

    def __init__(self, _dataset: Optional[DatasetDict] = None, force_embeddings: Optional = None, grp=0, radius=0.6,
                 min_dense=10, force_dist_matrix: Optional[np.ndarray] = None, use_cosine: bool = True):
        # all u need to understand from this constructor is that we use the embeddings and the model object here
        # it looks complexier because I run low end system so I wanted to make sure all the checkpoints
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
            db_scan_results["clusters"]
        )
        logger.info("Loaded contingency table and validating the results")
        score = self.silhouette(db_scan_results["clusters"], list(unique_clusters))
        return {
            "Noise %": self.percent(db_scan_results["noise"] / db_scan_results["cluster_num"]),
            "purity_score": self.purity_score(table),
            "nml_score": self.nml_score(table),
            "clusters_found": db_scan_results["cluster_num"],
            "noise": db_scan_results["noise"],
            "mean_silhouette_score": np.mean(score),
            "silhouette_score": score
        }

    @staticmethod
    def percent(num_):
        return round(num_ * 1e2, 2)

    def contingency_tab(self, got_this_many_clusters: int, clustered: List[int]):
        table = np.zeros((got_this_many_clusters, 20), dtype=int)
        db_scan_clusters, db_scan_clusters_mapped = np.unique(clustered, return_inverse=True)
        np.add.at(table, (db_scan_clusters_mapped, self.test_results), 1)
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

    def silhouette(self, clustered: List[int], unique_clusters: List[int]):
        internal_cluster_distance = np.zeros(len(clustered))
        external_cluster_distance = np.zeros(len(clustered))

        for cluster in unique_clusters:
            cluster_points = np.where(clustered == cluster)[0]

            mini_matrix = self.dist_matrix[np.ix_(cluster_points, cluster_points)]
            internal_cluster_distance[cluster_points] = np.nanmean(mini_matrix, axis=1)

            avg_distance = None

            for cluster_to in unique_clusters:
                if cluster_to == cluster:
                    continue

                other_cluster_points = np.where(clustered == cluster_to)[0]
                mini_matrix_to = self.dist_matrix[
                    np.ix_(cluster_points, other_cluster_points)
                ]
                _avg = np.nanmean(mini_matrix_to, axis=1)

                if avg_distance is None:
                    avg_distance = _avg
                else:
                    # https://stackoverflow.com/questions/39277638/element-wise-minimum-of-multiple-vectors-in-numpy
                    avg_distance = np.minimum.reduce([avg_distance, _avg])

            external_cluster_distance[cluster_points] = avg_distance

        return (
                external_cluster_distance - internal_cluster_distance
        ) / np.maximum(internal_cluster_distance, external_cluster_distance)

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
        fig.update_layout(showlegend=False)
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
            fig.add_trace(go.Scatter(fill='tozeroy', x=np.arange(len(run["silhouette_score"])), y=run["silhouette_score"], name=run_results_title[i]), row=2, col=2)
        fig.update_layout(showlegend=False)
        fig.update_layout(title="Comparison of Clustering Results", template="plotly_dark")
        fig.update_xaxes(title_text="Run")
        fig.update_yaxes(title_text="Score")
        return fig

    def __del__(self):
        del self.dist_matrix
        gc.collect()


# debugging script to make sure script can be modified to run fast
if __name__ == "__main__":
    # error = EvalClustering(grp=9, radius=0.00001, min_dense=10)
    # results = error.eval_db_scan(True)
    # print(results)
    # EvalClustering.plot_single_result(
    #     results["silhouette_score"]
    # ).show()

    # #
    EvalClustering.plot_multiple_results(
        [{"Noise %": 10, "purity_score": 0.5, "nml_score": 0.6, "clusters_found": 7,
          "mean_silhouette_score": np.mean([0.7, 0.8, 0.9]),
          "silhouette_score": [0.7, 0.8, 0.9]},
         {"Noise %": 5, "purity_score": 0.8, "nml_score": 0.9, "clusters_found": 10,
          "mean_silhouette_score": np.mean([0.85, 0.9, 0.95]), "silhouette_score": [0.85, 0.9, 0.95]},
         {"Noise %": 15, "purity_score": 0.7, "nml_score": 0.8, "clusters_found": 9,
          "mean_silhouette_score": np.mean([0.75, 0.8, 0.85]), "silhouette_score": [0.75, 0.8, 0.85]}],
        ["Run 1", "Run 2", "Run 3"]
    ).show()

    EvalClustering.plot_single_result(
        [0.7, 0.8, 0.9]
    ).show()
