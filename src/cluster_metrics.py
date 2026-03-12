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
        test_results_clustered, test_results_mapped = np.unique(test_results, return_inverse=True)

        np.add.at(table, (db_scan_clusters_mapped, test_results_mapped), 1)
        return table, db_scan_clusters

    @classmethod
    def normalized_mutual_info(cls, table):
        total = np.sum(table)

        # we first calculate the probablity of element in the contingency table
        prob = table / total

        # and among our clusters
        among_our_clusters = np.sum(prob, axis=0)
        among_actual_clusters = np.sum(prob, axis=1)

        entropy_of_our_clusters = -np.sum(among_our_clusters * np.log2(among_our_clusters))
        entropy_from_actual_clusters = -np.sum(among_actual_clusters * np.log2(among_actual_clusters))


        result = np.outer(among_our_clusters, among_actual_clusters)
        mask = result > 0
        mutual_info = np.sum(prob[mask] * np.log2(prob[mask] / result[mask]))
        return np.sum(
            mutual_info / np.sqrt(entropy_of_our_clusters * entropy_from_actual_clusters)
        )

    @classmethod
    def purity_score(cls, table):
        cluster_number = np.sum(table, axis=0)
        scores_for_clusters = np.max(table, axis=0) / cluster_number
        return scores_for_clusters, np.average(scores_for_clusters, weights=cluster_number / np.sum(table))

    @classmethod
    def ps(cls, clusters_we_got, test_results):
        return cls.purity_score(
            cls.contingency_tab(
                len(set(clusters_we_got)),
                clusters_we_got, test_results
            )[0]
        )

    @classmethod
    def s_score(cls, clusters_we_got, test_results, embeddings):
        dist_matrix = (1 - cosine_similarity(embeddings, embeddings))
        np.fill_diagonal(dist_matrix, np.nan)
        _, unique_ = cls.contingency_tab(len(set(clusters_we_got)), clusters_we_got, test_results)
        return cls.silhouette(clusters_we_got, unique_, dist_matrix)

    @classmethod
    def nml_score(cls, clusters_we_got, test_results):
        return cls.normalized_mutual_info(
            cls.contingency_tab(
                len(set(clusters_we_got)),
                clusters_we_got, test_results
            )[0]
        )


class PlotCLusterMetrics:
    @classmethod
    def plot_single_results(cls, sil_scores: List[float]):
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


# Debugging / smoke-test suite
if __name__ == "__main__":
    from sklearn.metrics import (
        normalized_mutual_info_score,
        homogeneity_score,
        silhouette_score,
    )
    from sklearn.metrics.pairwise import cosine_distances


    def check(name, condition, extra=""):
        tag = "Passed" if condition else "Failed"
        print(f"  {tag}  {name}" + (f"  →  {extra}" if extra else ""))
        return condition


    all_passed = True

    # table test
    print("\n[1] contingency_tab")

    pred = np.array([0, 0, 1, 1, 2, 2])
    true = np.array([0, 0, 1, 1, 2, 2])
    table, labels = ClusterMetrics.contingency_tab(3, pred, true)

    check("shape", table.shape == (3, 3), str(table.shape))
    check("diagonal (perfect)", np.all(np.diag(table) == 2), str(np.diag(table)))

    # purity test
    print("\n[2] purity_score")

    # sklearn equivalent: homogeneity_score is a reasonable proxy
    pred = [0, 0, 1, 1, 2, 2, 2]
    true = [0, 0, 1, 1, 2, 0, 2]
    scores, overall = ClusterMetrics.ps(pred, true)
    sk_hom = homogeneity_score(pred, true)

    check("per-cluster scores", np.all((scores >= 0) & (scores <= 1)), str(scores))
    check("overall", 0.0 <= overall <= 1.0, f"{overall:.4f}")
    check("overall", overall > 0, f"purity={overall:.4f} homogeneity={sk_hom:.4f}")

    # perfect clustering → purity == 1
    pred_p = [0, 0, 1, 1]
    true_p = [0, 0, 1, 1]
    _, overall_p = ClusterMetrics.ps(pred_p, true_p)
    all_passed &= check("perfect clustering → purity=1", np.isclose(overall_p, 1.0), str(overall_p))

    # ── 3. normalized_mutual_info ────────────────────────────────────────────
    print("\n[3] normalized_mutual_info")

    for pred_l, true_l, desc in [
        ([0, 0, 1, 1], [0, 0, 1, 1], "perfect"),
        ([0, 1, 0, 1], [0, 0, 1, 1], "random"),
        ([0, 0, 0, 0], [0, 1, 2, 3], "all-same pred"),
    ]:
        pred_a, true_a = np.array(pred_l), np.array(true_l)
        sk_nmi = normalized_mutual_info_score(pred_a, true_a)
        table, _ = ClusterMetrics.contingency_tab(len(set(pred_l)), pred_a, true_a)
        ours = ClusterMetrics.nml_score(pred_a, true_a)
        print("HERE", sk_nmi)
        print("HERE0", ours)

    # ── 4. silhouette / s_score ──────────────────────────────────────────────
    print("\n[4] silhouette / s_score")

    rng = np.random.default_rng(42)
    # blobs pointing in clearly different *directions* so cosine distance is large
    emb_a = rng.normal(loc=[1, 0], scale=0.05, size=(30, 2))  # points right
    emb_b = rng.normal(loc=[0, 1], scale=0.05, size=(30, 2))  # points up
    embeddings = np.vstack([emb_a, emb_b])
    labels_s = np.array([0] * 30 + [1] * 30)

    sk_sil = silhouette_score(embeddings, labels_s, metric="cosine")
    ours_sil = ClusterMetrics.s_score(labels_s, labels_s, embeddings)
    all_passed &= check("s_score returns array", isinstance(ours_sil, np.ndarray), str(ours_sil.shape))
    all_passed &= check(
        "s_score mean close to sklearn",
        np.isclose(np.mean(ours_sil), sk_sil, atol=0.05),
        f"ours={np.mean(ours_sil):.4f} sklearn={sk_sil:.4f}"
    )
    all_passed &= check("well-separated clusters → high silhouette", np.mean(ours_sil) > 0.8,
                        f"mean={np.mean(ours_sil):.4f}")

    # accepts list-of-tuples (the original bug)
    small_emb = [(1, 2), (3, 4), (10, 11), (12, 13)]
    small_lab = [0, 0, 1, 1]
    try:
        res = ClusterMetrics.s_score(small_lab, small_lab, small_emb)
        all_passed &= check("list-of-tuples embeddings — no crash", True, str(res))
    except Exception as e:
        all_passed &= check("list-of-tuples embeddings — no crash", False, str(e))

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n" + ("=" * 50))
    print(f"  {'ALL TESTS PASSED ✅' if all_passed else 'SOME TESTS FAILED ❌'}")
    print("=" * 50)
