import numpy as np
from typing import List

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
from rich.table import Table
from sklearn.manifold import TSNE
from scipy.sparse import triu
from scipy.sparse.csgraph import connected_components
import numpy as np
import matplotlib.pyplot as plt


class ClusterMetrics:
    @classmethod
    def silhouette(
        cls, clustered: np.ndarray, unique_clusters: List[int], dist_matrix: np.ndarray
    ):
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
                # we refer to the distance provided by the function input
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
        return (external_cluster_distance - internal_cluster_distance) / np.maximum(
            internal_cluster_distance, external_cluster_distance
        )

    @staticmethod
    def percent(num_):
        # just a helper function to convert a number to a percentage
        return round(num_ * 1e2, 2)

    @classmethod
    def contingency_tab(
        cls, got_this_many_clusters: int, clustered: List[int], test_results: List[int]
    ):
        # we would be building the table where number of clusters we got * number of actual clusters
        # where each element tells us which cluster we got for which actual cluster
        table = np.zeros((got_this_many_clusters, len(set(test_results))), dtype=int)

        # we are doing this since the clusters from db scan can we any number so we are normalizing the values we got
        # like 3, 4, 5, 4 => 0, 1, 2, 1
        db_scan_clusters, db_scan_clusters_mapped = np.unique(
            clustered, return_inverse=True
        )
        test_results_clustered, test_results_mapped = np.unique(
            test_results, return_inverse=True
        )

        # we update their frequency
        np.add.at(table, (db_scan_clusters_mapped, test_results_mapped), 1)
        return table, db_scan_clusters

    @classmethod
    def normalized_mutual_info(cls, table):
        total = np.sum(table)
        prob = table / total

        among_our_clusters = np.sum(prob, axis=0)
        among_actual_clusters = np.sum(prob, axis=1)

        entropy_of_our_clusters = -np.sum(
            among_our_clusters * np.log2(among_our_clusters)
        )
        entropy_from_actual_clusters = -np.sum(
            among_actual_clusters * np.log2(among_actual_clusters)
        )

        joint = np.outer(among_actual_clusters, among_our_clusters)
        mask = prob > 0
        mutual_info = np.sum(prob[mask] * np.log2(prob[mask] / joint[mask]))
        r = mutual_info / np.sqrt(
            entropy_of_our_clusters * entropy_from_actual_clusters
        )
        return 0 if np.isnan(r) else r

    @classmethod
    def purity_score(cls, table):
        cluster_number = np.sum(table, axis=0)
        scores_for_clusters = np.max(table, axis=0) / cluster_number
        return scores_for_clusters, np.average(
            scores_for_clusters, weights=cluster_number / np.sum(table)
        )

    @classmethod
    def ps(cls, clusters_we_got, test_results):
        return cls.purity_score(
            cls.contingency_tab(
                len(set(clusters_we_got)), clusters_we_got, test_results
            )[0]
        )

    @classmethod
    def s_score(cls, clusters_we_got, test_results, embeddings):
        # we would be calculating all possible distances between every point
        dist_matrix = 1 - cosine_similarity(embeddings, embeddings)
        np.fill_diagonal(dist_matrix, np.nan)
        _, unique_ = cls.contingency_tab(
            len(set(clusters_we_got)), clusters_we_got, test_results
        )
        return cls.silhouette(clusters_we_got, unique_, dist_matrix)

    @classmethod
    def nml_score(cls, clusters_we_got, test_results):
        return cls.normalized_mutual_info(
            cls.contingency_tab(
                len(set(clusters_we_got)), clusters_we_got, test_results
            )[0]
        )

    @classmethod
    def eval_results(cls, features_arr, clusters, grps, title):
        clusters_arr = np.asarray(clusters)
        clusters_masked = clusters_arr != -1

        nml_score = ClusterMetrics.nml_score(
            clusters_arr[clusters_masked], grps[clusters_masked]
        )
        purity_score = ClusterMetrics.ps(
            clusters_arr[clusters_masked], grps[clusters_masked]
        )
        sil_score = ClusterMetrics.s_score(
            clusters_arr[clusters_masked],
            grps[clusters_masked],
            features_arr[clusters_masked],
        )

        table = Table(title=f"{title} Results")
        table.add_column("Metric", justify="right")
        table.add_column("Score", justify="right")

        table.add_row("NML", str("{:.3}".format(nml_score)))
        table.add_row("Purity", str("{:.3}".format(purity_score[1])))
        table.add_row("Silhouette", str("{:.3}".format(np.mean(sil_score))))

        return table


class DimRed:
    proj: np.ndarray
    label_proj: np.ndarray

    def __init__(self, data_set: np.ndarray, labels_lists, label_names):
        self.labels_list = labels_lists
        self.label_names = label_names
        self.data_set = data_set
        self.proj = None

    def tsne(self, n_components: int = 2):
        dim_red = TSNE(n_components=n_components)
        self.proj = dim_red.fit_transform(self.data_set)
        return self.proj

    def display_proj(self, title="", r="", w=900, h=800):
        if self.proj is None:
            self.tsne()
        palette = px.colors.qualitative.Light24

        label_names_list = [self.label_names[int(_)] for _ in self.labels_list]
        unique_labels = list(dict.fromkeys(label_names_list))
        color_map = {
            label: palette[i % len(palette)] for i, label in enumerate(unique_labels)
        }

        fig = go.Figure()

        for label in unique_labels:
            mask = [i for i, l in enumerate(label_names_list) if l == label]
            fig.add_trace(
                go.Scatter(
                    x=self.proj[mask, 0],
                    y=self.proj[mask, 1],
                    mode="markers",
                    name=label,
                    marker=dict(color=color_map[label], size=6),
                    showlegend=True,
                )
            )

        fig.update_layout(title=f"Dataset (Based on the {title})", width=w, height=h)
        fig.show(renderer=r)

    def pca_dim(self, n_components: int = 2):
        return PCA(n_components=n_components, random_state=42)


def make_wordcloud(text, title, ax):
    wc = WordCloud(
        stopwords=STOPWORDS, max_words=100, background_color="white"
    ).generate(text)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title)
    ax.axis("off")


def tight_word_clouds(texts, labels_list, label_names=None):
    # Group texts by cluster label
    cluster_texts = pd.Series(texts).groupby(labels_list).apply(lambda x: " ".join(x))

    n_clusters = len(cluster_texts)
    ncols = 2
    nrows = -(-n_clusters // ncols)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, nrows * 3))
    axs = axs.flatten()

    for i, (label, text) in enumerate(cluster_texts.items()):
        make_wordcloud(
            text,
            f"{'Cluster: ' if not label_names else ''}{label if not label_names else (label_names[label])}",
            axs[i],
        )

    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_results(projections, clusters, title):
    fig = px.scatter(
        projections,
        x=0,
        y=1,
        color=[str(_) for _ in clusters],
        labels={"color": "labels"},
        color_discrete_sequence=px.colors.qualitative.Light24,
        title=f"Dataset (Based on {title})",
    )
    fig.show(renderer="png")


def distance_matrix_of_clusters(
    grps, dist_matrix, label_names, title="", plot=True, r=None
):
    # Initialize a 20x20 matrix to store average distances
    n_clusters = 20
    avg_dist_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(n_clusters):
            # Extract the sub-matrix of distances between points in cluster i and cluster j
            sub_matrix = dist_matrix[grps == i][:, grps == j]

            if i == j:
                # Intra-cluster: Mean of the upper triangle (excluding the diagonal 0s)
                upper_idx = np.triu_indices_from(sub_matrix, k=1)
                avg_dist_matrix[i, j] = np.mean(sub_matrix[upper_idx])
            else:
                # Inter-cluster: Mean of all pairwise distances between the two groups
                avg_dist_matrix[i, j] = np.mean(sub_matrix)
    if not plot:
        return avg_dist_matrix
    plot_distance_matrix(avg_dist_matrix, label_names, title, r)


def plot_distance_matrix(dist_matrix, labels, title, r=None):
    fig = px.imshow(
        dist_matrix,
        x=labels,
        y=labels,
        labels=dict(x="Cluster B", y="Cluster A", color="Avg Distance"),
        title=title,
        color_continuous_scale="Viridis",  # You can change this to 'RdBu' or 'Plasma'
        aspect="auto",
    )

    # Layout adjustments for readability
    fig.update_layout(width=900, height=900, xaxis_nticks=20, yaxis_nticks=20)

    fig.show(renderer=r)


def plot_confusion_matrix(actual_clusters, returned, label_names, title="", r=None):
    grps_named = pd.Series(actual_clusters).map(lambda x: label_names[int(x)])

    df_cm = pd.crosstab(
        index=grps_named,
        columns=returned,
        rownames=["True Labels"],
        colnames=["Actual Labels"],
    )

    df_cm.columns = [f"{c}" for c in df_cm.columns]

    fig = px.imshow(
        df_cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Clusters", y="Groups", color="Count"),
    )

    fig.update_layout(
        xaxis=dict(side="top"),  # Cluster labels on top
        title=title,
        width=900,
        height=800,
    )

    fig.show(renderer=r)
