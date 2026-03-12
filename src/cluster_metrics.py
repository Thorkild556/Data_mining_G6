import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


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
        return (
                external_cluster_distance - internal_cluster_distance
        ) / np.maximum(internal_cluster_distance, external_cluster_distance)

    @staticmethod
    def percent(num_):
        # just a helper function to convert a number to a percentage
        return round(num_ * 1e2, 2)

    @classmethod
    def contingency_tab(cls, got_this_many_clusters: int, clustered: List[int], test_results: List[int]):
        # we would be building the table where number of clusters we got * number of actual clusters
        # where each element tells us which cluster we got for which actual cluster
        table = np.zeros((got_this_many_clusters, len(set(test_results))), dtype=int)

        # we are doing this since the clusters from db scan can we any number so we are normalizing the values we got
        # like 3, 4, 5, 4 => 0, 1, 2, 1
        db_scan_clusters, db_scan_clusters_mapped = np.unique(clustered, return_inverse=True)
        test_results_clustered, test_results_mapped = np.unique(test_results, return_inverse=True)

        # we update their frequency
        np.add.at(table, (db_scan_clusters_mapped, test_results_mapped), 1)
        return table, db_scan_clusters

    @classmethod
    def normalized_mutual_info(cls, table):
        total = np.sum(table)
        prob = table / total

        among_our_clusters = np.sum(prob, axis=0)
        among_actual_clusters = np.sum(prob, axis=1)

        entropy_of_our_clusters = -np.sum(among_our_clusters * np.log2(among_our_clusters))
        entropy_from_actual_clusters = -np.sum(among_actual_clusters * np.log2(among_actual_clusters))

        joint = np.outer(among_actual_clusters, among_our_clusters)
        mask = prob > 0
        mutual_info = np.sum(prob[mask] * np.log2(prob[mask] / joint[mask]))
        r = mutual_info / np.sqrt(entropy_of_our_clusters * entropy_from_actual_clusters)
        return 0 if np.isnan(r) else r

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
