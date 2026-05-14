from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


@dataclass
class LSAResult:
    svd: TruncatedSVD
    document_topic_matrix: np.ndarray
    dominant_topics: np.ndarray
    topic_table: pd.DataFrame
    explained_variance_ratio: float
    n_topics: int
    selection_method: str
    selection_diagnostics: pd.DataFrame | None = None


@dataclass
class TopicSelectionResult:
    n_topics: int
    selection_method: str
    diagnostics: pd.DataFrame


def _ranked_term_indices(weights: np.ndarray, top_n_terms: int) -> np.ndarray:
    return np.argsort(np.abs(weights))[::-1][:top_n_terms]


def _validate_topic_search_bounds(tfidf_matrix, min_topics: int, max_topics: int | None) -> tuple[int, int]:
    max_valid_topics = min(tfidf_matrix.shape) - 1
    if max_valid_topics < 2:
        raise ValueError("TF-IDF matrix must support at least 2 topics for elbow selection.")

    if max_topics is None:
        max_topics = min(15, max_valid_topics)
    else:
        max_topics = min(max_topics, max_valid_topics)

    if min_topics < 2:
        raise ValueError("min_topics must be at least 2.")

    if min_topics > max_topics:
        raise ValueError(
            f"Topic search range is empty: min_topics={min_topics}, max_topics={max_topics}."
        )

    return min_topics, max_topics


def select_lsa_topic_count_by_elbow(
    tfidf_matrix,
    min_topics: int = 2,
    max_topics: int | None = None,
    random_state: int = 42,
    n_iter: int = 20,
) -> TopicSelectionResult:
    min_topics, max_topics = _validate_topic_search_bounds(tfidf_matrix, min_topics, max_topics)
    topic_counts = np.arange(min_topics, max_topics + 1)

    cumulative_explained_variance = []
    for topic_count in topic_counts:
        svd = TruncatedSVD(
            n_components=int(topic_count),
            algorithm="randomized",
            n_iter=n_iter,
            random_state=random_state,
        )
        svd.fit(tfidf_matrix)
        cumulative_explained_variance.append(float(np.sum(svd.explained_variance_ratio_)))

    explained_variance = np.asarray(cumulative_explained_variance, dtype=float)
    if len(topic_counts) == 1 or np.isclose(explained_variance[0], explained_variance[-1]):
        normalized_topic_counts = np.zeros(len(topic_counts), dtype=float)
        normalized_explained_variance = np.zeros(len(topic_counts), dtype=float)
        elbow_scores = np.zeros(len(topic_counts), dtype=float)
        selected_idx = 0
    else:
        normalized_topic_counts = (topic_counts - topic_counts[0]) / (topic_counts[-1] - topic_counts[0])
        normalized_explained_variance = (
            (explained_variance - explained_variance[0])
            / (explained_variance[-1] - explained_variance[0])
        )
        elbow_scores = normalized_explained_variance - normalized_topic_counts
        selected_idx = int(np.argmax(elbow_scores))

    selected_n_topics = int(topic_counts[selected_idx])
    diagnostics = pd.DataFrame(
        {
            "n_topics": topic_counts.astype(int),
            "explained_variance_ratio": explained_variance,
            "normalized_topic_count": normalized_topic_counts,
            "normalized_explained_variance": normalized_explained_variance,
            "elbow_score": elbow_scores,
        }
    )
    diagnostics["selected"] = diagnostics["n_topics"] == selected_n_topics

    return TopicSelectionResult(
        n_topics=selected_n_topics,
        selection_method="elbow on cumulative explained variance",
        diagnostics=diagnostics,
    )


def fit_lsa_topics(
    tfidf_matrix,
    feature_names,
    n_topics: int,
    top_n_terms: int = 10,
    random_state: int = 42,
    n_iter: int = 20,
    selection_method: str = "manual",
    selection_diagnostics: pd.DataFrame | None = None,
) -> LSAResult:
    svd = TruncatedSVD(
        n_components=n_topics,
        algorithm="randomized",
        n_iter=n_iter,
        random_state=random_state,
    )
    document_topic_matrix = svd.fit_transform(tfidf_matrix)

    topic_rows = []
    for topic_id, weights in enumerate(svd.components_):
        ranked_idx = _ranked_term_indices(weights, top_n_terms)
        top_terms = [feature_names[idx] for idx in ranked_idx]
        topic_rows.append(
            {
                "topic_id": topic_id,
                "top_terms": ", ".join(top_terms),
                "topic_strength": float(np.mean(np.abs(document_topic_matrix[:, topic_id]))),
                "explained_variance_ratio": float(svd.explained_variance_ratio_[topic_id]),
            }
        )

    dominant_topics = np.argmax(np.abs(document_topic_matrix), axis=1)

    return LSAResult(
        svd=svd,
        document_topic_matrix=document_topic_matrix,
        dominant_topics=dominant_topics,
        topic_table=pd.DataFrame(topic_rows).sort_values(
            by="explained_variance_ratio",
            ascending=False,
        ),
        explained_variance_ratio=float(np.sum(svd.explained_variance_ratio_)),
        n_topics=n_topics,
        selection_method=selection_method,
        selection_diagnostics=selection_diagnostics,
    )


def representative_documents_for_topics(
    texts,
    document_topic_matrix: np.ndarray,
    dominant_topics: np.ndarray,
    top_n_docs: int = 3,
    preview_chars: int = 180,
) -> pd.DataFrame:
    rows = []
    topic_ids = np.unique(dominant_topics)

    for topic_id in topic_ids:
        topic_mask = dominant_topics == topic_id
        topic_indices = np.where(topic_mask)[0]
        topic_scores = np.abs(document_topic_matrix[topic_indices, topic_id])
        ranked_local_idx = np.argsort(topic_scores)[::-1][:top_n_docs]

        for rank, local_idx in enumerate(ranked_local_idx, start=1):
            doc_idx = topic_indices[local_idx]
            rows.append(
                {
                    "topic_id": int(topic_id),
                    "rank": rank,
                    "document_index": int(doc_idx),
                    "topic_score": float(topic_scores[local_idx]),
                    "preview": texts[doc_idx][:preview_chars].replace("\n", " "),
                }
            )

    return pd.DataFrame(rows)


def compare_topics_to_groups(
    tfidf_matrix,
    feature_names,
    group_labels,
    dominant_topics: np.ndarray,
    topic_table: pd.DataFrame,
    top_n_terms: int = 10,
    group_name: str = "group",
    group_value_names: list[str] | tuple[str, ...] | dict[int, str] | None = None,
    exclude_group_value: int | str | None = None,
):
    group_labels = np.asarray(group_labels)
    dominant_topics = np.asarray(dominant_topics)
    valid_mask = np.ones(len(group_labels), dtype=bool)

    if exclude_group_value is not None:
        valid_mask &= group_labels != exclude_group_value

    if not np.any(valid_mask):
        raise ValueError(f"No valid {group_name} labels were provided for comparison.")

    topic_group_counts = pd.crosstab(
        pd.Series(group_labels[valid_mask], name=group_name),
        pd.Series(dominant_topics[valid_mask], name="topic"),
    )
    topic_group_share = topic_group_counts.div(topic_group_counts.sum(axis=1), axis=0).fillna(0.0)

    topic_lookup = topic_table.set_index("topic_id")["top_terms"].to_dict()
    rows = []

    def resolve_group_value_name(group_value):
        if group_value_names is None:
            return str(group_value)
        if isinstance(group_value_names, dict):
            return str(group_value_names.get(group_value, group_value))
        group_index = int(group_value)
        if 0 <= group_index < len(group_value_names):
            return str(group_value_names[group_index])
        return str(group_value)

    group_id_column = f"{group_name}_id"
    group_name_column = f"{group_name}_name"
    group_size_column = f"{group_name}_size"
    group_top_terms_column = f"{group_name}_top_terms"

    for group_value in topic_group_counts.index:
        group_mask = group_labels == group_value
        group_topic_counts = topic_group_counts.loc[group_value]
        matched_topic = int(group_topic_counts.idxmax())

        group_term_weights = sparse.csr_matrix(tfidf_matrix[group_mask]).sum(axis=0)
        group_term_weights = np.asarray(group_term_weights).ravel()
        ranked_idx = np.argsort(group_term_weights)[::-1][:top_n_terms]
        group_terms = [feature_names[idx] for idx in ranked_idx]

        topic_terms = [term.strip() for term in topic_lookup[matched_topic].split(",")]
        overlap = sorted(set(group_terms).intersection(topic_terms))

        rows.append(
            {
                group_id_column: group_value,
                group_name_column: resolve_group_value_name(group_value),
                group_size_column: int(np.sum(group_mask)),
                "matched_topic": matched_topic,
                "matched_topic_share": float(topic_group_share.loc[group_value, matched_topic]),
                group_top_terms_column: ", ".join(group_terms),
                "topic_top_terms": ", ".join(topic_terms),
                "term_overlap": ", ".join(overlap) if overlap else "-",
                "overlap_count": len(overlap),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values(
        by=["matched_topic_share", "overlap_count"],
        ascending=[False, False],
    )
    return topic_group_counts, topic_group_share, comparison_df


def compare_topics_to_clusters(
    tfidf_matrix,
    feature_names,
    cluster_labels,
    dominant_topics: np.ndarray,
    topic_table: pd.DataFrame,
    top_n_terms: int = 10,
):
    return compare_topics_to_groups(
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
        group_labels=cluster_labels,
        dominant_topics=dominant_topics,
        topic_table=topic_table,
        top_n_terms=top_n_terms,
        group_name="cluster",
        exclude_group_value=-1,
    )


def plot_topic_group_heatmap(
    topic_group_share: pd.DataFrame,
    title: str,
    group_name: str = "Group",
    group_value_names: list[str] | tuple[str, ...] | dict[int, str] | None = None,
    topic_table: pd.DataFrame | None = None,
    topic_label_terms: int = 4,
    show_topic_terms: bool = False,
    r: str | None = None,
):
    plot_df = topic_group_share.copy()
    figure_height = max(800, 28 * len(plot_df.index) + 180)

    if group_value_names is not None:
        def resolve_group_name(group_value):
            if isinstance(group_value_names, dict):
                return str(group_value_names.get(group_value, group_value))
            group_index = int(group_value)
            if 0 <= group_index < len(group_value_names):
                return str(group_value_names[group_index])
            return str(group_value)

        plot_df.index = [resolve_group_name(group_value) for group_value in plot_df.index]

    topic_lookup = None
    if topic_table is not None:
        topic_lookup = topic_table.set_index("topic_id")["top_terms"].to_dict()

    renamed_columns = {}
    for topic_id in plot_df.columns:
        topic_label = f"Topic {int(topic_id)}"
        if show_topic_terms and topic_lookup is not None:
            topic_terms = topic_lookup.get(topic_id)
            if topic_terms:
                top_terms = [term.strip() for term in topic_terms.split(",")[:topic_label_terms]]
                topic_label = topic_label + "<br>" + "<br>".join(top_terms)
        renamed_columns[topic_id] = topic_label

    plot_df = plot_df.rename(columns=renamed_columns)

    fig = px.imshow(
        plot_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "LSA Topic", "y": group_name, "color": "Share"},
        title=title,
    )
    fig.update_layout(
        xaxis=dict(side="top"),
        yaxis=dict(automargin=True),
        margin=dict(l=280, r=40, t=100, b=40),
        width=950,
        height=figure_height,
    )
    fig.show(renderer=r)


def plot_topic_cluster_heatmap(
    topic_cluster_share: pd.DataFrame,
    title: str,
    group_value_names: list[str] | tuple[str, ...] | dict[int, str] | None = None,
    topic_table: pd.DataFrame | None = None,
    show_topic_terms: bool = False,
    r: str | None = None,
):
    plot_topic_group_heatmap(
        topic_group_share=topic_cluster_share,
        title=title,
        group_name="Cluster",
        group_value_names=group_value_names,
        topic_table=topic_table,
        show_topic_terms=show_topic_terms,
        r=r,
    )


def plot_topic_selection_curve(topic_selection: TopicSelectionResult, title: str, r: str | None = None):
    curve = topic_selection.diagnostics.copy()
    fig = px.line(
        curve,
        x="n_topics",
        y="explained_variance_ratio",
        markers=True,
        title=title,
        labels={
            "n_topics": "Number of topics",
            "explained_variance_ratio": "Cumulative explained variance",
        },
    )

    selected_row = curve[curve["selected"]]
    fig.add_scatter(
        x=selected_row["n_topics"],
        y=selected_row["explained_variance_ratio"],
        mode="markers",
        marker={"size": 12, "color": "#d62728"},
        name="Selected elbow",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(width=950, height=550)
    fig.show(renderer=r)
