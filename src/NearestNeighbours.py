import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


# this function is for plotting the graph
def plot_cluster_graph(
        G,
        grps,
        label_names,
        title="Cluster Graph",
        figsize=(16, 14),
        layout_k=0.08,
        iterations=100,
        node_size=10,
        edge_alpha=0.08,
        edge_width=0.15,
        cmap=plt.cm.tab20,
        seed=42,
        title_fontsize=22,
        title_y=0.86,

        legend_y=1.01,
        legend_ncols=4,
        legend_fontsize=11
):
    grps = np.array(grps)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    pos = nx.spring_layout(
        G,
        seed=seed,
        k=layout_k,
        iterations=iterations
    )
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        alpha=edge_alpha,
        width=edge_width,
        edge_color="gray"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=grps,
        cmap=cmap,
        node_size=node_size,
        alpha=0.9,
        linewidths=0
    )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        y=title_y
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    unique_groups = np.unique(grps)

    legend_elements = []

    for i, grp in enumerate(unique_groups):
        color = cmap(i / max(1, len(unique_groups) - 1))

        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=label_names[grp],
                markerfacecolor=color,
                markersize=8
            )
        )
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_ncols,
        fontsize=legend_fontsize,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.show()


def inspect_knn_components(knn_graph, directed=False, return_labels=True):
    n_components, labels = connected_components(
        csgraph=knn_graph,
        directed=directed,
        return_labels=True,
    )

    component_sizes = {
        comp_id: int((labels == comp_id).sum())
        for comp_id in range(n_components)
    }

    result = {
        "n_components": int(n_components),
        "component_sizes": component_sizes,
        "largest_component_size": max(component_sizes.values()),
        "smallest_component_size": min(component_sizes.values()),
        "is_connected": n_components == 1,
    }

    if return_labels:
        result["labels"] = labels

    return result


# this is for deciding the value of "nearest_neighbors"
def preprocessing_step1(features, is_mutual=False, k_min=3, k_max=50):
    k_values = []
    n_components_list = []
    isolated_nodes_list = []

    for k in range(k_min, k_max + 1):
        A = kneighbors_graph(
            features,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
            metric="cosine"
        )

        # this is for the Standard kNN
        if not is_mutual:
            A = A.maximum(A.T)

        # Mutual kNN where we only create an edge if there's a mutual neighbors (within k)
        else:
            A = A.minimum(A.T)

        G = nx.from_scipy_sparse_array(A)

        # subgraphs count of those
        n_components = nx.number_connected_components(G)
        # this is the count of the nodes where they have no edges (subgraph with only one node kind of)
        isolated_nodes = nx.number_of_isolates(G)

        k_values.append(k)
        n_components_list.append(n_components)
        isolated_nodes_list.append(isolated_nodes)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, n_components_list, marker="o", label="Connected components")
    plt.plot(k_values, isolated_nodes_list, marker="o", label="Isolated nodes")

    plt.xlabel("Number of neighbors k")
    plt.ylabel("Count")
    plt.title("Graph Connectivity Diagnostics Across k")
    plt.legend()
    plt.grid(True)
    plt.show()


def degree_plot(G, figsize=(12, 5)):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize
    )

    axes[0].plot(
        degree_sequence,
        color="royalblue",
        marker="o",
        markersize=3,
        linewidth=1
    )

    axes[0].set_title("Degree Rank Plot")
    axes[0].set_ylabel("Degree")
    axes[0].set_xlabel("Rank")

    unique_deg, counts = np.unique(
        degree_sequence,
        return_counts=True
    )

    axes[1].bar(
        unique_deg,
        counts,
        color="gray",
        edgecolor="black"
    )

    axes[1].set_title("Degree Histogram")
    axes[1].set_xlabel("Degree")
    axes[1].set_ylabel("# Nodes")

    plt.tight_layout()
    plt.show()


def plot_eigenvalues_and_eigengaps(A, n_eigs=50, top_k_spikes=3):
    L = laplacian(A, normed=True)
    eigenvalues, _ = eigsh(L, k=n_eigs, which="SM")
    eigenvalues = np.sort(eigenvalues)
    eigengaps = np.diff(eigenvalues)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(
        np.arange(1, len(eigenvalues) + 1),
        eigenvalues,
        marker="o"
    )

    axes[0].set_title("Eigenvalues of Normalized Laplacian")
    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True)

    axes[1].plot(
        np.arange(1, len(eigengaps) + 1),
        eigengaps,
        marker="o"
    )

    axes[1].set_title("Eigengap Plot")
    axes[1].set_xlabel("Gap after eigenvalue index i")
    axes[1].set_ylabel(r"$\lambda_{i+1} - \lambda_i$")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    top_indices = np.argsort(eigengaps)[::-1][:top_k_spikes]

    print(f"\nTop {top_k_spikes} Eigengap Spikes:")
    print("-" * 60)

    for idx in top_indices:
        gap = eigengaps[idx]

        print(
            f"Gap after λ_{idx + 1}: "
            f"{gap:.6f}  "
            f"=> possible number of clusters k = {idx + 1}"
        )


def plot_spectral_components(
        mutual_graph,
        max_components=16,
        figsize=(18, 12),
        sort_values=True,
):
    L = laplacian(mutual_graph, normed=True)
    eigenvalues, eigenvectors = eigsh(
        L,
        k=max_components + 1,
        which="SM"
    )
    idx = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvectors = eigenvectors[:, 1:]
    eigenvalues = eigenvalues[1:]

    n_components = eigenvectors.shape[1]

    cols = 4
    rows = int(np.ceil(n_components / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize
    )

    axes = np.array(axes).reshape(-1)

    for i in range(n_components):

        v = eigenvectors[:, i]

        if sort_values:
            v_plot = np.sort(v)
            x = np.arange(len(v_plot))
            xlabel = "Sorted node rank"
        else:
            v_plot = v
            x = np.arange(len(v_plot))
            xlabel = "Node index"

        axes[i].plot(x, v_plot)

        axes[i].set_title(
            rf"$\lambda_{{{i + 2}}}$ = {eigenvalues[i]:.4f}"
        )

        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel("Eigenvector value")
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle(
        "Spectral Components (Eigenvectors of Normalized Laplacian)",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

    return eigenvalues, eigenvectors
