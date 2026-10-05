"""Heatmaps and clustering of models and cross-workload errors."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

from ..model import COEF_COLS
from .common import save

CMAP = "coolwarm_r"


def coefficient_distances(models):
    """Euclidean distance between workloads' coefficient vectors."""
    labels = models["workload"].tolist()
    return pd.DataFrame(squareform(pdist(models[COEF_COLS].to_numpy(dtype=float))), index=labels, columns=labels)


def plot_model_distances(models, split, out_dir, exclude=()):
    """Coefficient distance matrix of one split's models: plain, clustered on the
    distance rows, and clustered by the coefficient vectors themselves."""
    models = models[~models["workload"].isin(exclude)]
    distances = coefficient_distances(models)
    name = split.lower()

    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=False, xticklabels=True, yticklabels=True)
    plt.title(f"{split}: Coefficient Distance Matrix")
    plt.tight_layout()
    save(plt, out_dir / f"{name}_coefficient_distance.pdf")

    grid = sns.clustermap(distances, metric="euclidean", method="average", cmap=CMAP, linewidths=0.5)
    grid.fig.suptitle(f"{split}: Clustered Coefficient Distance", y=1.02)
    save(grid, out_dir / f"{name}_coefficient_distance_clustered.pdf")

    Z = linkage(pdist(models[COEF_COLS].to_numpy(dtype=float)), method="average")
    grid = sns.clustermap(distances, row_linkage=Z, col_linkage=Z, cmap=CMAP, linewidths=0.5)
    grid.fig.suptitle(f"{split}: Clustered by Coefficient Vectors", y=1.02)
    save(grid, out_dir / f"{name}_coefficient_vectors_clustered.pdf")


def plot_error_matrix(errors, name, out_dir, exclude=()):
    """Cross-workload error matrix (rows: data, columns: model): heatmap,
    dendrogram of the data workloads and their clustered distance matrix."""
    errors = errors.drop(columns=[c for c in exclude if c in errors.columns])
    errors = errors.dropna(axis=0, how="any").dropna(axis=1, how="any")

    plt.figure(figsize=(10, 8))
    im = plt.imshow(errors.values, aspect="equal", origin="upper", cmap=CMAP)
    plt.colorbar(im, label="Error (%)")
    plt.xticks(ticks=np.arange(len(errors.columns)), labels=errors.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(errors.index)), labels=errors.index)
    plt.tight_layout()
    save(plt, out_dir / f"{name}_heatmap.pdf", dpi=300)

    plt.figure(figsize=(10, 5))
    dendrogram(linkage(errors.values, method="average", metric="euclidean"), labels=errors.index.tolist(),
               orientation="top")
    plt.xticks(rotation=90)
    plt.title(f"{name.capitalize()} Dendrogram")
    plt.tight_layout()
    save(plt, out_dir / f"{name}_dendrogram.pdf", dpi=300)

    distances = pd.DataFrame(squareform(pdist(errors.values)), index=errors.index, columns=errors.index)
    order = distances.index[leaves_list(linkage(distances.values, method="average"))]
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances.loc[order, order], cmap=CMAP, square=True, cbar_kws={"label": "Euclidean distance"})
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    save(plt, out_dir / f"{name}_distance_matrix_clustered_heatmap.pdf")


def plot_transformation(transformation, out_dir, exclude=()):
    T = transformation.drop(index=[x for x in exclude if x in transformation.index],
                            columns=[x for x in exclude if x in transformation.columns])
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(T.values, aspect="auto", cmap=CMAP)
    ax.set_xticks(np.arange(len(T.columns)))
    ax.set_yticks(np.arange(len(T.index)))
    ax.set_xticklabels(T.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(T.index, fontsize=7)
    ax.set_title("Transformation Matrix Heatmap")
    plt.colorbar(im).set_label("Transformation Weight")
    plt.tight_layout()
    save(plt, out_dir / "transformation_matrix_heatmap.pdf", dpi=300, bbox_inches="tight")
