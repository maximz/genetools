#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib

matplotlib.use("Agg")

from genetools import plots

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)


@pytest.fixture
def adata_obs():
    """Fixture that returns an anndata object.
    This downloads 5.9 MB of data upon the first call of the function and stores it in ./data/pbmc3k_raw.h5ad.
    Then processed version cached at ./data/pbmc3k.h5ad (not versioned) and .data/pbmc3k.obs.csv (versioned for reference plot consistency).

    To regenerate: rm ./data/pbmc3k.obs.csv
    """
    import scanpy as sc
    import pandas as pd
    import os

    # cache output of this fixture so we can make baseline figures
    if os.path.exists("data/pbmc3k.obs.csv"):
        return pd.read_csv("data/pbmc3k.obs.csv", index_col=0)

    # Following https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    adata = sc.datasets.pbmc3k()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    mito_genes = adata.var_names.str.startswith("MT-")
    adata.obs["percent_mito"] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(
        adata.X, axis=1
    )
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1
    adata = adata[adata.obs.n_genes < 2500, :]
    adata = adata[adata.obs.percent_mito < 0.05, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.louvain(adata)

    # these will be out of order
    new_cluster_names = [
        "CD4 T",
        "CD14 Monocytes",
        "B",
        "CD8 T",
        "NK",
        "FCGR3A Monocytes",
        "Dendritic",
        "Megakaryocytes",
    ]
    adata.rename_categories("louvain", new_cluster_names)

    # Pull umap into obs
    adata.obs["umap_1"] = adata.obsm["X_umap"][:, 0]
    adata.obs["umap_2"] = adata.obsm["X_umap"][:, 1]

    # Pull a gene value into obs
    adata.obs["CST3"] = adata[:, "CST3"].X

    adata.write("data/pbmc3k.h5ad", compression="gzip")
    adata.obs.to_csv("data/pbmc3k.obs.csv")

    return adata


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_umap_scatter_discrete(adata_obs):
    """Test umap_scatter with discrete hue."""
    fig, _ = plots.umap_scatter(
        data=adata_obs,
        umap_1_key="umap_1",
        umap_2_key="umap_2",
        hue_key="louvain",
        label_key="louvain",
    )
    return fig


@pytest.mark.mpl_image_compare
def test_umap_scatter_continuous(adata_obs):
    """Test umap_scatter with continouous hue."""
    fig, _ = plots.umap_scatter(
        data=adata_obs,
        umap_1_key="umap_1",
        umap_2_key="umap_2",
        hue_key="CST3",
        continuous_hue=True,
        label_key="louvain",
    )
    return fig


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_horizontal_stacked_bar_plot():
    df = pd.DataFrame(
        {
            "cluster": [
                "Cluster 1",
                "Cluster 1",
                "Cluster 2",
                "Cluster 2",
                "Cluster 3",
                "Cluster 3",
            ],
            "cell_type": ["B cell", "T cell", "B cell", "T cell", "B cell", "T cell"],
            "frequency": [0.25, 0.75, 15, 5, 250, 750],
        }
    )
    fig, _ = plots.horizontal_stacked_bar_plot(
        df,
        index_key="cluster",
        hue_key="cell_type",
        value_key="frequency",
        normalize=True,
    )
    return fig
