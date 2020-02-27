#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc

from genetools import scanpy_helpers


@pytest.fixture
def adata():
    """Fixture that returns an anndata object.
    This downloads 5.9 MB of data upon the first call of the function and stores it in ./data/pbmc3k_raw.h5ad.
    Then processed version cached at ./data/pbmc3k_2.h5ad (not versioned).

    To regenerate: rm ./data/pbmc3k_2.h5ad
    """
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import os

    # cache output of this fixture for local test speed
    cache_fname = "data/pbmc3k_2.h5ad"
    if os.path.exists(cache_fname):
        return sc.read(cache_fname)

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

    adata.write(cache_fname, compression="gzip")

    return adata


def test_find_all_markers(adata):
    cluster_markers_df = scanpy_helpers.find_all_markers(
        adata, cluster_key="louvain", pval_cutoff=0.05, log2fc_min=0.25
    )

    # make sure filtering worked right
    assert (
        not (cluster_markers_df["pvals_adj"] > 0.05).any().any()
    ), "high pvals should be filtered out"
    assert (
        not (cluster_markers_df["logfoldchanges"] < 0.25).any().any()
    ), "low logFC should be filtered out"

    # make sure all expected output columns are present
    expected_col_list = [
        "scores",
        "gene",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "louvain",
        "rank",
    ]
    assert np.array_equal(cluster_markers_df.columns, expected_col_list)

    # check cluster_key dtype (not categorical)
    assert (
        cluster_markers_df["louvain"].dtype
        == adata.obs["louvain"].cat.categories.dtype
        == "object"
    )

    # make sure all clusters are represented
    expected_clusters = adata.obs["louvain"].sort_values().unique()
    actual_clusters = cluster_markers_df["louvain"].sort_values().unique()
    assert np.array_equal(actual_clusters, expected_clusters)
