#!/usr/bin/env python

"""Tests for `genetools` package."""

import numpy as np
import anndata
import scipy

from genetools import scanpy_helpers, stats


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

    # make sure all expected output columns are present (order doesn't matter)
    expected_col_list = [
        "scores",
        "gene",
        "logfoldchanges",
        "pvals",
        "pvals_adj",
        "louvain",
        "rank",
    ]
    assert set(cluster_markers_df.columns) == set(expected_col_list)

    # check cluster_key dtype (not categorical)
    assert (
        cluster_markers_df["louvain"].dtype
        == adata.obs["louvain"].cat.categories.dtype
        == "object"
    )

    # make sure all clusters are represented
    assert set(adata.obs["louvain"]) == set(cluster_markers_df["louvain"])


def test_clr_normalization():
    """This tests genetools.stats.clr_normalize and genetools.scanpy_helpers.clr_normalize"""
    # make anndata with 20 cells x 14 antibodies
    # try sparse and dense versions to make sure they work identically
    rows = []
    for _ in range(20):
        rows.append(
            np.array([0, 0, 0, np.nan] + list(np.random.randint(0, 25, size=10)))
        )

    dense = np.vstack(rows)
    dense_adata = anndata.AnnData(X=dense)
    dense_adata_by_cell = dense_adata.copy()

    sparse = scipy.sparse.lil_matrix(dense)
    sparse_adata = anndata.AnnData(X=sparse)

    assert sparse_adata.X.shape == dense_adata.X.shape == (20, 14)
    assert scipy.sparse.issparse(sparse_adata.X)
    assert not scipy.sparse.issparse(dense_adata.X)

    scanpy_helpers.clr_normalize(sparse_adata)
    scanpy_helpers.clr_normalize(dense_adata)
    scanpy_helpers.clr_normalize(dense_adata_by_cell, axis=1)

    # No longer sparse!
    assert not scipy.sparse.issparse(sparse_adata.X)
    assert not scipy.sparse.issparse(dense_adata.X)

    assert sparse_adata.X.shape == dense_adata.X.shape == dense.shape == sparse.shape

    assert np.allclose(sparse_adata.X, dense_adata.X, equal_nan=True)

    # Confirm the values match running the function manually over a column (a protein)
    manual_run_2nd_col = stats._seurat_clr(dense[:, 1])
    assert manual_run_2nd_col.shape[0] == 20
    assert np.allclose(manual_run_2nd_col, dense_adata.X[:, 1], equal_nan=True)

    # Confirm the values match running the function manually over a row (a cell)
    manual_run_2nd_row = stats._seurat_clr(dense[1, :])
    assert manual_run_2nd_row.shape[0] == 14
    assert np.allclose(manual_run_2nd_row, dense_adata_by_cell.X[1, :], equal_nan=True)

    # Also test "inplace" parameter
    dense_adata2 = anndata.AnnData(X=dense)
    dense_adata3 = scanpy_helpers.clr_normalize(dense_adata2, inplace=False)
    assert np.allclose(dense_adata.X, dense_adata3.X, equal_nan=True)
    assert not np.allclose(dense_adata2.X, dense_adata3.X, equal_nan=True)
