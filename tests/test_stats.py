#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal

from genetools import stats


def test_rank_normalize_numpy():
    actual = stats.rank_normalize(np.array([1000, -500, 300]))
    expected = np.array([3, 1, 2])
    assert np.array_equal(expected, actual)


def test_rank_normalize_list():
    actual = stats.rank_normalize([1000, -500, 300])
    expected = np.array([3, 1, 2])
    assert np.array_equal(expected, actual)


def test_rank_normalize_series():
    """This tests accept_series decorator."""
    actual = stats.rank_normalize(pd.Series([1000, -500, 300]))
    expected = pd.Series([3, 1, 2])
    assert expected.equals(actual)


def test_percentile_normalize():
    actual = stats.percentile_normalize(np.array([1000, -500, 300]))
    expected = [1, 1 / 3, 2 / 3]
    assert np.array_equal(expected, actual)


def test_normalize_rows():
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3)))
    row_sums = stats.normalize_rows(df).sum(axis=1)
    assert len(row_sums) == 5, "sanity check"
    assert np.allclose(row_sums, 1.0)


def test_normalize_cols():
    df = pd.DataFrame(np.random.randint(0, 100, size=(5, 3)))
    column_sums = stats.normalize_columns(df).sum(axis=0)
    assert len(column_sums) == 3, "sanity check"
    assert np.allclose(column_sums, 1.0)


def test_self_coclustering():
    """Coclustering sanity check: dataset against itself should always yield 100%
    """
    cluster_ids = [1, 1, 2, 2]
    # 6 potential relationships in each dataset: 1-2, 1-3, 1-4, 2-3, 2-4, 3-4
    # Actual coclustering in each dataset: 1-2, 3-4
    # Shared coclusterings across datasets: 1-2, 3-4 (100%)
    assert (
        stats.coclustering(cluster_ids, cluster_ids)
        == stats._coclustering_slow(cluster_ids, cluster_ids)
        == 1.0
    )


def test_coclustering():
    """Coclustering example:
    - 2 Coclusterings in dataset 1: Cells 1+2; Cells 3+4
    - 3 Coclusterings in dataset 2: Cells 1+2; Cells 2+3; Cells 1+3
    - 4 unique total coclusterings
    - 1 shared coclusterings: Cells 1+2
    - So percent of cells-in-same-cluster relationships from one dataset that hold in the other: 1/4
    """

    cluster_ids1 = ["Cluster1", "Cluster1", "Cluster2", "Cluster2", "Cluster3"]
    cluster_ids2 = ["ClusterA", "ClusterA", "ClusterA", "Cluster2", "Cluster3"]
    assert (
        stats.coclustering(cluster_ids1, cluster_ids2)
        == stats._coclustering_slow(cluster_ids1, cluster_ids2)
        == 1 / 4
    )


def test_coclustering_2():
    """Coclustering example:
    - 2 coclusterings in dataset 1: cells 1+2, cells 5+6
    - 4 coclusterings in dataset 2: 1+2, 2+3, 1+3, 5+6
    - That's 5 unique coclusterings across both datasets: 1+2, 1+3, 2+3, 3+4, 5+6
    - Two are shared: 1+2, 5+6
    - Result: 2/5
    """
    cluster_ids1 = [
        "Cluster1",
        "Cluster1",
        "Cluster2",
        "Cluster2",
        "Cluster3",
        "Cluster3",
    ]
    cluster_ids2 = [
        "ClusterA",
        "ClusterA",
        "ClusterA",
        "Cluster2",
        "Cluster3",
        "Cluster3",
    ]
    assert (
        stats.coclustering(cluster_ids1, cluster_ids2)
        == stats._coclustering_slow(cluster_ids1, cluster_ids2)
        == 2 / 5
    )


def test_coclustering_3():
    """Coclustering example (cell pairs are one-indexed):
    - Cell pair (1, 2) is coclustered in both datasets
    - Cell pair (3, 4) is coclustered in dataset 1 only
    - Cell pair (4, 5) is coclustered in dataset 2 only
    - All 7 other cell pairs are not coclustered in either dataset
    """
    cluster_ids1 = [0, 0, 1, 1, 2]
    cluster_ids2 = ["a", "a", "b", "c", "c"]
    assert (
        stats.coclustering(cluster_ids1, cluster_ids2)
        == stats._coclustering_slow(cluster_ids1, cluster_ids2)
        == 1 / 3
    )


def test_intersect_marker_genes():
    reference_marker_genes = {
        "CD4 T cells": ["T cell marker 1", "T cell marker 2", "CD4 marker"],
        "CD8 T cells": ["T cell marker 1", "T cell marker 2", "CD8 marker"],
        "B cells": ["B cell marker 1", "B cell marker 2"],
    }
    noise_genes = ["Noise %d" % i for i in range(10)]
    query_lists = {
        "Cluster 1": ["T cell marker 1", "CD4 marker"] + noise_genes,
        "Cluster 2": ["T cell marker 1", "CD8 marker"] + noise_genes,
        "Cluster 3": ["T cell marker 2", "CD8 marker"] + noise_genes,
        "Cluster 4": ["T cell marker 1", "CD8 marker", "B cell marker 1"] + noise_genes,
        "Cluster 5": noise_genes,
    }
    expected_map = {
        "Cluster 1": "CD4 T cells",
        "Cluster 2": "CD8 T cells",
        "Cluster 3": "CD8 T cells-1",  # duplicates renamed
        "Cluster 4": "CD8 T cells-2",  # duplicates renamed
        "Cluster 5": "CD4 T cells (maybe)",  # low confidence match
    }
    expected_df = pd.DataFrame(
        {
            "query_cluster_id": [
                "Cluster 1",
                "Cluster 2",
                "Cluster 3",
                "Cluster 4",
                "Cluster 5",
            ],
            "new_cluster": [
                "CD4 T cells",
                "CD8 T cells",
                "CD8 T cells-1",
                "CD8 T cells-2",
                "CD4 T cells (maybe)",
            ],
            "low_confidence_call": [False, False, False, False, True],
            "highest_score": [2 / 3, 2 / 3, 2 / 3, 2 / 3, 0],
            "next_match": [
                "CD8 T cells",
                "CD4 T cells",
                "CD4 T cells",
                "B cells",
                "CD8 T cells",
            ],
        }
    ).set_index("query_cluster_id")
    results, label_map, low_confidence_percentage = stats.intersect_marker_genes(
        reference_marker_genes, query_lists, low_confidence_suffix=" (maybe)"
    )
    assert label_map == expected_map
    assert low_confidence_percentage == 1 / 5
    # ignoring dtypes
    assert_frame_equal(results, expected_df, check_dtype=False)
    # with fixed dtypes (pandas>=1.0.0 feature)
    assert results.convert_dtypes().equals(expected_df.convert_dtypes())


def test_marker_gene_conversion():
    # The conversion suggested in intersect_marker_genes doctsring
    df = pd.DataFrame({"cluster": ["a", "a", "b"], "gene": [1, 2, 1]})
    actual = df.groupby("cluster")["gene"].apply(list).to_dict()
    expected = {"a": [1, 2], "b": [1]}
    assert expected == actual


@pytest.mark.xfail(raises=AssertionError)
def test_intersect_marker_genes_too_small_reference():
    stats.intersect_marker_genes(
        {"CD4 T cells": ["T cell marker 1", "T cell marker 2", "CD4 marker"]}, {}
    )


@pytest.mark.xfail(raises=AssertionError)
def test_intersect_marker_genes_too_small_reference_2():
    stats.intersect_marker_genes({}, {})
