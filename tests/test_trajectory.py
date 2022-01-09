#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib

matplotlib.use("Agg")

from genetools import plots, trajectory, helpers

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

n_roots = 100
# TODO: multiple root clusters
root_cluster = "B"


@pytest.fixture(scope="module")
def roots(adata):
    # Choose roots
    return helpers.sample_cells_from_clusters(
        adata.obs, n_roots, "louvain", [root_cluster]
    )


def test_roots(adata, roots):
    # Confirm roots come from the right clusters
    assert len(roots) == n_roots
    assert all(adata[roots].obs["louvain"] == root_cluster)


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_plot_roots(adata, roots):
    # Plot root points over umap
    fig, _ = plots.umap_scatter(
        data=adata.obs,
        umap_1_key="umap_1",
        umap_2_key="umap_2",
        hue_key="louvain",
        label_key="louvain",
        highlight_cell_names=roots,
    )
    return fig


@pytest.fixture(scope="module")
def trajectories(adata, roots):
    # run pseudotime
    return trajectory.monte_carlo_pseudotime(adata, roots)


def test_trajectory_shape(trajectories, adata, roots):
    # Should have one value per cell, times the number of experiments
    assert trajectories.shape[0] == adata.obs.shape[0] * len(roots)


def test_trajectory_ranges(trajectories):
    assert all(trajectories["pseudotime"] >= 0.0)
    assert all(trajectories["pseudotime"] <= 1.0)
    assert not any(trajectories["pseudotime"].isna())


def test_trajectory_colnames(trajectories):
    expected_cols = ["cell_barcode", "root_cell", "pseudotime"]
    for col in expected_cols:
        assert col in trajectories.columns


def test_get_end_points(trajectories, roots):
    df = trajectory.get_cells_at_percentile(trajectories, 1.0)
    assert df.shape[0] == len(roots)
    assert all(df["pseudotime"]) == 1.0


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_stochasticity_plot(trajectories):
    return trajectory.plot_stochasticity(trajectories)


"""
Tutorial order:
- roots choose, plot, get value counts by cluster
- run trajectories
- end points get value counts by cluster, and plot
- stochasticity plot


TODO:

===

mean order
spectral order:
    n_trajectories_sample = 100  # number of trajectories to look at
    n_cells_sample = 4400  # number of cells to look at
    root_cell_key = 'root_cell'
    barcode_key = 'full_barcode'
    After running: left merge into the obs.
compare_trajectories


left merge mean_trajectory back


# def plot_trajectory():
#     fig, ax = genetools.plots.umap_scatter(
#         adata.obs,
#         hue_key="mean_trajectory",
#         continuous_hue=True,
#         label_key="cluster_label",
#         marker_size=0.1,
#         umap_1_key="coembed_umap_1",
#         umap_2_key="coembed_umap_2",
#         figsize=(8, 8),
#     )
#     ax.set_title("Mean pseudotime on Coembed")
#     savefig(fig, "out/scanpy/coembed.pseudotime.16wk.rob_coembed.umap.png", dpi=300)


# plot density
# fig = plot_pseudotime_density(adata, suptitle="Pseudotime Coembed (mean trajectory)", cluster_label_key="cluster_label",value_key="mean_trajectory")


"""
