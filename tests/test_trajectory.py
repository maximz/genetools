#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib

matplotlib.use("Agg")

from genetools import plots, trajectory

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)


"""
TODO:
choose roots

    # afterwards:
    # adata[roots].obs["cluster_label"].value_counts()
    # plot_umap_highlight_cells

===

- Plot root or end points. with umap_scatter

===
run

    # col_names, pseudotime_vals, many_pt = run_pseudotime(adata, roots, ["full_barcode", "cluster_label"])


===

get end points

    # then look at adata[roots].obs["cluster_label"].value_counts()
    # but first need to ['full_barcode'] it

===

stochasticity plot

===

mean order
spectral order:
    n_trajectories_sample = 100  # number of trajectories to look at
    n_cells_sample = 4400  # number of cells to look at
    root_cell_key = 'root_cell'
    barcode_key = 'full_barcode'
    After running: left merge into the obs.
compare_trajectories



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


"""
