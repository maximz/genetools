#!/usr/bin/env python

"""Tests for `genetools` plots.

Keep in mind when writing plotting tests:
- Use `@pytest.mark.mpl_image_compare` decorator to automatically do snapshot testing. See README.md for how to regenerate snapshots.
- `plt.tight_layout()` seems to produce different figure dimensions across different platforms.
    - To generate figures in a consistent way with Github Actions CI, we now run tests locally in a Debian-based Docker image as well.
- Some figures seem not to export right unless you save with tight bounding box (specifically legends outside figure are cut off):
    - `@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})`
"""

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt

from genetools.palette import HueValueStyle

matplotlib.use("Agg")

from genetools import plots

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_scatterplot_discrete(adata):
    """Test scatterplot with discrete hue."""
    fig, _ = plots.scatterplot(
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        hue_key="louvain",
        alpha=0.8,
        legend_title="Cluster",
        label_key="louvain",
        remove_x_ticks=True,
        remove_y_ticks=True,
    )
    return fig


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_scatterplot_continuous(adata):
    """Test scatterplot with continouous hue."""
    # also test supplying our own axes
    fig, ax = plt.subplots()
    fig, _ = plots.scatterplot(
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        hue_key="CST3",
        continuous_hue=True,
        ax=ax,
        alpha=0.8,
        legend_title="Cluster",
        equal_aspect_ratio=True,
        label_key="louvain",
    )
    return fig


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_stacked_bar_plot():
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
    fig, _ = plots.stacked_bar_plot(
        df,
        index_key="cluster",
        hue_key="cell_type",
        value_key="frequency",
        normalize=True,
    )
    return fig


@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_stacked_bar_plot_autocompute_frequencies():
    df = pd.DataFrame(
        [{"disease": "Covid", "cluster": 1, "expanded": "Not expanded"}] * 10
        + [{"disease": "Covid", "cluster": 1, "expanded": "Expanded"}] * 20
        + [{"disease": "Covid", "cluster": 2, "expanded": "Not expanded"}] * 50
        + [{"disease": "Covid", "cluster": 2, "expanded": "Expanded"}] * 5
        + [{"disease": "Covid", "cluster": 3, "expanded": "Not expanded"}] * 15
        + [{"disease": "Covid", "cluster": 3, "expanded": "Expanded"}] * 15
        ###
        + [{"disease": "Healthy", "cluster": 1, "expanded": "Not expanded"}] * 5
        + [{"disease": "Healthy", "cluster": 1, "expanded": "Expanded"}] * 45
        + [{"disease": "Healthy", "cluster": 2, "expanded": "Not expanded"}] * 15
        + [{"disease": "Healthy", "cluster": 2, "expanded": "Expanded"}] * 17
        + [{"disease": "Healthy", "cluster": 3, "expanded": "Not expanded"}] * 8
        + [{"disease": "Healthy", "cluster": 3, "expanded": "Expanded"}] * 3
    )
    # also test unnormalized, vertical, with specific hue_order and palette and legend title
    palette = {
        "Not expanded": HueValueStyle(color="#0ec7ff"),
        "Expanded": HueValueStyle(color="#ff0ec3", hatch="//"),
    }

    # plot per disease using subplots with shared X axis
    fig, axarr = plt.subplots(
        figsize=(4, 8), nrows=df["disease"].nunique(), sharex=True, sharey=False
    )
    for ix, (ax, (disease, grp)) in enumerate(zip(axarr, df.groupby("disease"))):
        # only plot legend for top-most axis
        enable_legend = ix == 0
        plots.stacked_bar_plot(
            grp,
            index_key="cluster",
            hue_key="expanded",
            ax=ax,
            normalize=False,
            vertical=True,
            palette=palette,
            hue_order=["Not expanded", "Expanded"],
            axis_label="Number of cells",
            legend_title="Status",
            enable_legend=enable_legend,
        )
        ax.set_title(disease)

    return fig
