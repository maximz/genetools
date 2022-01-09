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
