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
import seaborn as sns
import warnings
import hashlib

from genetools import plots
from genetools.palette import HueValueStyle

matplotlib.use("Agg")

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

# Define our own decorator for snapshot testing.
snapshot_image = pytest.mark.mpl_image_compare(savefig_kwargs=plots._savefig_defaults)


@snapshot_image
def test_scatterplot_discrete(adata):
    """Test scatterplot with discrete hue."""
    fig, _ = plots.scatterplot(
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        hue_key="louvain",
        marker_size=15,
        alpha=0.8,
        marker=".",
        legend_title="Cluster",
        label_key="louvain",
        remove_x_ticks=True,
        remove_y_ticks=True,
    )
    return fig


@snapshot_image
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
        marker_size=None,
        alpha=0.8,
        marker=".",
        legend_title="Cluster",
        equal_aspect_ratio=True,
        label_key="louvain",
    )
    return fig


@snapshot_image
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


@snapshot_image
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


@snapshot_image
def test_wrap_axis_labels():
    df = pd.DataFrame(
        [{"cluster": "very long cluster name 1", "expanded": "Not expanded"}] * 10
        + [{"cluster": "very long cluster name 1", "expanded": "Expanded"}] * 20
        + [{"cluster": "very long cluster name 2", "expanded": "Not expanded"}] * 50
        + [{"cluster": "very long cluster name 2", "expanded": "Expanded"}] * 5
        + [{"cluster": "very long cluster name 3", "expanded": "Not expanded"}] * 15
        + [{"cluster": "very long cluster name 3", "expanded": "Expanded"}] * 15
    )
    fig, ax = plots.stacked_bar_plot(
        df,
        index_key="cluster",
        hue_key="expanded",
        figsize=(8, 8),
        normalize=False,
        vertical=True,
    )

    # it's tempting to do a before-vs-after comparison of label text directly,
    # but tick labels are not actually available until the plot is drawn (see the comments in wrap_tick_labels()),
    # and forcing plot drawing for the test would interfere with actually testing that wrap_tick_labels does that on its own.

    # Wrap axis text labels
    # Confirm there is no UserWarning: FixedFormatter should only be used together with FixedLocator
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plots.wrap_tick_labels(ax, wrap_x_axis=True, wrap_y_axis=True, wrap_amount=10)

    assert (
        ax.get_xticklabels()[0].get_text() != "very long cluster name 1"
    ), "x tick labels should be wrapped"
    return fig


@snapshot_image
def test_add_sample_size_to_labels():
    # make more sick patients and have their distances be more dispersed
    n_healthy = 10
    n_sick = 20
    df = pd.DataFrame(
        {
            "distance": np.hstack(
                [np.random.randn(n_healthy), np.random.randn(n_sick) * 3]
            ),
            "disease type": ["Healthy"] * n_healthy + ["SARS-CoV-2 Patient"] * n_sick,
        }
    )
    df["distance"] += 10
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="distance", y="disease type", ax=ax)

    # Add sample size to labels
    ax.set_yticklabels(
        plots.add_sample_size_to_labels(
            labels=ax.get_yticklabels(), data=df, hue_key="disease type"
        )
    )
    assert ax.get_yticklabels()[0].get_text() == "Healthy\n($n=10$)"

    return fig


@snapshot_image
def test_wrap_labels_overrides_any_linebreaks_in_labels():
    # make more sick patients and have their distances be more dispersed
    n_healthy = 10
    n_sick = 20
    df = pd.DataFrame(
        {
            "distance": np.hstack(
                [np.random.randn(n_healthy), np.random.randn(n_sick) * 3]
            ),
            "disease type": ["Healthy"] * n_healthy + ["SARS-CoV-2 Patient"] * n_sick,
        }
    )
    df["distance"] += 10
    fig, ax = plt.subplots()
    sns.boxplot(data=df, y="distance", x="disease type", ax=ax)

    # Add sample size to labels
    ax.set_xticklabels(
        plots.add_sample_size_to_labels(
            labels=ax.get_xticklabels(), data=df, hue_key="disease type"
        )
    )

    # Wrap y-axis text labels
    # The labels will have linebreaks already from previous step,
    # but wrap_tick_labels will remove them as needed to follow its wrap_amount parameter
    assert ax.get_xticklabels()[0].get_text() == "Healthy\n($n=10$)"  # line break
    plots.wrap_tick_labels(ax, wrap_x_axis=True, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break

    # Also confirm that rerunning has no further effect
    plots.wrap_tick_labels(ax, wrap_x_axis=False, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break
    plots.wrap_tick_labels(ax, wrap_x_axis=True, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break

    return fig


def test_pdf_deterministic_output(tmp_path):
    # Can't use snapshot_image here because pytest-mpl doesn't support PDF
    # So we are doing our own snapshot test md5 checksum here
    # This also allows us to test genetools.plots.savefig directly.

    fname = tmp_path / "test_pdf_determinstic_output.pdf"

    fig = plt.figure()
    plt.scatter([0, 1], [0, 1], label="Series")
    plt.legend(loc="best")
    plt.title("Title")
    plt.xlabel("X axis")
    plt.xlabel("Y axis")

    plots.savefig(fig, fname)

    def get_md5(fname):
        with open(fname, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    # A hacky snapshot test: put expected md5sum here
    expected_md5 = "4fae9ebe9cb8f837aed495fee12ca179"
    observed_md5 = get_md5(fname)
    assert (
        expected_md5 == observed_md5
    ), f"{fname} md5sum mismatch: got {observed_md5}, expected {expected_md5}"


@snapshot_image
def test_palette_with_unfilled_shapes():
    df = pd.DataFrame()
    df["x"] = np.random.randn(5)
    df["y"] = np.random.randn(5)
    df["hue"] = "groupA"
    # HueValueStyle demo: unfilled shapes. Set facecolors to "none" and set edgecolors to desired color.
    palette = {
        "groupA": HueValueStyle(
            color=sns.color_palette("bright")[0],
            edgecolors=sns.color_palette("bright")[0],
            facecolors="none",
            marker="^",
            marker_size_scale_factor=1.5,
            linewidths=1.5,
            zorder=10,
        )
    }
    fig, _ = plots.scatterplot(
        data=df,
        x_axis_key="x",
        y_axis_key="y",
        hue_key="hue",
        discrete_palette=palette,
        marker="o",
        marker_edge_color="none",
        marker_size=None,
    )
    return fig
