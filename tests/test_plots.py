#!/usr/bin/env python

"""Tests for `genetools` genetools.plots.

Keep in mind when writing plotting tests:
- Use `@pytest.mark.mpl_image_compare` decorator to automatically do snapshot testing. See README.md for how to regenerate snapshots.
- `plt.tight_layout()` seems to produce different figure dimensions across different platforms.
    - To generate figures in a consistent way with Github Actions CI, we now run tests locally in a Debian-based Docker image as well.
- Some figures seem not to export right unless you save with tight bounding box (specifically legends outside figure are cut off):
    - `@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})`

We defined our own decorator for snapshot testing: @snapshot_image
"""

import pytest
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import hashlib
import packaging.version
import scipy.stats

import genetools
from genetools.palette import HueValueStyle

from .conftest import snapshot_image

random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)


@pytest.fixture
def rng():
    """Return a random number generator."""
    return np.random.default_rng(random_seed)


def _boxplot_extra_kwargs_for_new_seaborn(categorical_var: str) -> dict:
    """Seaborn v0.13 changed sns.boxplot behavior"""
    common_params = {
        # For some reason, seaborn v0.13 boxplots are now affected by mpl.rcParams which have:
        # "boxplot.whiskerprops.linestyle": "--" (dashed)
        # "boxplot.flierprops.marker": "+"
        # and more.
        # We can investigate rcParams more closely with this code:
        # import logging
        # logger = logging.getLogger(__name__)
        # import matplotlib
        # d = {k: v for k, v in matplotlib.rcParams.items() if k.startswith("boxplot")}
        # logger.info(d)
        # Let's apply those rcParams as consistent styles for our <v0.13 and >=v0.13 tests:
        "whiskerprops": {"linestyle": "solid"},
        "flierprops": {
            "marker": "+",
            "markeredgecolor": "k",
            "markeredgewidth": 1.0,
            "markerfacecolor": "auto",
            "markersize": 6.0,
        },
    }
    if packaging.version.parse(sns.__version__) >= packaging.version.parse("0.13.0"):
        # Boxplot hue and legend API change in seaborn 0.13+
        common_params |= dict(
            hue=categorical_var,
            legend=False,
        )
    return common_params


@snapshot_image
def test_scatterplot_discrete(adata):
    """Test scatterplot with discrete hue."""
    fig, ax = genetools.plots.scatterplot(
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        hue_key="louvain",
        marker_size=15,
        alpha=0.8,
        marker=".",
        legend_title="Cluster",
        remove_x_ticks=True,
        remove_y_ticks=True,
    )
    # Add cluster labels
    genetools.plots.superimpose_group_labels(
        ax,
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        label_key="louvain",
    )
    return fig


@snapshot_image
def test_scatterplot_continuous(adata):
    """Test scatterplot with continouous hue."""
    # also test supplying our own axes
    fig, ax = plt.subplots()
    genetools.plots.scatterplot(
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
    )
    # Add cluster labels
    genetools.plots.superimpose_group_labels(
        ax,
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        label_key="louvain",
    )
    plt.title("Scatterplot supports both discrete and continuous hues.")
    return fig


@snapshot_image
def test_scatterplot_no_hue(adata):
    """Test scatterplot with no hue, but many HueValueStyle defaults."""
    fig, ax = genetools.plots.scatterplot(
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        hue_key=None,
        marker_size=15,
        alpha=0.8,
        marker=".",
        marker_zorder=1,
        marker_size_scale_factor=5.0,
        legend_size_scale_factor=1.0,
        marker_face_color=None,
        marker_linewidths=2.0,
        legend_title="Cluster",
        remove_x_ticks=True,
        remove_y_ticks=True,
    )
    # Add cluster labels
    genetools.plots.superimpose_group_labels(
        ax,
        data=adata.obs,
        x_axis_key="umap_1",
        y_axis_key="umap_2",
        label_key="louvain",
        label_alpha=1.0,
        label_color="blue",
        label_size=30,
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
    fig, _ = genetools.plots.stacked_bar_plot(
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
        genetools.plots.stacked_bar_plot(
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
        + [{"cluster": "very/long/cluster/name/3", "expanded": "Not expanded"}] * 15
        + [{"cluster": "very/long/cluster/name/3", "expanded": "Expanded"}] * 15
    )
    fig, ax = genetools.plots.stacked_bar_plot(
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
        genetools.plots.wrap_tick_labels(
            ax, wrap_x_axis=True, wrap_y_axis=True, wrap_amount=10
        )

    assert (
        ax.get_xticklabels()[0].get_text() != "very long cluster name 1"
    ), "x tick labels should be wrapped"
    return fig


@pytest.fixture
def categorical_df(rng):
    # make more sick patients and have their distances be more dispersed
    n_healthy = 10
    n_sick = 20
    df = pd.DataFrame(
        {
            "distance": np.hstack(
                [rng.standard_normal(n_healthy), rng.standard_normal(n_sick) * 3]
            ),
            "disease type": ["Healthy"] * n_healthy + ["SARS-CoV-2 Patient"] * n_sick,
        }
    )
    df["distance"] += 10
    df["x"] = rng.standard_normal(df.shape[0])
    return df


@snapshot_image
def test_add_sample_size_to_labels(categorical_df):
    fig, ax = plt.subplots()
    sns.boxplot(
        data=categorical_df,
        x="distance",
        y="disease type",
        ax=ax,
        **_boxplot_extra_kwargs_for_new_seaborn(categorical_var="disease type"),
    )

    # Add sample size to labels
    ax.set_yticklabels(
        genetools.plots.add_sample_size_to_labels(
            labels=ax.get_yticklabels(), data=categorical_df, hue_key="disease type"
        )
    )
    assert ax.get_yticklabels()[0].get_text() == "Healthy\n($n=10$)"

    return fig


@snapshot_image
def test_add_sample_size_to_numerical_labels(categorical_df):
    # make sure we can add sample size to numerical labels
    fig, ax = plt.subplots()
    # make these discrete numerical categories
    categorical_df["distance_categorical"] = (
        categorical_df["distance"].apply(int).astype(int)
    )
    # plot a range of values for each numerical category
    sns.boxplot(
        data=categorical_df,
        x="distance_categorical",
        y="distance",
        ax=ax,
        palette=sns.color_palette(),
        **_boxplot_extra_kwargs_for_new_seaborn(categorical_var="distance_categorical"),
    )

    # Add sample size to labels
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            labels=ax.get_xticklabels(),
            data=categorical_df,
            hue_key="distance_categorical",
        )
    )
    assert (
        "\n($n=0$)" not in ax.get_xticklabels()[0].get_text()
    ), "could not find sample sizes for numerical categories"

    return fig


@snapshot_image
def test_add_sample_size_to_boolean_labels(categorical_df):
    # make sure we can add sample size to numerical labels
    fig, ax = plt.subplots()
    # make these discrete boolean categories
    categorical_df["disease type boolean"] = categorical_df["disease type"] == "Healthy"
    # plot a range of values for each numerical category
    sns.boxplot(
        data=categorical_df,
        x="disease type boolean",
        y="distance",
        ax=ax,
        **_boxplot_extra_kwargs_for_new_seaborn(categorical_var="disease type boolean"),
    )

    # Add sample size to labels
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            labels=ax.get_xticklabels(),
            data=categorical_df,
            hue_key="disease type boolean",
        )
    )
    assert (
        "\n($n=0$)" not in ax.get_xticklabels()[0].get_text()
    ), "could not find sample sizes for boolean categories"

    return fig


@snapshot_image
def test_wrap_labels_overrides_any_linebreaks_in_labels(categorical_df):
    fig, ax = plt.subplots()
    sns.boxplot(
        data=categorical_df,
        y="distance",
        x="disease type",
        ax=ax,
        **_boxplot_extra_kwargs_for_new_seaborn(categorical_var="disease type"),
    )

    # Add sample size to labels
    ax.set_xticklabels(
        genetools.plots.add_sample_size_to_labels(
            labels=ax.get_xticklabels(), data=categorical_df, hue_key="disease type"
        )
    )

    # Wrap y-axis text labels
    # The labels will have linebreaks already from previous step,
    # but wrap_tick_labels will remove them as needed to follow its wrap_amount parameter
    assert ax.get_xticklabels()[0].get_text() == "Healthy\n($n=10$)"  # line break
    genetools.plots.wrap_tick_labels(ax, wrap_x_axis=True, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break

    # Also confirm that rerunning has no further effect
    genetools.plots.wrap_tick_labels(ax, wrap_x_axis=False, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break
    genetools.plots.wrap_tick_labels(ax, wrap_x_axis=True, wrap_y_axis=False)
    assert (
        ax.get_xticklabels()[0].get_text() == "Healthy ($n=10$)"
    )  # no more line break

    return fig


@snapshot_image
def test_add_sample_size_to_legend(categorical_df):
    fig, ax = genetools.plots.scatterplot(
        data=categorical_df,
        x_axis_key="distance",
        y_axis_key="disease type",
        hue_key="disease type",
    )

    # Add sample size to legend
    genetools.plots.add_sample_size_to_legend(
        ax=ax, data=categorical_df, hue_key="disease type"
    )
    assert ax.get_legend().get_texts()[0].get_text() == "Healthy ($n=10$)"
    assert ax.get_legend().get_texts()[1].get_text() == "SARS-CoV-2 Patient ($n=20$)"

    return fig


# Marking as a manual snapshot test so it is skipped when not running in Docker or in Github Actions (controlled by --run-snapshots pytest flag)
@pytest.mark.snapshot_custom
def test_pdf_deterministic_output(tmp_path, snapshot):
    # Can't use snapshot_image here because pytest-mpl doesn't support PDF
    # So we are doing our own snapshot test md5 checksum here.
    #
    # Unlike pytest-mpl, which allows minute differences in the resulting image,
    # the MD5 checksum relies on exact output from matplotlib, which is not guaranteed to be stable.
    # The expected MD5 checksum will change based on matplotlib version unfortunately.
    # We use syrupy to store the expected MD5 checksum. Regenerate with pytest --snapshot-update
    #
    # This also allows us to test genetools.plots.savefig directly.
    #

    fname = tmp_path / "test_pdf_determinstic_output.pdf"

    fig = plt.figure()
    plt.scatter([0, 1], [0, 1], label="Series")
    plt.legend(loc="best")
    plt.title("Title")
    plt.xlabel("X axis")
    plt.xlabel("Y axis")

    genetools.plots.savefig(fig, fname)

    def get_md5(fname):
        with open(fname, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    observed_md5 = get_md5(fname)
    assert (
        observed_md5 == snapshot
    ), f"{fname} md5sum mismatch: got {observed_md5}, expected {snapshot}"


@snapshot_image
def test_palette_with_unfilled_shapes(rng):
    df = pd.DataFrame()
    df["x"] = rng.standard_normal(5)
    df["y"] = rng.standard_normal(5)
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
    fig, _ = genetools.plots.scatterplot(
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


#####


## Density plots:
# First we will define the data and showcase several ways of plotting it.
# Then we will snapshot test our special density function.


@pytest.fixture
def data():
    # Per numpy.histogram2d docs:
    # Generate non-symmetric test data
    n = 10000
    x = np.linspace(1, 100, n)
    y = 2 * np.log(x) + np.random.rand(n) - 0.5
    data = pd.DataFrame({"x": x, "y": y})
    return data


@snapshot_image
def test_scatter(data):
    ax = sns.scatterplot(data=data, x="x", y="y", alpha=0.5)
    return ax.get_figure()


@snapshot_image
def test_scatter_joint(data):
    g = sns.jointplot(data=data, x="x", y="y", alpha=0.5)
    return g.fig


@snapshot_image
def test_hexbin(data):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(
        data["x"],
        data["y"],
        gridsize=20,  # set grid size
        cmap="rainbow",
        # can also add: mintcnt=10
        # which is the minimum number of points in a hexagon to color it
        # i.e. the minimum count for background cells
        # https://stackoverflow.com/a/5405654/130164
    )
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


@snapshot_image
def test_overall_density(data):
    binned_data = scipy.stats.binned_statistic_2d(
        x=data["x"],
        y=data["y"],
        values=None,
        statistic="count",
        bins=20,
        expand_binnumbers=True,
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    plt.pcolormesh(pltX, pltY, binned_data.statistic.T, cmap="rainbow")
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


@snapshot_image
def test_overall_density_filtered(data):
    n_bins = 20
    quantile = 0.50

    binned_data = scipy.stats.binned_statistic_2d(
        x=data["x"],
        y=data["y"],
        values=None,
        statistic="count",
        bins=n_bins,
        expand_binnumbers=True,
    )

    # which bin does each point belong to
    bin_number_df = pd.DataFrame(binned_data.binnumber, index=["x_bin", "y_bin"]).T

    # filter out any beyond-edge bins that capture values outside bin bounds (e.g. due to range parameter)
    bin_number_df = bin_number_df[
        (bin_number_df["x_bin"] >= 1)
        & (bin_number_df["x_bin"] <= n_bins)
        & (bin_number_df["y_bin"] >= 1)
        & (bin_number_df["y_bin"] <= n_bins)
    ]

    # bin sizes: number of points per bin
    bin_sizes = bin_number_df.groupby(["x_bin", "y_bin"], observed=False).size()

    # Fill N/A counts
    bin_sizes = bin_sizes.reindex(
        pd.MultiIndex.from_product(
            [
                range(1, binned_data.statistic.shape[0] + 1),
                range(1, binned_data.statistic.shape[1] + 1),
            ],
            names=bin_sizes.index.names,
        ),
        fill_value=0,
    )

    # choose bins to remove: drop bins with low number of counts, i.e. low overall density
    bins_to_remove = bin_sizes[bin_sizes <= bin_sizes.quantile(quantile)].reset_index(
        name="size"
    )

    # Plot
    # Note need to transpose and handle one-indexed bin IDs.
    newstat = binned_data.statistic.T
    newstat[
        bins_to_remove["y_bin"].values - 1, bins_to_remove["x_bin"].values - 1
    ] = np.nan
    fig, ax = plt.subplots(figsize=(5, 5))
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    plt.pcolormesh(pltX, pltY, newstat, cmap="rainbow")
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    return fig


@snapshot_image
def test_relative_density(data):
    data2 = pd.concat(
        [
            data.assign(classname="A"),
            pd.DataFrame(
                {
                    "x": np.random.randint(1, 100, 1000),
                    "y": np.random.randint(1, 10, 1000),
                    "classname": "B",
                }
            ),
        ],
        axis=0,
    )
    fig, ax, _ = genetools.plots.two_class_relative_density_plot(
        data2,
        x_key="x",
        y_key="y",
        hue_key="classname",
        positive_class="A",
        colorbar_label="proportion",
        quantile=0.90,
    )
    ax.plot(data["x"], 2 * np.log(data["x"]), "k-")
    ax.set_title("Two-class relative density")
    return fig
    # TODO: add test we have same results with balanced_class_weights=True or False when class frequencies are identical (e.g. bump B class to 10000).
    # Maybe return statistic directly so we can compare.


####


@pytest.fixture
def dotplot_data():
    items = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for x in range(5):
        for y in range(10):
            items.append(
                # Generate mean (positive or negative]) and standard deviation (>= 0)
                {
                    "x": alphabet[x],
                    "y": alphabet[y],
                    "mean": np.random.randn(),
                    "std": np.random.rand(),
                }
            )

    data = pd.DataFrame(items)
    assert all(data["std"] >= 0)
    return data


@snapshot_image
def test_dotplot(dotplot_data):
    # Example with mean and standard deviation:
    # Circle color represents the mean.
    # Circle size represents stability (inverse of standard deviation).

    with sns.plotting_context("paper"):
        with sns.axes_style("white"):
            fig, _ = genetools.plots.plot_two_key_color_and_size_dotplot(
                data=dotplot_data,
                x_axis_key="x",
                y_axis_key="y",
                color_key="mean",
                size_key="std",
                inverse_size=True,  # Big circles are trustworthy/stable across the average, while little circles aren't
                color_legend_text="Mean",
                size_legend_text="Inverse std. dev.",
                min_marker_size=20,  # so that the smallest circle for zero standard deviation is still visible
                # set diverging colormap centered at 0
                # so that bold circles are strong effects, while near-white circles are weak effects
                color_cmap="RdBu_r",
                color_vcenter=0,
                figsize=(8, 6),
            )
            plt.title("Color and size dotplot")
            return fig


@snapshot_image
def test_dotplot_single_key(dotplot_data):
    # Example with mean and standard deviation:
    # Circle color represents the mean.
    # Circle size represents stability (inverse of standard deviation).

    with sns.plotting_context("paper"):
        with sns.axes_style("white"):
            fig, _ = genetools.plots.plot_color_and_size_dotplot(
                data=dotplot_data,
                x_axis_key="x",
                y_axis_key="y",
                value_key="mean",
                legend_text="Mean",
                figsize=(8, 6),
            )
            return fig


####


@snapshot_image
def test_make_and_plot_confusion_matrix():
    y_true = ["a", "a", "b", "b", "c"]
    y_pred = [1, 2, 3, 4, 1]
    cm = genetools.stats.make_confusion_matrix(
        y_true, y_pred, "Ground truth", "Predicted"
    )
    fig, _ = genetools.plots.plot_confusion_matrix(cm)
    return fig
