import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import textwrap

from typing import Tuple, Union, List, Dict

from .palette import HueValueStyle, convert_palette_list_to_dict

# Pulled this out of function so we can use it in tests directly.
_savefig_defaults = {
    # Handle legends outside of figure
    # see https://github.com/mwaskom/seaborn/blob/77e3b6b03763d24cc99a8134ee9a6f43b32b8e7b/seaborn/axisgrid.py#L63
    "bbox_inches": "tight",
    # Determinstic PDF output:
    # see https://matplotlib.org/2.1.1/users/whats_new.html#reproducible-ps-pdf-and-svg-output
    # and https://github.com/matplotlib/matplotlib/issues/6317/
    # and https://github.com/matplotlib/matplotlib/pull/6597
    # and https://github.com/matplotlib/matplotlib/pull/7748
    # Supposedly this should have done the job, but it doesn't seem to work:
    # 'metadata': {'creationDate': None}
}


def savefig(fig: matplotlib.figure.Figure, *args, **kwargs):
    """
    Save figure with smart defaults:

    * Tight bounding box -- necessary for legends outside of figure
    * Determinsistic PDF output by fixing SOURCE_DATE_EPOCH to Jan 1, 2000
    * Editable text objects when outputing a vector PDF

    Example usage: ``genetools.plots.savefig(fig, "my_plot.png", dpi=300)``.

    Any positional or keyword arguments are passed to ``matplotlib.pyplot.savefig``.

    :param fig: Figure to save.
    :type fig: matplotlib.figure.Figure
    """
    # combine the two dictionaries
    kwargs = {**_savefig_defaults, **kwargs}

    # Determinsitic PDF output: set SOURCE_DATE_EPOCH to a constant value temporarily
    # Per docs, passing metadata to savefig should have done the job, but it doesn't seem to work.
    original_source_date_epoch = os.environ.pop("SOURCE_DATE_EPOCH", None)
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"  # 2000 Jan 01 00:00:00 UTC

    try:
        # To ensure text is editable when we save figures in vector format,
        # set fonttype to Type 42 (TrueType)
        # per https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files
        # and https://stackoverflow.com/a/54111532/130164
        with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(*args, **kwargs)
    finally:
        # Restore SOURCE_DATE_EPOCH to original value
        if original_source_date_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = original_source_date_epoch


def scatterplot(
    data,
    x_axis_key,
    y_axis_key,
    hue_key,
    continuous_hue=False,
    continuous_cmap="viridis",
    discrete_palette: Union[
        Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]]
    ] = None,
    ax: matplotlib.axes.Axes = None,
    figsize=(8, 8),
    marker_size=25,
    alpha=1.0,
    na_color="lightgray",
    marker="o",
    marker_edge_color="none",
    enable_legend=True,
    legend_hues=None,
    legend_title=None,
    sort_legend_hues=True,
    autoscale=True,
    equal_aspect_ratio=False,
    plotnonfinite=False,
    label_key=None,
    label_z_order=100,
    label_color="k",
    label_alpha=0.8,
    label_size=15,
    remove_x_ticks=False,
    remove_y_ticks=False,
    tight_layout=True,
    despine=True,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Scatterplot colored by a discrete or continuous "hue" grouping variable.

    For discrete hues, pass continuous_hue=False and a dictionary of colors and/or HueValueStyle objects in discrete_palette.

    Figure size will grow beyond the figsize parameter setting, because the legend is pulled out of figure.
    So you must use ``fig.savefig('filename', bbox_inches='tight')``.
    This is provided automatically by ``genetools.plots.savefig(fig, 'filename')``.

    If using with scanpy, to get umap data from adata.obsm into adata.obs, try:

    .. code-block:: python

        data = adata.obs.assign(umap_1=adata.obsm["X_umap"][:, 0], umap_2=adata.obsm["X_umap"][:, 1])

    :param data: Input data, e.g. anndata.obs
    :type data: pandas.DataFrame
    :param x_axis_key: Column name to plot on X axis
    :type x_axis_key: str
    :param y_axis_key: Column name to plot on Y axis
    :type y_axis_key: str
    :param hue_key: Column name with hue groups that will be used to color points
    :type hue_key: str
    :param continuous_hue: Whether the hue column takes continuous or discrete/categorical values, defaults to False.
    :type continuous_hue: bool, optional
    :param continuous_cmap: Colormap to use for plotting continuous hue grouping variable, defaults to "viridis"
    :type continuous_cmap: str, optional
    :param discrete_palette: Palette of colors and/or HueValueStyle objects to use for plotting discrete/categorical hue groups, defaults to None. Supply a matplotlib palette name, list of colors, or dict mapping hue values to colors or to HueValueStyle objects (or a mix of the two).
    :type discrete_palette: ``Union[ Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]] ]``, optional
    :param ax: Existing matplotlib Axes to plot on, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param figsize: Size of figure to generate if no existing ax was provided, defaults to (8, 8)
    :type figsize: tuple, optional
    :param marker_size: Base marker size. Maybe scaled by individual HueValueStyles. Defaults to 25
    :type marker_size: int, optional
    :param alpha: Default point transparency, unless overriden by a HueValueStyle, defaults to 1.0
    :type alpha: float, optional
    :param na_color: Fallback color to use for discrete hue categories that do not have an assigned style in discrete_palette, defaults to "lightgray"
    :type na_color: str, optional
    :param marker: Default marker style, unless overriden by a HueValueStyle, defaults to "o". For plots with many points, try "." instead.
    :type marker: str, optional
    :param marker_edge_color: Default marker edge color, unless overriden by a HueValueStyle, defaults to "none" (no edge border drawn). Another common choice is "face", so the edge color matches the face color.
    :type marker_edge_color: str, optional
    :param enable_legend: Whether legend (or colorbar if continuous_hue) should be drawn. Defaults to True. May want to disable if plotting multiple subplots/panels.
    :type enable_legend: bool, optional
    :param legend_hues: Optionally override the list of hue values to include in legend, e.g. to add any hue values missing from the plotted subset of data; defaults to None
    :type legend_hues: list, optional
    :param legend_title: Specify a title for the legend. Defaults to None, in which case the hue_key is used.
    :type legend_title: str, optional
    :param sort_legend_hues: Enable sorting of legend hues, defaults to True
    :type sort_legend_hues: bool, optional
    :param autoscale: Enable automatic zoom in, defaults to True
    :type autoscale: bool, optional
    :param equal_aspect_ratio: Plot with equal aspect ratio, defaults to False
    :type equal_aspect_ratio: bool, optional
    :param plotnonfinite: For continuous hues, whether to plot points with inf or nan value, defaults to False
    :type plotnonfinite: bool, optional
    :param label_key: Optional column name specifying group text labels to superimpose on plot, defaults to None
    :type label_key: str, optional
    :param label_z_order: Z-index for superimposed group text labels, defaults to 100
    :type label_z_order: int, optional
    :param label_color: Color for superimposed group text labels, defaults to "k"
    :type label_color: str, optional
    :param label_alpha: Opacity for superimposed group text labels, defaults to 0.8
    :type label_alpha: float, optional
    :param label_size: Text size of superimposed group labels, defaults to 15
    :type label_size: int, optional
    :param remove_x_ticks: Remove X axis tick marks and labels, defaults to False
    :type remove_x_ticks: bool, optional
    :param remove_y_ticks: Remove Y axis tick marks and labels, defaults to False
    :type remove_y_ticks: bool, optional
    :raises ValueError: Must specify correct number of colors if supplying a custom palette
    :param tight_layout: whether to format the figure with tight_layout, defaults to True
    :type tight_layout: bool, optional
    :param despine: whether to despine (remove the top and right figure borders), defaults to True
    :type despine: bool, optional
    :return: Matplotlib Figure and Axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    if ax is None:
        # Make figure from scratch
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # Passed in an existing ax
        fig = ax.get_figure()

    # store PathCollection returned by ax.scatter
    scattered_object = None

    default_style = HueValueStyle(
        color=na_color, marker=marker, edgecolors=marker_edge_color, alpha=alpha
    )

    if continuous_hue:
        # plot continuous variable with a colorbar
        scattered_object = ax.scatter(
            data[x_axis_key].values,
            data[y_axis_key].values,
            c=data[hue_key].values,
            cmap=continuous_cmap,
            **default_style.render_scatter_continuous_props(marker_size=marker_size),
            plotnonfinite=plotnonfinite,
            **kwargs,
        )

    else:
        # plot discrete hue
        if legend_hues is None:
            # if all possible legend hues weren't provided, this implies the user does not expect any legend hues beyond what is available in this slice
            legend_hues = data[hue_key].unique()

        if discrete_palette is None:
            discrete_palette = sns.color_palette("bright", n_colors=len(legend_hues))

        # if discrete_palette isn't a dict, then it must be a list
        # the trouble now is that we want to reserve slots for all of legend_hues, even if some are not in this slice of the dataset (imagine multiple axes and running this method on a slice at a time to plot on each axis)
        # so preregister the colors for all legend_hues
        # to do so, we convert palette into a dict with color assignments for each possible hue value.
        discrete_palette = convert_palette_list_to_dict(
            discrete_palette, legend_hues, sort_hues=True
        )

        for hue_value, hue_df in data.groupby(hue_key, observed=True):
            # look up marker style for this hue value, if supplied in palette
            # if what's in palette is just a color, not a HueValueStyle, cast into HueValueStyle, i.e. apply default marker style with palette color
            # if this hue value is not a key in the palette dict at all, then fall back to na_color color and default marker style
            marker_style: HueValueStyle = HueValueStyle.from_color(
                discrete_palette.get(hue_value, default_style)
            )
            # set defaults using passed in parameters
            marker_style = marker_style.apply_defaults(default_style)

            scattered_object = ax.scatter(
                hue_df[x_axis_key].values,
                hue_df[y_axis_key].values,
                **marker_style.render_scatter_props(marker_size=marker_size),
                plotnonfinite=plotnonfinite,
                **kwargs,
            )

    # Run tight_layout before adding legend,
    # especially before adding inset_axes colorbar (which wouldn't be included in tight_layout anyway, but may throw error on some matplotlib versions)
    # https://github.com/matplotlib/matplotlib/issues/21749
    if tight_layout:
        fig.tight_layout()

    if enable_legend:
        if continuous_hue:
            # color bar
            # see also https://stackoverflow.com/a/44642014/130164
            # pull colorbar out of axis by creating a special axis for the colorbar
            # specify width and height relative to parent bbox
            colorbar_ax = inset_axes(
                ax,
                width="5%",
                height="80%",
                loc="center left",
                bbox_to_anchor=(1.05, 0.0, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )

            colorbar = fig.colorbar(scattered_object, cax=colorbar_ax)
            if legend_title is not None:
                colorbar_ax.set_title(legend_title)

        else:
            # Create legend, and add any missing colors
            legend_handles = []
            for hue_value in legend_hues:
                # look up marker style for this hue value, if supplied in palette
                # if what's in palette is just a color, not a HueValueStyle, cast into HueValueStyle, i.e. apply default marker style with palette color
                # if this hue value is not a key in the palette dict at all, then fall back to na_color color and default marker style
                marker_style: HueValueStyle = HueValueStyle.from_color(
                    discrete_palette.get(hue_value, default_style)
                )
                # set defaults using passed in parameters
                marker_style = marker_style.apply_defaults(default_style)

                # Convert all to string so we can sort even if mixed input types like ["M", "F", np.nan]
                hue_value = str(hue_value)

                legend_handles.append(
                    ax.scatter(
                        [],
                        [],
                        label=hue_value,
                        **marker_style.render_scatter_legend_props(),
                    )
                )

            # Sort legend
            if sort_legend_hues:
                legend_handles = sorted(
                    legend_handles, key=lambda handle: handle.get_label()
                )

            leg = ax.legend(
                # what items to include in legend
                handles=legend_handles,
                # place legend outside figure
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
                borderaxespad=0.0,
                # dot size
                markerscale=1.5,
                # no border
                frameon=False,
                # transparent background
                framealpha=0.0,
                # legend title
                title=legend_title,
                # legend title font properties: TODO: requires newer matplotlib:
                # title_fontproperties={"weight": "bold", "size": "medium"},
                # Configure number of points in legend
                numpoints=1,
                scatterpoints=1,
            )
            # set legend title to bold - workaround for title_fontproperties missing from old matplotlib versions
            leg.set_title(title=legend_title, prop={"weight": "bold", "size": "medium"})
            # align legend title left
            leg._legend_box.align = "left"

    # add cluster labels
    if label_key is not None:
        for label, grp in data.groupby(label_key, observed=True):
            ax.annotate(
                f"{label}",
                grp[[x_axis_key, y_axis_key]].mean().values,  # mean of x and y
                horizontalalignment="center",
                verticalalignment="center_baseline",
                size=label_size,
                weight="bold",
                alpha=label_alpha,
                color=label_color,
                zorder=label_z_order,
                # set background color https://stackoverflow.com/a/23698794/130164
                bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "white"},
            )

    if autoscale:
        # automatic zoom in
        ax.autoscale_view()

    if equal_aspect_ratio:
        ax.set_aspect("equal", "datalim")

    # Turn off tick marks and labels
    if remove_x_ticks:
        ax.set_xticks([])
        ax.set_xticklabels([])
    if remove_y_ticks:
        ax.set_yticks([])
        ax.set_yticklabels([])

    if despine:
        sns.despine(ax=ax)

    return fig, ax


def stacked_bar_plot(
    data,
    index_key,
    hue_key,
    value_key=None,
    ax: matplotlib.axes.Axes = None,
    figsize=(8, 8),
    normalize=True,
    vertical=False,
    palette: Union[
        Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]]
    ] = None,
    na_color="lightgray",
    hue_order=None,
    axis_label="Frequency",
    enable_legend=True,
    legend_title=None,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Stacked bar chart.

    The ``index_key`` groups form the bars, and the ``hue_key`` groups subdivide the bars.
    The ``value_key`` determines the subdivision sizes, and is computed automatically if not provided.

    See https://observablehq.com/@d3/stacked-normalized-horizontal-bar for inspiration and colors.

    Figure size will grow beyond the figsize parameter setting, because the legend is pulled out of figure.
    So you must use ``fig.savefig('filename', bbox_inches='tight')``.
    This is provided automatically by ``genetools.plots.savefig(fig, 'filename')``.

    :param data: Plot data containing at minimum the columns identified by ``index_key``, ``hue_key``, and optionally ``value_key``.
    :type data: pandas.DataFrame
    :param index_key: Column name defining the rows.
    :type index_key: str
    :param hue_key: Column name defining the horizontal bar categories.
    :type hue_key: str
    :param value_key: Column name defining the bar sizes. If not supplied, this method will calculate group frequencies automatically
    :type value_key: str, optional.
    :param ax: Existing matplotlib Axes to plot on, defaults to None
    :type ax: matplotlib.axes.Axes, optional
    :param figsize: Size of figure to generate if no existing ax was provided, defaults to (8, 8)
    :type figsize: tuple, optional
    :param normalize: Normalize each row's frequencies to sum to 1, defaults to True
    :type normalize: bool, optional
    :param vertical: Plot stacked bars vertically, defaults to False (horizontal)
    :type vertical: bool, optional
    :param palette: Palette of colors and/or HueValuStyle objects to style the bars corresponding to each hue value, defaults to None (in which case default palette used). Supply a matplotlib palette name, list of colors, or dict mapping hue values to colors or to HueValueStyle objects (or a mix of the two).
    :type palette: ``Union[ Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]] ]``, optional
    :param na_color: Fallback color to use for hue values that do not have an assigned style in palette, defaults to "lightgray"
    :type na_color: str, optional
    :param hue_order: Optionally specify order of bar subdivisions. This order is applied from the beginning (bottom or left) to the end (top or right) of the bar. Defaults to None
    :type hue_order: list, optional
    :param axis_label: Label for the axis along which the frequency values are drawn, defaults to "Frequency"
    :type axis_label: str, optional
    :param enable_legend: Whether legend should be drawn. Defaults to True. May want to disable if plotting multiple subplots/panels.
    :type enable_legend: bool, optional
    :param legend_title: Specify a title for the legend. Defaults to None, in which case the hue_key is used.
    :type legend_title: str, optional
    :raises ValueError: Must specify correct number of colors if supplying a custom palette
    :return: Matplotlib Figure and Axes
    :rtype: (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """

    if value_key is None:
        # calculate frequency ourselves
        value_key = "frequency"
        data = (
            data.groupby(index_key)[hue_key]
            .value_counts()
            .rename(value_key)
            .reset_index()
        )

    plot_df = data[[index_key, value_key, hue_key]].copy()

    # Convert all fields to string or categorical dtypes so we can sort, even if mixed input types like strings and nan,
    # and so they behave categorically, not numerically.
    for col in [index_key, hue_key]:
        plot_df[col] = plot_df[col].astype("category")

    if normalize:
        # Normalize values to sum to 1 per row
        plot_df[value_key] = plot_df.groupby(index_key, sort=False)[value_key].apply(
            lambda g: g / g.sum()
        )

    # Sort so we maintain consistent order before we calculate cumulative value.
    # This ensures that the hues are in the same order for every bar.
    # This also applies the hue_order if one was supplied, because we set the categories as ordered.
    if hue_order is None:
        hue_order = sorted(plot_df[hue_key].unique())
    plot_df[hue_key] = plot_df[hue_key].cat.set_categories(hue_order, ordered=True)
    plot_df = plot_df.sort_values([index_key, hue_key])

    # Accumulate value with every subsequent box/hue as we go across each index/row
    # These will become row-level "left offsets" for each hue
    cum_value_key = value_key + "_cumulative_value"
    plot_df[cum_value_key] = plot_df.groupby(index_key, sort=False)[value_key].cumsum()

    # create colors
    if palette is None:
        # Use default palette if none provided
        palette = sns.color_palette("muted", n_colors=len(hue_order))

    # if palette isn't a dict, then it must be a list. Convert to dict, i.e. register color for each hue value
    palette = convert_palette_list_to_dict(palette, hue_order, sort_hues=False)

    with sns.axes_style("white"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            # Passed in an existing ax
            fig = ax.get_figure()

        plot_func = ax.bar if vertical else ax.barh

        # Go hue-by-hue, and plot down the rows
        # Use observed=True in case hue column is categorical and some expected categories are missing
        for hue_name in hue_order:
            hue_data = plot_df[plot_df[hue_key] == hue_name]

            # look up marker style for this hue value, if supplied in palette
            # if what's in palette is just a color, not a HueValueStyle, cast into HueValueStyle, i.e. apply default marker style with palette color
            # if this hue value is not a key in the palette dict at all, then fall back to na_color color and default marker style
            hue_style: HueValueStyle = HueValueStyle.from_color(
                palette.get(hue_name, HueValueStyle(color=na_color))
            )

            plot_func(
                # convert to string in case this was a numerical column, so matplotlib plots it as a categorical variable
                hue_data[index_key].values.astype(str),
                hue_data[value_key].values,
                align="center",
                label=hue_name,
                **hue_style.render_rectangle_props(),
                **{
                    ("bottom" if vertical else "left"): (
                        hue_data[cum_value_key] - hue_data[value_key]
                    ).values,
                    ("width" if vertical else "height"): 0.25,
                },
                snap=False,  # Disable alignment to pixel boundaries, which means bars can get hidden if figure size too small: https://github.com/matplotlib/matplotlib/issues/8808#issuecomment-311349594
            )

        if enable_legend:
            legend_title = legend_title if legend_title is not None else hue_key

            # get current legend items
            handles, labels = ax.get_legend_handles_labels()
            if vertical:
                # When plotting vertical stacked bar plot:
                # We plot from bottom to top, so the top-most bar is the last one in the legend list.
                # It makes more sense to reverse the legend order, so top-most bar corresponds to top-most legend item.
                handles = reversed(handles)
                labels = reversed(labels)

            leg = ax.legend(
                handles=handles,
                labels=labels,
                # place legend outside figure
                bbox_to_anchor=(1.05, 0.5),
                loc="center left",
                borderaxespad=0.0,
                # no border
                frameon=False,
                # transparent background
                framealpha=0.0,
                # legend title
                title=legend_title,
                # legend title font properties
                # TODO: requires newer matplotlib:
                # title_fontproperties={"weight": "bold", "size": "medium"},
            )
            # set legend title to bold - workaround for title_fontproperties missing from old matplotlib versions
            leg.set_title(title=legend_title, prop={"weight": "bold", "size": "medium"})
            # align legend title left
            leg._legend_box.align = "left"

        if vertical:
            ax.set_ylabel(axis_label)

            # tight bounds
            ax.set_xlim(min(ax.get_xticks()) - 1, max(ax.get_xticks()) + 1)
        else:
            ax.set_xlabel(axis_label)

            # tight bounds
            ax.set_ylim(min(ax.get_yticks()) - 1, max(ax.get_yticks()) + 1)

            # invert y-axis so that labels are in sorted order from top to bottom
            ax.invert_yaxis()

    sns.despine(ax=ax)
    return fig, ax


def wrap_tick_labels(
    ax: matplotlib.axes.Axes, wrap_x_axis=True, wrap_y_axis=True, wrap_amount=20
) -> matplotlib.axes.Axes:
    """Add text wrapping to tick labels on x and/or y axes on any plot.

    May override existing line breaks in tick labels.

    :param ax: existing plot with tick labels to be wrapped
    :type ax: matplotlib.axes.Axes
    :param wrap_x_axis: whether to wrap x-axis tick labels, defaults to True
    :type wrap_x_axis: bool, optional
    :param wrap_y_axis: whether to wrap y-axis tick labels, defaults to True
    :type wrap_y_axis: bool, optional
    :param wrap_amount: length of each line of text, defaults to 20
    :type wrap_amount: int, optional
    :return: plot with modified tick labels
    :rtype: matplotlib.axes.Axes
    """

    # At this point, ax.get_xticklabels() may return empty tick labels and emit UserWarning: FixedFormatter should only be used together with FixedLocator
    # It seems this happens for numerical axes specifically.
    # Must draw the canvas to position the ticks: https://stackoverflow.com/a/41124884/130164
    # And must assign tick locations prior to assigning tick labels, i.e. set_ticks(get_ticks()): https://stackoverflow.com/a/68794383/130164
    ax.get_figure().canvas.draw()

    def wrap_labels(labels):
        for label in labels:
            label.set_text("\n".join(textwrap.wrap(label.get_text(), wrap_amount)))
        return labels

    if wrap_x_axis:
        # Wrap x-axis text labels
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(wrap_labels(ax.get_xticklabels()))

    if wrap_y_axis:
        # Wrap y-axis text labels
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(wrap_labels(ax.get_yticklabels()))

    return ax


def add_sample_size_to_labels(labels: list, data: pd.DataFrame, hue_key: str) -> list:
    """Add sample size to tick labels on any plot with categorical groups.

    Sample size for each label is extracted from the ``hue_key`` column of dataframe ``data``.

    Pairs well with ``genetools.plots.wrap_tick_labels(ax)``.

    Example usage:

    .. code-block:: python

        ax.set_xticklabels(
            genetools.plots.add_sample_size_to_labels(
                ax.get_xticklabels(),
                df,
                "Group"
            )
        )

    :param labels: list of tick labels corresponding to groups in ``data[hue_key]``
    :type labels: list
    :param data: dataset with categorical groups
    :type data: pd.DataFrame
    :param hue_key: column name specifying categorical groups in dataset ``data``
    :type hue_key: str
    :return: modified tick labels with group sample sizes attached
    :rtype: list
    """

    def _make_label(hue_value):
        sample_size = data[data[hue_key] == hue_value].shape[0]
        return f"{hue_value}\n($n={sample_size}$)"

    return [_make_label(label.get_text()) for label in labels]


####


# def stacked_density_plot(
#     data,
#     row_var,
#     hue_var,
#     value_var,
#     col_var=None,
#     overlap=False,
#     suptitle=None,
#     figsize=None,
#     hue_order=None,
#     row_order=None,
#     palette=None,
# ):
#     """
#     Multiple density plot.
#     Adapted from old work at https://github.com/hammerlab/infino/blob/develop/analyze_cut.py#L912

#     For row_order, consider row_order=reversed(list(range(data.ylevel.values.max()+1)))
#     """

#     with sns.plotting_context("notebook"):
#         with sns.axes_style("white", rc={"axes.facecolor": (0, 0, 0, 0)}):
#             g = sns.FacetGrid(
#                 data,
#                 row=row_var,
#                 hue=hue_var,
#                 col=col_var,
#                 row_order=row_order,
#                 hue_order=hue_order,
#                 aspect=15,
#                 height=0.5,
#                 palette=palette,
#                 sharey=False,  # important -- they don't share y ranges.
#             )

#             ## Draw the densities in a few steps
#             # this is the shaded area
#             g.map(sns.kdeplot, value_var, clip_on=False, shade=True, alpha=0.8, lw=2)

#             # this is the dividing horizontal line
#             g.map(plt.axhline, y=0, lw=2, clip_on=False, ls="dashed")

#             ### Add label for each facet.

#             def label(**kwargs):
#                 """
#                 kwargs is e.g.: {'color': (0.4918017777777778, 0.25275644444444445, 0.3333333333333333), 'label': 'Name of the row'}
#                 """
#                 color = kwargs["color"]
#                 label = kwargs["label"]
#                 ax = plt.gca()  # map() changes current axis repeatedly
#                 # x=1 if plot_on_right else 0; ha="right" if plot_on_right else "left",
#                 ax.text(
#                     1.25,
#                     0.5,
#                     label,
#                     #                         fontweight="bold",
#                     color=color,
#                     #                     ha="right",
#                     ha="left",
#                     va="center",
#                     transform=ax.transAxes,
#                     fontsize="x-small",
#                     #                                                        fontsize='x-large', #15,
#                     #                             bbox=dict(facecolor='yellow', alpha=0.3)
#                 )

#             g.map(label)

#             ## Beautify the plot.
#             g.set(xlim=(-0.01, 1.01))
#             # seems to do the trick along with sharey=False
#             g.set(ylim=(0, None))

#             # Some `subplots_adjust` line is necessary. without this, nothing appears
#             if not overlap:
#                 g.fig.subplots_adjust(hspace=0)

#             # Remove axes details that don't play will with overlap
#             g.set_titles("")
#             # g.set_titles(col_template="{col_name}", row_template="")
#             g.set(yticks=[], ylabel="")
#             g.despine(bottom=True, left=True)

#             # fix x axis
#             g.set_xlabels("Pseudotime")

#             # resize
#             if figsize:
#                 g.fig.set_size_inches(figsize[0], figsize[1])
#             else:
#                 cur_size = g.fig.get_size_inches()
#                 increase_vertical = 3  # 7 #4 # 3
#                 g.fig.set_size_inches(cur_size[0], cur_size[1] + increase_vertical)

#             if suptitle is not None:
#                 g.fig.suptitle(suptitle, fontsize="medium")

#             # tighten
#             g.fig.tight_layout()

#             # then reoverlap
#             if overlap:
#                 g.fig.subplots_adjust(hspace=-0.1)

#             return g, g.fig


# TODO: density umap plot
# TODO: two class density plots.
