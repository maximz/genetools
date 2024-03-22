import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import textwrap

from typing import Tuple, Union, List, Dict

from typing import Optional

import genetools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.cm
import matplotlib.figure
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.ticker


from genetools.palette import HueValueStyle, convert_palette_list_to_dict

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
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    hue_key: str = None,
    continuous_hue=False,
    continuous_cmap="viridis",
    discrete_palette: Union[
        Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]]
    ] = None,
    ax: matplotlib.axes.Axes = None,
    figsize=(8, 8),
    marker_size=25,
    alpha: float = 1.0,
    na_color="lightgray",
    marker: str = "o",
    marker_edge_color: str = "none",
    marker_zorder: int = 1,
    marker_size_scale_factor: float = 1.0,
    legend_size_scale_factor: float = 1.0,
    marker_face_color: str = None,
    marker_linewidths: float = None,
    enable_legend=True,
    legend_hues: List[str] = None,
    legend_title: str = None,
    sort_legend_hues=True,
    autoscale=True,
    equal_aspect_ratio=False,
    plotnonfinite=False,
    remove_x_ticks=False,
    remove_y_ticks=False,
    tight_layout=True,
    despine=True,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Scatterplot colored by a discrete or continuous "hue" grouping variable.

    For discrete hues, pass ``continuous_hue = False`` and a dictionary of colors and/or HueValueStyle objects in ``discrete_palette``.

    Figure size will grow beyond the figsize parameter setting, because the legend is pulled out of figure.
    So you must use ``fig.savefig('filename', bbox_inches='tight')``.
    This is provided automatically by ``genetools.plots.savefig(fig, 'filename')``.

    If using with scanpy, to join umap data from ``adata.obsm`` with other plot data in ``adata.obs``, try:

    .. code-block:: python

        data = adata.obs.assign(umap_1=adata.obsm["X_umap"][:, 0], umap_2=adata.obsm["X_umap"][:, 1])

    If ``hue_key = None``, then all points will be colored by ``na_color``
    and styled with parameters ``alpha``, ``marker``, ``marker_size``, ``zorder``, and ``marker_edge_color``.
    The legend will be disabled.

    :param data: Input data, e.g. anndata.obs
    :type data: pandas.DataFrame
    :param x_axis_key: Column name to plot on X axis
    :type x_axis_key: str
    :param y_axis_key: Column name to plot on Y axis
    :type y_axis_key: str
    :param hue_key: Column name with hue groups that will be used to color points. defaults to None to color all points consistently.
    :type hue_key: str, optional
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
    :param marker_zorder: Default marker z-order, unless overriden by a HueValueStyle, defaults to 1
    :type marker_zorder: int, optional
    :param marker_size_scale_factor: Default marker size scale factor, unless overriden by a HueValueStyle, defaults to 1.0
    :type marker_size_scale_factor: float, optional
    :param legend_size_scale_factor: Default legend size scale factor, unless overriden by a HueValueStyle, defaults to 1.0
    :type legend_size_scale_factor: float, optional
    :param marker_face_color: Default marker face color, unless overriden by a HueValueStyle, defaults to None (uses point color).
    :type marker_face_color: str, optional
    :param marker_linewidths: Default marker line widths, unless overriden by a HueValueStyle, defaults to None
    :type marker_linewidths: float, optional
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
        alpha=alpha,
        color=na_color,
        marker=marker,
        edgecolors=marker_edge_color,
        zorder=marker_zorder,
        marker_size_scale_factor=marker_size_scale_factor,
        legend_size_scale_factor=legend_size_scale_factor,
        facecolors=marker_face_color,
        linewidths=marker_linewidths,
    )

    if hue_key is None:
        scattered_object = ax.scatter(
            data[x_axis_key].values,
            data[y_axis_key].values,
            **default_style.render_scatter_props(marker_size=marker_size),
            plotnonfinite=plotnonfinite,
            **kwargs,
        )
    elif continuous_hue:
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

    if enable_legend and hue_key is not None:
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

            fig.colorbar(scattered_object, cax=colorbar_ax)
            if legend_title is not None:
                colorbar_ax.set_title(legend_title)

            # set global "current axes" back to main axes,
            # so that any calls like plt.title target main ax rather than inset colorbar_ax
            plt.sca(ax)

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
    value_key: Optional[str] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize=(8, 8),
    normalize=True,
    vertical=False,
    palette: Optional[
        Union[Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]]]
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
    :param palette: Palette of colors and/or HueValueStyle objects to style the bars corresponding to each hue value, defaults to None (in which case default palette used). Supply a matplotlib palette name, list of colors, or dict mapping hue values to colors or to HueValueStyle objects (or a mix of the two).
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
        data = data.groupby(index_key)[hue_key].value_counts().rename(value_key)

        # If categoricals, pandas < 1.5 loses hue_key column name: https://github.com/pandas-dev/pandas/issues/44324
        # There may have also been a brief 1.4.x regression that caused a similar issue for non-categoricals: "Bug in DataFrameGroupBy.value_counts() where subset had no effect (GH46383)"
        # So let's make sure the index names are set right.
        # TODO: remove this once pandas 1.5 is widely used.
        data.index.names = [index_key, hue_key]

        data = data.reset_index()

    plot_df = data[[index_key, value_key, hue_key]].copy()

    # Convert all fields to string or categorical dtypes so we can sort, even if mixed input types like strings and nan,
    # and so they behave categorically, not numerically.
    for col in [index_key, hue_key]:
        plot_df[col] = plot_df[col].astype("category")

    if normalize:
        # Normalize values to sum to 1 per row
        plot_df[value_key] = plot_df.groupby(index_key)[value_key].transform(
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
    plot_df[cum_value_key] = plot_df.groupby(index_key)[value_key].cumsum()

    # create colors
    if palette is None:
        # Use default palette if none provided
        palette = sns.color_palette("muted", n_colors=len(hue_order))

    # if palette isn't a dict, then it must be a list. Convert to dict, i.e. register color for each hue value
    # this also validates that there are enough colors for each hue value
    palette = convert_palette_list_to_dict(palette, hue_order, sort_hues=False)

    with sns.axes_style("white"):
        if ax is None:
            # Normally we would unpack thuple as fig, ax = plt.subplots(), but we want to set ax type explicitly here
            _tuple = plt.subplots(figsize=figsize)
            fig = _tuple[0]
            ax: matplotlib.axes.Axes = _tuple[1]
        else:
            # Passed in an existing ax
            fig: matplotlib.figure.Figure = ax.get_figure()

        plot_func = ax.bar if vertical else ax.barh

        # Go hue-by-hue, and plot down the rows
        # TODO: Add a test case for some expected hue_order entry missing from the data's hue column
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
    ax: matplotlib.axes.Axes,
    wrap_x_axis=True,
    wrap_y_axis=True,
    wrap_amount=20,
    break_characters=["/"],
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
    :param break_characters: characters at which to encourage to breaking text into lines, defaults to ['/']. set to None or [] to disable.
    :type break_characters: list, optional
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
            original_text = label.get_text()
            if break_characters is not None:
                # encourage breaking at this character. e.g. convert "/" to "/ " to encourage line break there.
                for break_character in break_characters:
                    break_character_stripped = break_character.strip()
                    original_text = original_text.replace(
                        break_character_stripped, f"{break_character_stripped} "
                    )
            label.set_text("\n".join(textwrap.wrap(original_text, wrap_amount)))
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


def superimpose_group_labels(
    ax: matplotlib.axes.Axes,
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    label_key: str,
    label_z_order=100,
    label_color="k",
    label_alpha=0.8,
    label_size=15,
) -> matplotlib.axes.Axes:
    """Add group (cluster) labels to existing plot.

    :param ax: matplotlib Axes for existing plot
    :type ax: matplotlib.axes.Axes
    :param data: [description]
    :type data: pd.DataFrame
    :param x_axis_key: Column name to plot on X axis
    :type x_axis_key: str
    :param y_axis_key: Column name to plot on Y axis
    :type y_axis_key: str
    :param label_key: Column name specifying categorical group text labels to superimpose on plot, defaults to None
    :type label_key: str, optional
    :param label_z_order: Z-index for superimposed group text labels, defaults to 100
    :type label_z_order: int, optional
    :param label_color: Color for superimposed group text labels, defaults to "k"
    :type label_color: str, optional
    :param label_alpha: Opacity for superimposed group text labels, defaults to 0.8
    :type label_alpha: float, optional
    :param label_size: Text size of superimposed group labels, defaults to 15
    :type label_size: int, optional
    :return: matplotlib Axes with superimposed group labels
    :rtype: matplotlib.axes.Axes
    """
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

    # Labels start as strings
    # Convert labels into the dtype of the column we will compare them against
    # E.g. if the labels represent numerical categories, we cast to a numerical type before comparison
    labels_cast_to_correct_dtype = pd.Series(
        [label.get_text() for label in labels]
    ).astype(data[hue_key].dtype)

    # Make new labels
    return [_make_label(label) for label in labels_cast_to_correct_dtype]


def add_sample_size_to_legend(
    ax: matplotlib.axes.Axes, data: pd.DataFrame, hue_key: str
) -> matplotlib.axes.Axes:
    """Add sample size to legend labels on any plot with categorical hues.

    Sample size for each label is extracted from the ``hue_key`` column of dataframe ``data``.

    Example usage:

    .. code-block:: python

        fig, ax = genetools.plots.scatterplot(
            data=df,
            x_axis_key="x",
            y_axis_key="y",
            hue_key="Group"
        )
        genetools.plots.add_sample_size_to_legend(
            ax=ax,
            data=df,
            hue_key="Group"
        )

    :param ax: matplotlib Axes for existing plot
    :type ax: matplotlib.axes.Axes
    :param data: dataset with categorical groups
    :type data: pd.DataFrame
    :param hue_key: column name specifying categorical groups in dataset ``data``
    :type hue_key: str
    :return: matplotlib Axes with modified legend labels with group sample sizes attached
    :rtype: matplotlib.axes.Axes
    """

    def _make_label(hue_value):
        sample_size = data[data[hue_key] == hue_value].shape[0]
        return f"{hue_value} ($n={sample_size}$)"

    legend = ax.get_legend()
    for label in legend.get_texts():
        label.set_text(_make_label(label.get_text()))

    return ax


####


def _common_dotplot(
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    color_key: str,
    size_key: str,
    ###
    color_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    color_vcenter: Optional[float] = None,
    size_vmin: Optional[float] = None,
    size_vmax: Optional[float] = None,
    size_vcenter: Optional[float] = None,
    ###
    # Configure which entries will be displayed in size legend:
    # n_legend_items_for_size: int = 5,
    extend_size_legend_to_vmin_vmax: bool = False,
    # Providing representative_sizes_for_legend will override n_legend_items_for_size and extend_size_legend_to_vmin_vmax
    representative_sizes_for_legend: Optional[List[float]] = None,
    # If should_size_legend_items_be_colored is False, all size legend items will be uniform black
    should_size_legend_items_be_colored: bool = False,
    ###
    figsize: Optional[Tuple[float, float]] = None,
    inverse_size: bool = False,
    min_marker_size: int = 1,
    marker_size_scale_factor: int = 100,
    grid: bool = True,
) -> Tuple[
    matplotlib.figure.Figure,
    matplotlib.axes.Axes,
    matplotlib.collections.PathCollection,
    List[matplotlib.collections.PathCollection],
    List[str],
]:
    # Inspiration from:
    # https://stackoverflow.com/a/59384782/130164
    # https://stackoverflow.com/a/65654470/130164
    # https://stackoverflow.com/a/63559754/130164

    fig = None
    try:
        if figsize is None:
            # autosize
            n_cols = data[x_axis_key].nunique()
            n_rows = data[y_axis_key].nunique()
            figsize = (n_cols * 1.5, n_rows / 2.5)

        def _generate_size_norm_input_data(
            size_values: Union[np.ndarray, pd.Series, List[float]],
        ) -> np.ndarray:
            # Clip the size values at size_vmin and size_vmax (which may be None) before scaling
            # (Similar clipping for color is done in the scatter call below)
            clipped_size_data = np.array(size_values)
            if size_vmin is not None or size_vmax is not None:
                clipped_size_data = np.clip(clipped_size_data, size_vmin, size_vmax)

            if size_vcenter is not None:
                # Support a vcenter for size, similar to color_vcenter, to allow divergent size palettes so negative values can have big sizes.
                # Example: size_vcenter=0 would make 0 have small size while 1 and -1 have large size.
                # Calculate the absolute difference from vcenter
                abs_diff_from_vcenter = np.abs(clipped_size_data - size_vcenter)
                clipped_size_data = abs_diff_from_vcenter

            # Note we have to reshape/ravel the data because this scaler expects a 2d array
            return clipped_size_data.reshape(-1, 1)

        # Fit the scaler to the clipped data
        # Due to clip=True, we do not need to repeat manual np.clip before calling size_scaller.transform.
        # Note we have to reshape/ravel the data because this scaler expects a 2d array
        size_scaler = MinMaxScaler(clip=True).fit(
            _generate_size_norm_input_data(data[size_key])
        )

        def size_norm(x: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
            """convert raw size data to plot marker size"""
            # Note we have to reshape/ravel the data because this scaler expects a 2d array
            transformed = size_scaler.transform(
                _generate_size_norm_input_data(x)
            ).ravel()
            if inverse_size:
                transformed = 1 - transformed

            return min_marker_size + marker_size_scale_factor * transformed

        fig, ax = plt.subplots(figsize=figsize)

        if color_vcenter is not None:
            # Create norm centered at color_vcenter.
            # We have to also handle color_vmin and color_vmax if they are not None.
            # They can't be passed alongside a norm to ax.scatter: "ValueError: Passing a Normalize instance simultaneously with vmin/vmax is not supported.  Please pass vmin/vmax directly to the norm when creating it."
            plot_vmin, plot_vmax = None, None

            # Previously we used: norm = matplotlib.colors.CenteredNorm(vcenter=color_vcenter)
            # But CenteredNorm does not accept vmin, vmax in its constructor, so we have to switch to TwoSlopeNorm
            color_norm = matplotlib.colors.TwoSlopeNorm(
                vcenter=color_vcenter, vmin=color_vmin, vmax=color_vmax
            )
        else:
            plot_vmin, plot_vmax = color_vmin, color_vmax
            color_norm = None

        scatter = ax.scatter(
            data[x_axis_key].values,
            data[y_axis_key].values,
            c=data[color_key].values,
            s=size_norm(data[size_key]),
            cmap=color_cmap,
            norm=color_norm,
            vmin=plot_vmin,
            vmax=plot_vmax,
            alpha=1,
            # Add outlines to the scatter points:
            edgecolors="lightgrey",
            linewidths=0.5,
        )
        ax.set_xlim(-0.5, max(ax.get_xticks()) + 0.5)
        ax.set_ylim(-0.5, max(ax.get_yticks()) + 0.5)
        ax.invert_yaxis()  # respect initial ordering - go top to bottom

        # Create grid
        if grid:
            ax.set_xticks(np.array(ax.get_xticks()) - 0.5, minor=True)
            ax.set_yticks(np.array(ax.get_yticks()) - 0.5, minor=True)
            ax.grid(which="minor")

        # Aspect ratio
        ax.set_aspect("equal", "box")

        # At this point, ax.get_xticklabels() may return empty tick labels and emit UserWarning: FixedFormatter should only be used together with FixedLocator
        # Must draw the canvas to position the ticks: https://stackoverflow.com/a/41124884/130164
        # And must assign tick locations prior to assigning tick labels, i.e. set_ticks(get_ticks()): https://stackoverflow.com/a/68794383/130164
        fig.canvas.draw()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation="vertical")

        ## Produce size legend entries containing a cross section of values.

        # User may have passed representative_sizes_for_legend.
        if representative_sizes_for_legend is not None:
            representative_sizes_for_legend = np.array(representative_sizes_for_legend)
        else:
            # Generate evenly spaced values

            # First, find the extents:
            data_min_size_clipped, data_max_size_clipped = (
                np.min(data[size_key]),
                np.max(data[size_key]),
            )
            if size_vmin is not None or size_vmax is not None:
                data_min_size_clipped = np.clip(
                    data_min_size_clipped, size_vmin, size_vmax
                )
                data_max_size_clipped = np.clip(
                    data_max_size_clipped, size_vmin, size_vmax
                )

            if extend_size_legend_to_vmin_vmax:
                # If extend_size_legend_to_vmin_vmax is True, override extents to vmin and vmax (unless they are None)
                if size_vmin is not None:
                    data_min_size_clipped = size_vmin
                if size_vmax is not None:
                    data_max_size_clipped = size_vmax

            # Generate values:
            if (
                size_vcenter is not None
                and data_min_size_clipped <= size_vcenter <= data_max_size_clipped
            ):
                # If size_vcenter is in the data range, add it to the representative sizes.
                # Therefore generate one fewer entry through auto generation.
                add_extra = np.array([size_vcenter])
            else:
                add_extra = np.array([])

            # n_legend_items_for_size = 5
            # representative_sizes_for_legend = np.linspace(data_min_size_clipped, data_max_size_clipped, n_legend_items_for_size)
            # Instead of linspace, use a Locator as in legend_elements(): https://github.com/matplotlib/matplotlib/blob/eb02b108ea181930ab37717c75e07ba792e01f1d/lib/matplotlib/collections.py#L1143-L1152
            # num = n_legend_items_for_size - len(add_extra)
            # locator = matplotlib.ticker.MaxNLocator(nbins=num-1, min_n_ticks=num, steps=[1, 2, 2.5, 3, 5, 6, 8, 10])
            locator = matplotlib.ticker.AutoLocator()
            representative_sizes_for_legend: np.ndarray = np.hstack(
                [
                    locator.tick_values(data_min_size_clipped, data_max_size_clipped),
                    add_extra,
                ]
            )
            representative_sizes_for_legend = np.clip(
                representative_sizes_for_legend,
                data_min_size_clipped,
                data_max_size_clipped,
            )
            representative_sizes_for_legend = np.unique(
                representative_sizes_for_legend
            )  # remove any duplicates
            representative_sizes_for_legend.sort()

        # Calculate corresponding plot sizes
        representative_sizes_after_norm = size_norm(representative_sizes_for_legend)

        # Calculate corresponding plot colors
        if should_size_legend_items_be_colored:
            representative_colors = representative_sizes_for_legend
            if color_vmin is not None or color_vmax is not None:
                # Clip color values at color vmin and color max, which may be None
                representative_colors = np.clip(
                    representative_colors, color_vmin, color_vmax
                )
            if color_norm is not None:
                # Apply a color norm
                representative_colors = [
                    color_norm(value) for value in representative_colors
                ]
            # Apply color cmap
            # First, cast string cmaps to a callable function
            color_cmap_func = matplotlib.cm.get_cmap(color_cmap)
            representative_colors = [
                color_cmap_func(value) for value in representative_colors
            ]
        else:
            representative_colors = ["grey"] * len(representative_sizes_for_legend)

        # Create legend handles and labels manually
        handles = [
            plt.scatter(
                [],
                [],
                s=size,
                color=color,
                alpha=1,
                # Add outlines to the scatter points:
                edgecolors="lightgrey",
                linewidths=0.5,
            )
            for size, color in zip(
                representative_sizes_after_norm, representative_colors
            )
        ]
        # Create legend labels with %g formatting to remove any trailing zeros when converting a float to string
        # labels = [f"{value:g}" for value in representative_sizes_for_legend]
        # Use this instead to set a maximum precision and avoid scientific notation:
        labels = [
            np.format_float_positional(val, trim="-", unique=True, precision=4)
            for val in representative_sizes_for_legend
        ]

        return fig, ax, scatter, handles, labels
    except Exception as err:
        # If there is an error, close the figure to prevent it from being displayed in a partial or broken state
        if fig is not None:
            plt.close(fig)
        # Reraise
        raise err


def plot_color_and_size_dotplot(
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    value_key: str,
    color_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    color_and_size_vmin: Optional[float] = None,
    color_and_size_vmax: Optional[float] = None,
    color_and_size_vcenter: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    legend_text: Optional[str] = None,
    # n_legend_items: int = 5,
    extend_legend_to_vmin_vmax: bool = False,
    representative_values_for_legend: Optional[List[float]] = None,
    min_marker_size: int = 1,
    marker_size_scale_factor: int = 100,
    grid: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot dotplot heatmap showing a key as both color and size.
    """
    fig = None
    try:
        fig, ax, scatter, handles, labels = _common_dotplot(
            data=data,
            x_axis_key=x_axis_key,
            y_axis_key=y_axis_key,
            color_key=value_key,
            size_key=value_key,
            color_cmap=color_cmap,
            color_vmin=color_and_size_vmin,
            color_vmax=color_and_size_vmax,
            color_vcenter=color_and_size_vcenter,
            size_vmin=color_and_size_vmin,
            size_vmax=color_and_size_vmax,
            size_vcenter=color_and_size_vcenter,
            # n_legend_items_for_size=n_legend_items,
            extend_size_legend_to_vmin_vmax=extend_legend_to_vmin_vmax,
            representative_sizes_for_legend=representative_values_for_legend,
            # Size legend will represent both size_key and color_key:
            should_size_legend_items_be_colored=True,
            figsize=figsize,
            inverse_size=False,
            min_marker_size=min_marker_size,
            marker_size_scale_factor=marker_size_scale_factor,
            grid=grid,
        )

        def add_padding_to_multiline_string(text: str, padding: int):
            """
            Adds spaces to the beginning of every line in a multiline string.

            :param text: The original multiline string.
            :param padding: The number of spaces to add at the beginning of each line.
            :return: A new multiline string with added padding.
            """
            padding_spaces = " " * padding
            lines = text.split("\n")
            padded_lines = [padding_spaces + line for line in lines]
            return "\n".join(padded_lines)

        size_legend = ax.legend(
            handles,
            labels,
            #
            # Left horizontal-align and center vertical-align the legend relative to this anchor point:
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            #
            title=add_padding_to_multiline_string(
                legend_text if legend_text is not None else value_key, padding=3
            ),
            borderaxespad=0.0,
            frameon=False,
            framealpha=0.0,
            title_fontproperties={"weight": "bold", "size": "medium"},
            numpoints=1,
            scatterpoints=1,
            markerscale=1.0,
            borderpad=1,
        )
        # align legend title left
        size_legend._legend_box.align = "left"

        return fig, ax
    except Exception as err:
        # If there is an error, close the figure to prevent it from being displayed in a partial or broken state
        if fig is not None:
            plt.close(fig)
        # Reraise
        raise err


def plot_two_key_color_and_size_dotplot(
    data: pd.DataFrame,
    x_axis_key: str,
    y_axis_key: str,
    color_key: str,
    size_key: str,
    color_cmap: Optional[Union[str, matplotlib.colors.Colormap]] = None,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
    color_vcenter: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    size_vmin: Optional[float] = None,
    size_vmax: Optional[float] = None,
    size_vcenter: Optional[float] = None,
    # n_legend_items_for_size: int = 5,
    extend_size_legend_to_vmin_vmax: bool = False,
    representative_sizes_for_legend: Optional[List[float]] = None,
    inverse_size: bool = False,
    color_legend_text: Optional[str] = None,
    size_legend_text: Optional[str] = None,
    shared_legend_title: Optional[str] = None,
    min_marker_size: int = 1,
    marker_size_scale_factor: int = 100,
    grid: bool = True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Plot dotplot heatmap showing two keys together.

    Example with mean and standard deviation: Circle color represents the mean. Circle size represents stability (inverse of standard deviation). Suggestions for this use case:

    - Pass mean key as `color_key` and standard deviation key as `size_key`.
    - Set `inverse_size=True`. Big circles are trustworthy/stable across the average, while little circles aren't
    - Set `color_legend_text="Mean", size_legend_text="Inverse std. dev."`
    - Set `min_marker_size=20` so that the smallest circle for zero standard deviation is still visible
    - With a diverging colormap (e.g. `color_cmap='RdBu_r', color_vcenter=0`) bold circles are strong effects, while near-white circles are weak effects
    """
    fig = None
    try:
        fig, ax, scatter, handles, labels = _common_dotplot(
            data=data,
            x_axis_key=x_axis_key,
            y_axis_key=y_axis_key,
            color_key=color_key,
            size_key=size_key,
            color_cmap=color_cmap,
            color_vmin=color_vmin,
            color_vmax=color_vmax,
            color_vcenter=color_vcenter,
            size_vmin=size_vmin,
            size_vmax=size_vmax,
            size_vcenter=size_vcenter,
            # n_legend_items_for_size=n_legend_items_for_size,
            extend_size_legend_to_vmin_vmax=extend_size_legend_to_vmin_vmax,
            representative_sizes_for_legend=representative_sizes_for_legend,
            # Size legend should be based entirely on size_key, not color_key:
            should_size_legend_items_be_colored=False,
            figsize=figsize,
            inverse_size=inverse_size,
            min_marker_size=min_marker_size,
            marker_size_scale_factor=marker_size_scale_factor,
            grid=grid,
        )

        # Make colorbar axes:
        # [xcorner, ycorner, width, height]: https://stackoverflow.com/a/65943132/130164
        # consider [1.1, 0.5, 0.3, 0.4] for bigger colorbar
        cbar_ax = ax.inset_axes([1.20, 0.52, 0.3, 0.2], transform=ax.transAxes)

        # Make colorbar
        fig.colorbar(scatter, cax=cbar_ax).set_label(
            color_legend_text if color_legend_text is not None else color_key,
            rotation=0,
            size="medium",
            weight="bold",
            horizontalalignment="left",
            verticalalignment="center",
        )
        if shared_legend_title is not None:
            cbar_ax.set_title(
                shared_legend_title, loc="left", pad=10, fontweight="bold"
            )

        # Produce a legend with a cross section of sizes from the scatter.
        size_legend = ax.legend(
            handles,
            labels,
            # consider this for lower down:
            # bbox_to_anchor=(1.05, 0.25),
            bbox_to_anchor=(1.05, 0.48),
            loc="upper left",
            title=size_legend_text if size_legend_text is not None else size_key,
            borderaxespad=0.0,
            frameon=False,
            framealpha=0.0,
            title_fontproperties={"weight": "bold", "size": "medium"},
            numpoints=1,
            scatterpoints=1,
            markerscale=1.0,
        )
        # align legend title left
        size_legend._legend_box.align = "left"

        return fig, ax
    except Exception as err:
        # If there is an error, close the figure to prevent it from being displayed in a partial or broken state
        if fig is not None:
            plt.close(fig)
        # Reraise
        raise err


def two_class_relative_density_plot(
    data: pd.DataFrame,
    x_key: str,
    y_key: str,
    hue_key: str,
    positive_class: str,
    colorbar_label: Optional[str] = None,
    quantile: Optional[float] = 0.50,
    figsize=(8, 8),
    n_bins=50,
    range=None,  # Extents within which to make bins
    continuous_cmap: str = "RdBu_r",
    cmap_vcenter: Optional[float] = 0.5,
    balanced_class_weights=True,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, str]:
    """
    Two-class relative density plot.
    For alternatives, see contour KDEs in seaborn's displot function.
    (For general 2D density plots, see plt.hexbin, sns.jointplot, and plt.hist2d.)
    """
    import scipy.stats

    def _weighted_mean(arr: np.ndarray, true_weight: float, false_weight: float):
        # how many total values in bin
        count = arr.shape[0]
        # how many positive class values in bin
        count_true = (arr.astype(int) == 1).sum()
        # how many negative class values in bin
        count_false = count - count_true

        numerator = count_true * true_weight
        return numerator / (numerator + count_false * false_weight)

    if balanced_class_weights:
        # Account for imbalance in positive and negative class sizes.
        # Members of rarer classes should count more towards density.
        # Instead of counting 1 towards density, each item should count 1/n_total_for_its_class.

        # Example: a bin with 2 positive and 2 negative examples.
        # But positive class has 1000 items overall and negative class has 10000 overall.
        # Unweighted mean: 2/(2+2) = 1/2.
        # Weighted: (2/1000) / [ (2/1000) + (2/10000) ] = 0.91.
        # The bin is significantly more positive than negative, relative to base rates.

        n_positive = (data[hue_key] == positive_class).sum()
        n_negative = data.shape[0] - n_positive

        def statistic(arr):
            return _weighted_mean(
                arr=arr, true_weight=1 / n_positive, false_weight=1 / n_negative
            )
    else:
        statistic = "mean"

    binned_data = scipy.stats.binned_statistic_2d(
        data[x_key],
        data[y_key],
        data[hue_key] == positive_class,
        statistic=statistic,
        bins=n_bins,
        expand_binnumbers=True,
        range=range,
    )

    # which bin does each point belong to
    bin_number_df = pd.DataFrame(binned_data.binnumber, index=["x_bin", "y_bin"]).T

    # filter out any beyond-edge bins that capture values outside bin bounds (e.g. due to range parameter)
    # we don't want to modify `binned_data.statistic` for these bins, because their indices will be out of bounds in that array
    # and we don't want to include these bins in the bin sizes quantile calculation, since these bins won't be displayed on the plot.
    # (note that the bin numbers are 1-indexed, not 0-indexed! https://github.com/scipy/scipy/issues/7010#issuecomment-279264653)
    # (so anything in bin 0 or bin #bins+1 is out of bounds)
    bin_number_df = bin_number_df[
        (bin_number_df["x_bin"] >= 1)
        & (bin_number_df["x_bin"] <= n_bins)
        & (bin_number_df["y_bin"] >= 1)
        & (bin_number_df["y_bin"] <= n_bins)
    ]

    # bin sizes: number of points per bin
    bin_sizes = bin_number_df.groupby(["x_bin", "y_bin"]).size()

    # Fill N/A counts for bins with 0 items
    bin_sizes = bin_sizes.reindex(
        pd.MultiIndex.from_product(
            [
                np.arange(1, binned_data.statistic.shape[0] + 1),
                np.arange(1, binned_data.statistic.shape[1] + 1),
            ],
            names=bin_sizes.index.names,
        ),
        fill_value=0,
    )

    # Prepare to plot
    # binned_data.statistic does not follow Cartesian convention
    # we need to transpose to visualize.
    # see notes in numpy.histogram2d docs.
    plot_values = binned_data.statistic.T

    # Choose bins to remove: drop bins with low number of counts
    # i.e. low overall density
    if quantile is not None:
        bins_to_remove = bin_sizes[
            bin_sizes <= bin_sizes.quantile(quantile)
        ].reset_index(name="size")

        # Remove low-count bins by setting the color value to nan.
        # To multiple-index into a 2d array, list x dimensions first, then y dimensions second.
        # Note: the bin numbers are 1-indexed, not 0-indexed! (https://github.com/scipy/scipy/issues/7010#issuecomment-279264653)
        # Also note the swap of y and x in the bin numbers, because of the transpose above.
        # (See numpy.histogram2d docs for more info.)
        plot_values[
            bins_to_remove["y_bin"].values - 1, bins_to_remove["x_bin"].values - 1
        ] = np.nan

        remaining_bins = bin_sizes[bin_sizes > bin_sizes.quantile(quantile)]
    else:
        remaining_bins = bin_sizes

    # Plot, as in numpy histogram2d docs
    fig, ax = plt.subplots(figsize=figsize)
    pltX, pltY = np.meshgrid(binned_data.x_edge, binned_data.y_edge)
    colormesh = plt.pcolormesh(
        pltX,
        pltY,
        plot_values,
        cmap=continuous_cmap,
        norm=matplotlib.colors.CenteredNorm(vcenter=cmap_vcenter)
        if cmap_vcenter is not None
        else None,
    )

    if colorbar_label is not None:
        # Add color bar.
        # see also https://stackoverflow.com/a/44642014/130164
        # Pull colorbar out of axis by creating a special axis for the colorbar - rather than distorting main ax.
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

        fig.colorbar(colormesh, cax=colorbar_ax, label=colorbar_label)

        # set global "current axes" back to main axes,
        # so that any calls like plt.title target main ax rather than inset colorbar_ax
        plt.sca(ax)

    plt.xlabel(x_key)
    plt.ylabel(y_key)

    description = remaining_bins.describe()
    description = (
        f"Counts range from {description['min']:n} to {description['max']:n} per bin"
    )

    return fig, ax, description


def plot_triangular_heatmap(
    df: pd.DataFrame,
    cmap="Blues",
    colorbar_label="Value",
    figsize=(8, 6),
    vmin=None,
    vmax=None,
    annot=True,
    fmt=".2g",
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot lower triangular heatmap.

    Often followed with:

    .. code-block:: python

        genetools.plots.wrap_tick_labels(
            ax, wrap_x_axis=True, wrap_y_axis=True, wrap_amount=10
        )
    """

    with sns.axes_style("white"):
        # Based on https://seaborn.pydata.org/examples/many_pairwise_correlations.html

        fig, ax = plt.subplots(figsize=figsize)

        # Upper triangle mask: the part that will be hidden
        triangle_mask = np.triu(np.ones_like(df, dtype=bool))

        sns.heatmap(
            df,
            cmap=cmap,
            # Draw the heatmap with the mask and correct aspect ratio
            mask=triangle_mask,
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar_kws={"shrink": 0.5, "label": colorbar_label},
            linewidths=0,
            ax=ax,
            # force all tick labels to be drawn
            xticklabels=True,
            yticklabels=True,
            annot=annot,
            fmt=fmt,
        )

        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
        )
        ax.set_yticklabels(
            ax.get_yticklabels(),
            rotation=0,
        )

        return fig, ax


def get_point_size(sample_size: int, maximum_size: float = 100) -> float:
    """get scatterplot point size based on sample size (from scanpy), but cut off at maximum_size"""
    # avoid division by zero - set sample_size to 1 if it's zero
    return min(120000 / max(1, sample_size), maximum_size)


def plot_confusion_matrix(
    df: pd.DataFrame,
    ax: Optional[matplotlib.axes.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    outside_borders=True,
    inside_border_width=0.5,
    wrap_labels_amount: Optional[int] = 15,
    wrap_x_axis_labels=True,
    wrap_y_axis_labels=True,
    draw_colorbar=False,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    with sns.axes_style("white"):
        if ax is None:
            if figsize is None:
                # Automatic sizing of confusion matrix, based on df's shape
                margin = 0.25
                size_per_class = 0.8
                # width: give a little extra breathing room because horizontal labels will fill the space
                auto_width = margin * 2 + df.shape[1] * size_per_class * 1.2
                if not draw_colorbar:
                    # remove some unnecessary width usually allocated to colorbar
                    auto_width -= df.shape[1] / 5
                # height: don't need extra breathing room because labels go left-to-right not up-to-down
                auto_height = margin * 2 + df.shape[0] * size_per_class
                figsize = (auto_width, auto_height)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            # Passed in an existing ax
            fig = ax.get_figure()

        # add text with numeric values (annot=True), but without scientific notation (overriding fmt with "g" or "d")
        sns.heatmap(
            df,
            annot=True,
            fmt="g",
            cmap="Blues",
            ax=ax,
            linewidth=inside_border_width,
            cbar=draw_colorbar,
            # plot all x and y tick labels
            xticklabels=True,
            yticklabels=True,
        )
        plt.setp(ax.get_yticklabels(), rotation="horizontal", va="center")
        plt.setp(ax.get_xticklabels(), rotation="horizontal", ha="center")

        if outside_borders:
            # Activate outside borders
            for _, spine in ax.spines.items():
                spine.set_visible(True)

        if wrap_labels_amount is not None:
            # Wrap long tick labels
            genetools.plots.wrap_tick_labels(
                ax,
                wrap_amount=wrap_labels_amount,
                wrap_x_axis=wrap_x_axis_labels,
                wrap_y_axis=wrap_y_axis_labels,
            )

        return fig, ax


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
