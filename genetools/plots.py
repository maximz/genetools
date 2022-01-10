import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from typing import Union, List, Dict
import textwrap

from .palette import HueValueStyle, convert_palette_list_to_dict


def savefig(fig, *args, **kwargs):
    """
    Save figure with tight bounding box.
    From https://github.com/mwaskom/seaborn/blob/master/seaborn/axisgrid.py#L33
    """
    kwargs = kwargs.copy()
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(*args, **kwargs)


def scatterplot(
    data,
    x_axis_key,
    y_axis_key,
    hue_key,
    ax=None,
    figsize=(8, 8),
    enable_legend=True,
    alpha=1.0,
    continuous_hue=False,
    continuous_cmap="viridis",
    discrete_palette: Union[
        Dict[str, Union[HueValueStyle, str]], List[Union[HueValueStyle, str]]
    ] = None,
    marker_size=15,
    legend_hues=None,
    na_color="lightgray",
    marker=".",
    marker_edge_color="none",
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
    **kwargs,
):
    """Simple umap scatter plot, with legend outside figure.

    Note, for discrete hues (continuous_hue=False):
    Figure size will grow beyond the figsize parameter setting, because the legend is pulled out of figure.
    So you must use fig.savefig('filename', bbox_inches='tight').
    This is provided automatically by genetools.plots.savefig(fig, 'filename')

    If using with scanpy, to get umap data from adata.obsm into adata.obs, try:
    > data = helpers.horizontal_concat(adata.obs, adata.obsm.to_df()[['X_umap1', 'X_umap2']])

    :param data: input data, e.g. anndata.obs
    :type data: pandas.DataFrame
    :param umap_1_key: column name with first dimension of UMAP
    :type umap_1_key: string
    :param umap_2_key: column name with second dimension of UMAP
    :type umap_2_key: string
    :param hue_key: column name with hue that will be used to color points
    :type hue_key: string
    :param continuous_hue: whether hue column takes continuous values and colorbar should be shown, defaults to False
    :type continuous_hue: bool, optional
    :param label_key: column name with optional cluster labels, defaults to None
    :type label_key: string, optional
    :param marker_size: marker size, defaults to 15
    :type marker_size: int, optional
    :param figsize: figure size, defaults to (8, 8)
    :type figsize: tuple, optional
    :param discrete_palette: color palette for discrete hues, defaults to None
    :type discrete_palette: matplotlib palette name, list of colors, or dict mapping hue values to colors, optional
    :param continuous_cmap: colormap for continuous hues, defaults to None
    :type continuous_cmap: matplotlib.colors.Colormap, optional
    :param label_z_order: z-index for cluster labels, defaults to 10
    :type label_z_order: int, optional
    :param label_color: color for cluster labels, defaults to 'k'
    :type label_color: str, optional
    :param label_alpha: opacity for cluster labels, defaults to 0.5
    :type label_alpha: float, optional
    :param label_size: size of cluster labels, defaults to 20
    :type label_size: int, optional
    :return: matplotlib figure and axes
    :rtype: (matplotlib.Figure, matplotlib.Axes)


    supply ax or figsize
    legend_hues: override and add any missing colors to legend. optional
    plotnonfinite: for continuous hues, plot points with inf or nan value.
    """

    if ax is None:
        # Make figure from scratch
        fig, ax = plt.subplots(figsize=figsize)
    else:
        # Passed in an existing ax
        fig = ax.get_figure()

    # store PathCollection returned by ax.scatter
    scattered_object = None

    if continuous_hue:
        # plot continuous variable with a colorbar
        scattered_object = ax.scatter(
            data[x_axis_key].values,
            data[y_axis_key].values,
            c=data[hue_key].values,
            cmap=continuous_cmap,
            s=marker_size,
            marker=marker,
            edgecolors=marker_edge_color,
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
                discrete_palette.get(hue_value, HueValueStyle(color=na_color))
            )
            # TODO: set defaults using passed in parameters, and store back in discrete_palette for use in legend too

            scattered_object = ax.scatter(
                hue_df[x_axis_key].values,
                hue_df[y_axis_key].values,
                color=marker_style.color,
                s=marker_size * marker_style.marker_size_scale_factor,
                marker=marker_style.marker
                if marker_style.marker is not None
                else marker,
                facecolors=marker_style.facecolors,
                edgecolors=marker_style.edgecolors
                if marker_style.edgecolors is not None
                else marker_edge_color,
                linewidths=marker_style.linewidths,
                zorder=marker_style.zorder,
                alpha=marker_style.alpha if marker_style.alpha is not None else alpha,
                plotnonfinite=plotnonfinite,
                **kwargs,
            )

    # Run tight_layout before adding legend,
    # especially before adding inset_axes colorbar (which wouldn't be included in tight_layout anyway, but may throw error on some matplotlib versions)
    # https://github.com/matplotlib/matplotlib/issues/21749
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
                colorbar_ax.set_xlabel(legend_title)

        else:
            # Create legend, and add any missing colors
            legend_handles = []
            for hue_value in legend_hues:
                # look up marker style for this hue value, if supplied in palette
                # if what's in palette is just a color, not a HueValueStyle, cast into HueValueStyle, i.e. apply default marker style with palette color
                # if this hue value is not a key in the palette dict at all, then fall back to na_color color and default marker style
                marker_style: HueValueStyle = HueValueStyle.from_color(
                    discrete_palette.get(hue_value, HueValueStyle(color=na_color))
                )

                # apply scaling to the default marker size we'd get by calling ax.scatter without any size arguments.
                legend_marker_size = (
                    mpl.rcParams["lines.markersize"] ** 2
                ) * marker_style.legend_size_scale_factor

                # Convert all to string so we can sort even if mixed input types like ["M", "F", np.nan]
                hue_value = str(hue_value)

                legend_handles.append(
                    ax.scatter(
                        [],
                        [],
                        label=hue_value,
                        color=marker_style.color,
                        marker=marker_style.marker,
                        s=legend_marker_size,
                        facecolors=marker_style.facecolors,
                        edgecolors=marker_style.edgecolors,
                        linewidths=marker_style.linewidths,
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
                # legend title font properties
                # TODO: requires newer matplotlib:
                # title_fontproperties={"weight": "bold", "size": "medium"},
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

    sns.despine(ax=ax)

    return fig, ax


# def _pull_legend_out_of_figure():
#     """Pull legend outside figure to the right.
#     Note: this expands figsize so you have to savefig with bbox_inches='tight'

#     See:
#         - https://stackoverflow.com/a/34579525/130164
#         - https://matplotlib.org/tutorials/intermediate/legend_guide.html#legend-location
#     """
#     plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)


def stacked_bar_plot(
    data,
    index_key,
    hue_key,
    value_key=None,
    ax=None,
    figsize=(8, 8),
    vertical=False,
    palette=None,
    normalize=True,
    axis_label="Frequency",
    hue_order=None,
    legend_title=None,
    na_color="lightgray",
    enable_legend=True,
):
    """Horizontal stacked bar chart.

    Note, figure size will grow beyond the figsize parameter setting, because the legend is pulled out of figure.
    So you must use fig.savefig('filename', bbox_inches='tight').
    This is provided automatically by genetools.plots.savefig(fig, 'filename')

    See https://observablehq.com/@d3/stacked-normalized-horizontal-bar for inspiration and colors.

    :param data: Plot data containing at minimum the columns identified by [index_key], [hue_key], and [value_key].
    :type data: pandas.DataFrame
    :param index_key: Column name defining the rows.
    :type index_key: str
    :param hue_key: Column name defining the horizontal bar categories.
    :type hue_key: str
    :param value_key: Column name defining the bar sizes.
    :type value_key: str
    :param palette: Color palette, defaults to None (in which case default palette used)
    :type palette: matplotlib palette name, list of colors, or dict mapping hue values to colors, optional
    :param figsize: figure size, defaults to (8, 8)
    :type figsize: tuple, optional
    :param normalize: Normalize each row's frequencies to sum to 1, defaults to True
    :type normalize: bool, optional
    :raises ValueError: Must specify correct number of colors if supplying a custom palette
    :return: matplotlib figure and axes
    :rtype: (matplotlib.Figure, matplotlib.Axes)

    must supply ax or figsize
    hue_order is applied from beginning (bottom or left) to the end (top or right) of the bar
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
                color=hue_style.color,
                hatch=hue_style.hatch,
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
