import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def savefig(fig, *args, **kwargs):
    """
    Save figure with tight bounding box.
    From https://github.com/mwaskom/seaborn/blob/master/seaborn/axisgrid.py#L33
    """
    kwargs = kwargs.copy()
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(*args, **kwargs)


def umap_scatter(
    data,
    umap_1_key,
    umap_2_key,
    hue_key,
    continuous_hue=False,
    label_key=None,
    marker_size=15,
    figsize=(8, 8),
    discrete_palette=None,
    continuous_cmap="viridis",
    label_z_order=10,
    label_color="k",
    label_alpha=0.5,
    label_size=20,
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
    """
    fig, ax = plt.subplots(figsize=figsize)
    with sns.axes_style("white"):
        if continuous_hue:
            # plot continuous variable with a colorbar
            g = ax.scatter(
                data[umap_1_key].values,
                data[umap_2_key].values,
                c=data[hue_key].values,
                cmap=continuous_cmap,
                s=marker_size,
            )

            # color bar
            # see also https://stackoverflow.com/a/44642014/130164
            fig.colorbar(g)
        else:
            # plot discrete hues

            # create colors
            n_colors = data[hue_key].nunique()
            if not discrete_palette:
                discrete_palette = sns.color_palette("Spectral", n_colors=n_colors)

            if len(discrete_palette) < n_colors:
                raise ValueError("Not enough colors in palette")

            # subset to exact number of colors we need (otherwise seaborn throws error)
            discrete_palette = discrete_palette[:n_colors]

            # plot
            g = sns.scatterplot(
                data=data,
                x=umap_1_key,
                y=umap_2_key,
                hue=hue_key,
                palette=discrete_palette,
                ax=ax,
                legend="full",
                alpha=1,
                s=marker_size,
            )

        # equal aspect ratio
        ax.set_aspect("equal", "datalim")

        # add cluster labels
        if label_key is not None:
            for label, grp in data.groupby(label_key):
                plt.annotate(
                    "%s" % label,
                    grp[[umap_1_key, umap_2_key]].mean(),  # mean of x and y
                    horizontalalignment="center",
                    verticalalignment="center",
                    size=label_size,
                    weight="bold",
                    alpha=label_alpha,
                    color=label_color,
                    zorder=label_z_order,
                )

        sns.despine(ax=ax)

        # pull legend outside figure to the right
        # https://stackoverflow.com/a/34579525/130164
        # https://matplotlib.org/tutorials/intermediate/legend_guide.html#legend-location
        # note: this expands figsize so you have to savefig with bbox_inches='tight'
        if not continuous_hue:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

        return fig, ax


def horizontal_stacked_bar_plot(
    data, index_key, hue_key, value_key, palette=None, figsize=(8, 8), normalize=True
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
    """

    plot_df = data[[index_key, value_key, hue_key]].copy()

    # create colors
    n_colors = plot_df[hue_key].nunique()
    if not palette:
        palette = sns.color_palette("muted", n_colors=n_colors)

    if len(palette) < n_colors:
        raise ValueError("Not enough colors in palette")

    if normalize:
        # Normalize values to sum to 1 per row
        plot_df[value_key] = plot_df.groupby(index_key)[value_key].apply(
            lambda g: g / g.sum()
        )

    # Sort so we maintain consistent order before we calculate cumulative value
    plot_df = plot_df.sort_values([index_key, hue_key])

    # Accumulate value with every subsequent box/hue as we go across each index/row
    # These will become row-level "left offsets" for each hue
    cum_value_key = value_key + "_cumulative_value"
    plot_df[cum_value_key] = plot_df.groupby(index_key)[value_key].cumsum()

    with sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=figsize)

        # Go hue-by-hue, and plot down the rows
        for (color, (hue_name, hue_data)) in zip(palette, plot_df.groupby(hue_key)):
            ax.barh(
                hue_data[index_key].values,
                hue_data[value_key].values,
                align="center",
                height=0.25,
                left=(hue_data[cum_value_key] - hue_data[value_key]).values,
                label=hue_name,
                color=color,
            )

        # pull legend outside figure
        # https://stackoverflow.com/a/34579525/130164
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title=hue_key)

        plt.xlabel("Frequency")

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
