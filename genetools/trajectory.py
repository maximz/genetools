# run on all
import numpy as np
import pandas as pd
from genetools import plots, stats
import matplotlib.pyplot as plt
import seaborn as sns


def monte_carlo_pseudotime(adata_input, roots, id_vars):
    import scanpy as sc

    """[summary]

    :param adata_input: [description]
    :type adata_input: [type]
    :param roots: [description]
    :type roots: [type]
    :param id_vars: Identifier variables in anndata obs to uniquely identify a cell. Will be passed through to output.
    :type id_vars: [type]
    :return: [description]
    :rtype: [type]
    """
    # TODO: can we avoid passing in id_vars, and just set 'full_barcode' from obs_names

    adata = adata_input.copy()

    if "root_cell" in id_vars or "pseudotime" in id_vars:
        raise ValueError("Cannot include 'root_cell' or 'pseudotime' in id_vars")

    # root cell names
    col_names = []

    # each entry is one trajectory from one root cell: a Series, values are the pseudotime values per-cell, and name is the root cell
    pseudotime_vals = []

    for root_cell in roots:
        # Set root cell
        adata.uns["iroot"] = np.where(adata.obs.index == root_cell)[0][0]

        # Run DPT. Scanpy stores output in obs 'dpt_pseudotime
        sc.tl.dpt(adata, n_dcs=min(adata.obsm["X_diffmap"].shape[1], 10), copy=False)

        # Save this trajectory
        output = adata.obs["dpt_pseudotime"].copy()
        output.name = root_cell
        col_names.append(output.name)
        pseudotime_vals.append(output)

    # percentile normalize each pseudotime trajectory
    pseudotime_vals = pd.concat(pseudotime_vals, axis=1)
    pseudotime_vals_percentile_normalized = pseudotime_vals.apply(
        stats.percentile_normalize, axis=0
    )
    if pseudotime_vals_percentile_normalized.shape != pseudotime_vals.shape:
        raise ValueError()

    # Create dataframe where each row is a cell's value in a trajectory from a specific root cell.
    # i.e. convert to format: one value per cell, times # experiments
    df = pd.concat([adata_input.obs] + pseudotime_vals_percentile_normalized, axis=1)
    df = pd.melt(
        df.reset_index(),
        id_vars=id_vars,
        value_vars=col_names,
        var_name="root_cell",
        value_name="pseudotime",
    )

    if df.shape[0] != adata_input.obs.shape[0] * len(roots):
        raise ValueError(
            "Should have one value per cell, times the number of experiments"
        )

    return df


def choose_roots(adata, n_roots, cluster_key, cluster_names):
    """[summary]



    :param n_roots: [description]
    :type n_roots: [type]
    :param cluster_key: [description]
    :type cluster_key: [type]
    :param cluster_names: [description]
    :type cluster_names: [type]
    :return: [description]
    :rtype: [type]
    """
    return (
        adata.obs[adata.obs[cluster_key].isin(cluster_names)]
        .index.to_series()
        .sample(n_roots)
        .values
    )


def get_end_points(trajectory_df, id_vars):
    """[summary]

    :param trajectory_df: [description]
    :type trajectory_df: [type]
    :param id_vars: [description]
    :type id_vars: [type]
    :return: [description]
    :rtype: [type]
    """
    return trajectory_df.loc[
        trajectory_df.groupby("root_cell", sort=False)["pseudotime"].idxmax()
    ][id_vars]


def stochasticity(
    trajectory_df,
    xlabel="Pseudotime",
    ylabel="Number of unique barcodes in bin",
    title="How stochastic are different parts of the trajectory?",
):
    """[summary]

    :param trajectory_df: [description]
    :type trajectory_df: [type]
    :param xlabel: [description], defaults to "Pseudotime"
    :type xlabel: str, optional
    :param ylabel: [description], defaults to "Number of unique barcodes in bin"
    :type ylabel: str, optional
    :param title: [description], defaults to "How stochastic are different parts of the trajectory?"
    :type title: str, optional
    :return: [description]
    :rtype: [type]
    """
    df = trajectory_df.copy()

    # make bins
    pt_bins = np.histogram_bin_edges(
        df["pseudotime"], bins=50, range=(0, 1)
    )  # try bins='auto'

    # cut data into bins
    df["bin_id"] = pd.cut(df["pseudotime"], pt_bins)  # alternative: np.digitize

    # change bin ID to be left-offset of bin instead
    df["bin_id"] = df["bin_id"].apply(lambda interval: interval.left)

    # Histogram: number of values in each bin
    # for this analysis, we want this to be roughly uniform.
    # that's why we use percentile normalized pseudotime values above! otherwise it won't be uniform.
    # df["bin_id"].hist()

    # group by bin ID, count unique barcodes in that bin
    cells_by_bin = df.groupby("bin_id", sort=False)["full_barcode"].nunique()

    fig = plt.figure()
    plt.scatter(cells_by_bin.index, cells_by_bin.values)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    return fig


def mean_order(trajectory_df, barcode_key):
    """[summary]
    ## Ensemble: Mean trajectory

    trajectories have different linear scales:

    - so percentile normalize each trajectory first: replace start value by 0, end by 1, median by 0.5. now they have same scale, so averaging sensible (already done)
    - make sure nans→0s are being treated carefully, i.e. they are not being ranked ordinally. the pre-existing 0s (start cells) should be normalized to 0.
    - actually there are no nans above.

    In tests:
    # make sure concattable to obs
    assert all(mean_trajectory.index == adata.obs.index)

    # adata.obs["mean_trajectory"] = ensemble_pt_trajectories(
    #     adata, col_names, pseudotime_vals, roots
    # )

    :param trajectory_df: [description]
    :type trajectory_df: [type]
    :param barcode_key: [description]
    :type barcode_key: [type]
    :return: [description]
    :rtype: [type]
    """
    if any(trajectory_df["pseudotime"].isna()):
        raise ValueError("Some cells are unreachable (NaN pseudotimes)")

    # Take each cell's mean across all trajectories
    # "project cells onto this ensemble medoid trajectory"
    mean_trajectory = trajectory_df.groupby([barcode_key], sort=False)[
        "pseudotime"
    ].mean()
    if any(mean_trajectory.isna()):
        raise ValueError()

    # renormalize
    mean_trajectory = pd.Series(
        stats.percentile_normalize(mean_trajectory), index=mean_trajectory.index
    )

    return mean_trajectory


def spectral_order(
    trajectory_df,
    root_cell_key,
    barcode_key,
    n_trajectories_sample=None,
    n_cells_sample=None,
):
    """[summary]
    Expect percent normalized input.

    Use a kNN graph from scanpy instead of computing a dense similarity matrix:

    1. Make P, the cells-x-cells binary comparisons matrix of 1s and -1s based on cells beating each other in pseudotime trajectory, averaged over pseudotime trajectories.
    - Subsample cells at random.
    - Subsample matches at random.
    - At a point the basic order will pop out, no need to be perfect.
    2. Normalize each row to have same Euclidean norm
    3. Create Anndata from P. (View P as the usual cells x features matrix, where each cell is featurized in terms of the others.)
    4. Run scanpy `sc.pp.neighbors`. Now you have a kNN graph, like a compressed version of the full similarity graph above.
    5. Laplacian on `adata.uns['neighbors']['connectivities']`, the adjacency matrix of the kNN graph.
    6. 2nd smallest eigenvector

    Previously, we were creating dense similarity matrix, where the similarity measure between two cells `i, j` is the sum over all other cells `k` of $1 - abs(P_{i,j} - P_{j,k})$. The intuition is that this gives you the ranking similarity between any two cells by comparing their probabilities of being ranked higher than other reference cells. Now are are simplifying how to create this similarity matrix.


    :param trajectory_df: [description]
    :type trajectory_df: [type]
    :param root_cell_key: [description]
    :type root_cell_key: [type]
    :param barcode_key: [description]
    :type barcode_key: [type]
    :param n_trajectories_sample: [description], defaults to None
    :type n_trajectories_sample: [type], optional
    :param n_cells_sample: [description], defaults to None
    :type n_cells_sample: [type], optional
    :raises ValueError: [description]
    :return: [description]
    :rtype: [type]
    """

    # TODO: confirm percentile normalized input

    import scipy.sparse
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh
    from sklearn import preprocessing
    import anndata
    import scanpy as sc

    # Legacy code from when we passed around arrays of individual trajectories
    # (We might bring that API back):

    # pseudotime_ranks = pd.concat(pseudotime_vals, axis=1)
    # pseudotime_ranks = pseudotime_ranks.apply(percentile_normalize, axis=0)
    # pseudotime_ranks = pseudotime_ranks.T  # n_experiments x n_cells, percentile normalized

    # print(pseudotime_ranks.shape)
    # pseudotime_ranks

    # cells_sample = np.random.choice(
    #     pseudotime_ranks.columns, size=n_cells_sample, replace=False
    # )
    # cells_sample[:5]

    # matches_sample = np.random.choice(
    #     pseudotime_ranks.index, size=n_trajectories_sample, replace=False
    # )
    # matches_sample[:5]

    # pseudotime_ranks_subsample = pseudotime_ranks.loc[matches_sample][cells_sample]
    # pseudotime_ranks_subsample.shape, pseudotime_ranks.shape

    if not n_cells_sample:
        # default to 20% of cells
        n_cells_sample = int(trajectory_df[barcode_key].nunique() * 0.2)
    if not n_trajectories_sample:
        # default to 10% of trajectories
        n_trajectories_sample = int(trajectory_df[root_cell_key].nunique() * 0.1)

    cells_sample = np.random.choice(
        trajectory_df[barcode_key].unique(), size=n_cells_sample, replace=False
    )
    matches_sample = np.random.choice(
        trajectory_df[root_cell_key].unique(), size=n_trajectories_sample, replace=False
    )
    # n_experiments x n_cells, percentile normalized
    pseudotime_ranks_subsample = trajectory_df[
        (trajectory_df[root_cell_key].isin(matches_sample))
        & (trajectory_df[barcode_key].isin(cells_sample))
    ].pivot(index=root_cell_key, columns=barcode_key, values="pseudotime")

    # make P, the transition matrix
    # # number of matches, as in a tournament where each match = all players compete against each other
    # n_matches = pseudotime_ranks_subsample.shape[0]
    # # number of players in the tournament
    # n_cells = pseudotime_ranks_subsample.shape[1]

    matches = scipy.sparse.csr_matrix(
        (pseudotime_ranks_subsample.shape[1], pseudotime_ranks_subsample.shape[1])
    )

    # create lower triangular ones
    template_mat = scipy.sparse.tril(np.ones(matches.shape), format="csr")

    # for every experiment, get C comparisons matrix
    for _, row in pseudotime_ranks_subsample.iterrows():

        # reorder template matrix to match real order
        # basically, we want to move column #index to be column #argsort
        # and same for rows
        # where #index is an index into the lower-triangular matrix created above
        # and #argsort is the argsort of the real order

        # e.g. suppose the values are [4,1,2,3]
        # then the argsort is [2,3,4,1]
        # the index (as always) into the lower-triangular matrix is [1,2,3,4] to start
        # (in actuality the index and argsort would be zero-indexed. not the raw values)
        # so you would move column 1 to be column 2, column 2 to be column 3,
        # column 3 to be column 4, and column 4 to be column 1; and same for rows.
        # i.e. column #index --> column #argsort

        # we achieve this by creating a series from the argsort, so we will have
        # 1   2
        # 2   3
        # 3   4
        # 4   1

        # then we sort the values
        # that rearranges the index into 4,1,2,3
        # which is the order we put rows and columns of the lower-triangular matrix into.

        reorder_template = pd.Series(np.argsort(row.values)).sort_values().index.values
        # note: this is a copy not a view unfortunately.
        matches += template_mat[reorder_template, :][:, reorder_template]

    # maybe we can cut the below and just average?

    # # change 0s to -1s after the fact
    # # x - y = 2x - (x+y) , where x is number of 1s, y is number of -1s
    # # we have only been summing x, not x-y, so far
    # matches *= 2
    # # Have to convert to dense here
    # # subtracting a nonzero scalar from a sparse matrix is not supported
    # matches = matches.toarray() - n_matches

    # # compute sample probabilities
    # #     Q = np.sum(matches, axis=0)
    # Q = matches  # already summed
    # Q += 1 * n_matches
    # Q *= 0.5
    # Q *= 1.0 / n_matches
    Q = matches

    # normalize each row to have same euclidean norm

    Q_normalized = preprocessing.normalize(Q, norm="l2")

    Q_adata = anndata.AnnData(Q_normalized)

    # import scanpy as sc
    sc.pp.neighbors(Q_adata, use_rep="X")

    # laplacian of the adjacency matrix of the kNN graph
    laplacian_mat = laplacian(Q_adata.uns["neighbors"]["connectivities"], normed=False)

    _, eigenvecs = eigsh(laplacian_mat, 2, which="SM", tol=1e-8)

    # take second smallest eigenvalue
    # order along it
    labels = pseudotime_ranks_subsample.columns
    order = [label for (_, label) in sorted(zip(eigenvecs[:, 1], labels))]
    # assign orders 0 -> 1, normalize / num_cells
    spectral_order_series = pd.Series(
        np.arange(len(order)) / len(order), index=order, name="spectral_order"
    )

    # want no repeats in the order, i.e. all value counts should be 1
    if not all(spectral_order_series.reset_index()["index"].value_counts() == 1):
        raise ValueError("Unexpected repeats in the spectral ordering")

    # sometimes eigenvector signs will be flipped, giving the reverse ordering.
    # in these cases we should just reverse the ordering.

    # to determine whether this is the case:
    # choose a cell we know to be closer to the start (or end), and ensure it is closer to the start (or end).
    # for instance, run the mean ensembling, check whether the start or end point is ahead in the spectral ordering.
    # flip order as needed to match.

    # another way to do this:
    # look at mean spectral order of the root cells of all trajectories
    # flip order if needed.
    # (note: we subsampled the cells before running spectral ordering.
    # so instead of using 0-position, use min-position cells from the trajectories.)
    filtered_trajectories = trajectory_df[
        (trajectory_df[root_cell_key].isin(matches_sample))
        & (trajectory_df[barcode_key].isin(cells_sample))
    ]
    if (
        spectral_order_series[
            filtered_trajectories.loc[
                filtered_trajectories.groupby("root_cell_key")["pseudotime"].idxmin()
            ][barcode_key]
        ].median()
        > 0.5
    ):
        spectral_order_series = 1 - spectral_order_series

    return spectral_order_series


def compare_trajectories(
    trajectory_1,
    trajectory_2,
    trajectory_1_label="Trajectory 1",
    trajectory_2_label="Trajectory 2",
    title=None,
):
    """[summary]

    # per-cell difference in ordering, hist

    # adata.obs["mean_trajectory"], adata.obs["spectral_order"]
    # xlabel="Mean over percentile normalized trajectories"
    # ylabel="Spectral order"
    # title="Spectral vs mean pseudotime ensemble"

    :param trajectory_1: [description]
    :type trajectory_1: [type]
    :param trajectory_2: [description]
    :type trajectory_2: [type]
    :param trajectory_1_label: [description], defaults to 'Trajectory 1'
    :type trajectory_1_label: str, optional
    :param trajectory_2_label: [description], defaults to 'Trajectory 2'
    :type trajectory_2_label: str, optional
    :param title: [description], defaults to None
    :type title: [type], optional
    :return: [description]
    :rtype: [type]
    """

    # merge indexes
    compare_aggregates = pd.merge(
        trajectory_1, trajectory_2, left_index=True, right_index=True, how="inner"
    )
    trajectory_1, trajectory_2 = (
        compare_aggregates[trajectory_1.name],
        compare_aggregates[trajectory_2.name],
    )
    # print(compare_aggregates.shape)

    compare_aggregates["diff"] = trajectory_1 - trajectory_2
    compare_aggregates["abs_diff"] = compare_aggregates["diff"].abs()
    # compare_aggregates.head()

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(trajectory_1, trajectory_2, s=1, zorder=10)

    # plot y=x
    plt.plot([0, 1], [0, 1], "k", alpha=0.75, zorder=0)

    ax.set_aspect("equal")

    plt.xlabel(trajectory_1_label)
    plt.ylabel(trajectory_2_label)
    if title is not None:
        plt.title(title)
    sns.despine()

    return fig, compare_aggregates["abs_diff"].median()


def plot_pseudotime_density(
    adata, cluster_label_key, value_key, figsize=(6, 8.5), **kwargs
):
    """[summary]

    # fig = plot_pseudotime_density(adata, suptitle="Pseudotime Coembed (mean trajectory)", cluster_label_key="cluster_label",value_key="mean_trajectory")

    :param adata: [description]
    :type adata: [type]
    :param cluster_label_key: [description]
    :type cluster_label_key: [type]
    :param value_key: [description]
    :type value_key: [type]
    :param figsize: [description], defaults to (6, 8.5)
    :type figsize: tuple, optional
    :return: [description]
    :rtype: [type]
    """
    ## First, remove without one-cell clusters because we can't draw a density for those

    clusters_to_keep = adata.obs[cluster_label_key].value_counts()
    clusters_to_keep = clusters_to_keep[clusters_to_keep > 1].index.tolist()
    adata_plt = adata[adata.obs[cluster_label_key].isin(clusters_to_keep)].copy()
    # print("Removed 1-cell clusters")
    # print(adata.shape, "-->", adata_plt.shape)

    ## Set a new ylevel that only contains the remaining clusters

    # problem with using cluster_label_key as ylevel:
    # it's possibly a Categorical so it still contains the empty levels, so now those will have empty rows attached
    # just convert back to strings from categorical
    adata_plt.obs[cluster_label_key] = adata_plt.obs[cluster_label_key].astype("str")

    ## Set row order
    # sort rows by median value
    ylevel_order = (
        adata_plt.obs.groupby(cluster_label_key)[value_key]
        .median()
        .sort_values()
        .index.tolist()
    )

    return plots.stacked_density_plot(
        adata_plt.obs,
        row_var=cluster_label_key,
        hue_var=cluster_label_key,
        value_var=value_key,
        figsize=figsize,
        row_order=ylevel_order,
        xlabel="Pseudotime",
        **kwargs
    )
