import scipy
import numpy as np
import pandas as pd
from functools import wraps
import sklearn.utils.extmath
import sklearn.metrics
import scipy.special
from typing import Optional, List, Union
import dataclasses
from scipy.interpolate import interp1d

from genetools import helpers


def accept_series(func):
    """Decorator to seamlessly accept pandas Series in place of a numpy array, and returns with original Series index."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], pd.Series):
            return pd.Series(
                func(args[0].values, *args[1:], **kwargs), index=args[0].index
            )
        return func(*args, **kwargs)

    return wrapper


@accept_series
def percentile_normalize(values):
    """Percentile normalize.

    :param values: values to normalize
    :type values: numpy.ndarray or pandas.Series
    :return: percentile-normalized values
    :rtype: numpy.ndarray or pandas.Series
    """
    return scipy.stats.rankdata(values, "average") / len(values)


@accept_series
def rank_normalize(values):
    """Rank normalize, starting with rank 1. All ranks must be unique.

    :param values: values to normalize
    :type values: numpy.ndarray or pandas.Series
    :return: rank-normalized values
    :rtype: numpy.ndarray or pandas.Series
    """
    return scipy.stats.rankdata(values, method="ordinal")


def normalize_rows(df):
    """Make rows sum to 1.
    If a row is all zeroes, this will return NaNs for the row entries, since there is no way to make the row sum to 1.

    :param df: dataframe
    :type df: pandas.DataFrame
    :return: row-normalized dataframe
    :rtype: pandas.DataFrame
    """
    return df.div(df.sum(axis=1), axis=0)


def normalize_columns(df):
    """Make columns sum to 1.
    If a column is all zeroes, this will return NaNs for the column entries, since there is no way to make the column sum to 1.

    :param df: dataframe
    :type df: pandas.DataFrame
    :return: column-normalized dataframe
    :rtype: pandas.DataFrame
    """
    return df.div(df.sum(axis=0), axis=1)


def coclustering(cluster_ids_1, cluster_ids_2):
    """Compute coclustering percentage between two sets of cluster IDs for the same cells:
    Of the cell pairs clustered together by either or both methods, what percentage are clustered together by both methods?

    (The clusters are allowed to have different names across methods, and don't necessarily need to be ints.)

    :param cluster_ids_1: One set of cluster IDs.
    :type cluster_ids_1: numpy array-like
    :param cluster_ids_2: Another set of cluster IDs.
    :type cluster_ids_2: numpy array-like
    :return: Percentage of cell pairs clustered together by one or both methods that are also clustered together by the other method.
    :rtype: float
    """

    """
    Approach here comes from exploting some Rand index implementations
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    https://en.wikipedia.org/wiki/Rand_index
    https://stackoverflow.com/a/49586743/130164
    https://stats.stackexchange.com/a/157385/297

    We want (a) / (a + c + d) from the Rand index

    a: the number of pairs of elements in S that are in the same subset in X and in the same subset in Y
    c: the number of pairs of elements in S that are in the same subset in X and in different subsets in Y
    d: the number of pairs of elements in S that are in different subsets in X and in the same subset in Y

    in other words:
    a+c+d is the total number of elements coclustered in one, the other, or both datasets
    a is the number of elements coclustered in both datasets.
    """
    from scipy.special import comb

    # map to ints
    method1 = pd.factorize(cluster_ids_1)[0]
    method2 = pd.factorize(cluster_ids_2)[0]

    tp_plus_fp = comb(np.bincount(method1), 2).sum()
    tp_plus_fn = comb(np.bincount(method2), 2).sum()
    A = np.c_[(method1, method2)]
    # bincount: Count number of occurrences of each value in array of non-negative ints.
    # So basically, for those cells, this is the number of them that land in each method2 cluster
    # then we choose pairs.. and sum..
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum() for i in set(method1))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp

    return tp / (tp + fp + fn)


def _coclustering_slow(cluster_ids_1, cluster_ids_2):
    """Slow version of coclustering()"""

    def coclustering_same_dataset(cluster_ids):
        """For a given set of cluster IDs, compare them pairwise to find which cells are co-clustered (have same cluster ID)

        :param cluster_ids: One set of cluster IDs.
        :type cluster_ids: numpy array-like
        :return: n_cells x n_cells matrix, with binary values corresponding to whether the pair of cells had same cluster ID
        :rtype: numpy.ndarray
        """
        # see https://stackoverflow.com/a/46266707/130164 for pairwise comparisons
        # defensively cast to numpy array
        # [:, None] adds a new axis so the result is 2d
        dataset_coclustered = np.array(cluster_ids) == np.array(cluster_ids)[:, None]
        assert (
            dataset_coclustered.shape[0]
            == dataset_coclustered.shape[1]
            == len(cluster_ids)
        )
        return dataset_coclustered

    # get n_cells by n_cells binary matrix of coclusterings in each set of cluster IDs individually
    assert len(cluster_ids_1) == len(
        cluster_ids_2
    ), "The two sets of cluster IDs must be the same length"
    dataset1_coclustered = coclustering_same_dataset(cluster_ids_1)
    dataset2_coclustered = coclustering_same_dataset(cluster_ids_2)
    assert (
        dataset1_coclustered.shape
        == dataset2_coclustered.shape
        == (len(cluster_ids_1), len(cluster_ids_1))
    )

    # now compare if cells coclustered in dataset1 are also coclustered in dataset2
    def get_coclustering_indices(cells_by_cells):
        """From each dataset, extract off-diagonal indices that have Trues.
        These are the cell-cell coclustering relationships within each dataset.
        Reshape as a numpy array:
        - each row is one coclustering relationship
        - Columns are the cell IDs of the two cells involved.

        :param cells_by_cells: n_cells by n_cells binary matrix where True indicates that a pair of cells has same cluster label in one dataset.
        :type cells_by_cells: numpy.ndarray
        :return: array of shape (# coclustering relationships in this dataset, 2)
        :rtype: numpy.ndarray
        """
        # extract non-diagonal and true
        out = np.array(
            np.where(
                # Ignore Ruff E712 that asks us to change to "cells_by_cells is True"
                ~np.eye(cells_by_cells.shape[0], dtype=bool) & (cells_by_cells == True)  # noqa: E712
            )
        ).T
        assert out.shape[1] == 2, "each row should have two cell IDs"
        return out

    def convert_rows_to_tuple_set(arr):
        """Convert each row into a (cell ID, cell ID) tuple, and make a set out of all of them.

        :param arr: array of shape (# coclustering relationships in this dataset, 2)
        :type arr: numpy.ndarray
        :return: set of (cell ID, cell ID) tuples.
        :rtype: set
        """
        out = set(map(tuple, arr))
        assert [len(t) == 2 for t in out], "each tuple should have 2 entries"
        return out

    # run this on each dataset
    indices_1 = convert_rows_to_tuple_set(
        get_coclustering_indices(dataset1_coclustered)
    )
    indices_2 = convert_rows_to_tuple_set(
        get_coclustering_indices(dataset2_coclustered)
    )
    # now get percentage of unique coclusterings across both datasets that are shared between them
    # note: both the numerator and the denominator are double-counting each relationship, i.e. this is accounted for
    # e.g. if cells 1 and 2 are coclustered in dataset 1, that will create tuples (1, 2) and (2, 1)
    return len(indices_1.intersection(indices_2)) / len(indices_1.union(indices_2))


def intersect_marker_genes(
    reference_data,
    query_data,
    low_confidence_threshold=0.035,
    low_confidence_suffix="?",
):
    """Map cluster marker genes against reference lists to find top hits.

    query_data and reference_data should both be dictionaries where:
        - keys are cluster names or IDs
        - values are lists of genes associated with that cluster

    Or if you have a dataframe where each row contains a cluster ID and a gene name, you can convert to dict with ``df.groupby('cluster')['gene'].apply(list).to_dict()``

    Usage with an anndata/scanpy object on groups defined by ``adata.obs['louvain']``:

    .. code-block:: python

        # find marker genes for all clusters
        cluster_markers_df = genetools.scanpy_helpers.find_all_markers(adata, cluster_key='louvain')

        # convert to dict of clusters -> gene lists mapping
        cluster_marker_lists = cluster_markers_df.groupby('louvain')['gene'].apply(list).to_dict()

        # intersect with known marker gene lists
        results, label_map, low_confidence_percentage = genetools.stats.intersect_marker_genes(reference_marker_lists, cluster_marker_lists)

        # rename clusters in your anndata/scanpy object
        adata.obs['louvain_annotated'] = adata.obs['louvain'].copy().cat.rename_categories(label_map)

    Behavior:
        - Intersection scores are normalized to marker gene list sizes.
        - Resulting duplicate cluster names are renamed, ensuring that N original query clusters will map to N renamed clusters.

    :param reference_data: reference marker gene lists
    :type reference_data: dict
    :param query_data: query marker gene lists
    :type query_data: dict
    :param low_confidence_threshold: Minimal difference between top and subsequent hits for a confident call, defaults to 0.035
    :type low_confidence_threshold: float, optional
    :param low_confidence_suffix: Suffix for low-confidence cluster renamings, defaults to "?"
    :type low_confidence_suffix: str, optional
    :return: dataframe with cluster mapping details, a dictionary for renaming query cluster names, and percentage of low-confidence calls.
    :rtype: (pandas.DataFrame, dict, float) tuple
    """

    assert (
        len(reference_data.keys()) >= 2
    ), "Need at least two reference clusters otherwise everything will map to the first."

    # make dataframe with reference clusters as rows, query clusters as columns, cells are intersection scores
    marker_gene_intersection_df = pd.DataFrame()
    for query_cluster_id, query_cluster_genes in query_data.items():
        intersection_vals = {}
        for reference_cluster_id, reference_cluster_genes in reference_data.items():
            # get list intersection
            query_set = set(query_cluster_genes)
            reference_set = set(reference_cluster_genes)
            num_intersecting = len(query_set.intersection(reference_set))
            # normalize to shortest marker gene list
            # so score of 1 = all genes intersected
            min_set_size = min(len(query_set), len(reference_set))
            intersection_vals[reference_cluster_id] = num_intersecting / float(
                min_set_size
            )
        marker_gene_intersection_df[query_cluster_id] = pd.Series(intersection_vals)

    # show intersection heatmap
    # sns.heatmap(marker_gene_intersection_df)

    # choose top match and report how different it is from second and third match
    marker_gene_label_results = {}
    for query_cluster_id in marker_gene_intersection_df.columns:
        intersection_values = (
            marker_gene_intersection_df[query_cluster_id]
            .sort_values(ascending=False)
            .head(n=2)
        )
        marker_gene_label_results[query_cluster_id] = {
            "top_match": intersection_values.index[0],
            "next_match": intersection_values.index[1],
            "highest_score": intersection_values[0],
            "diff_from_next_match": (intersection_values[0] - intersection_values[1]),
        }

    # rearrange for readability
    marker_gene_results_df = pd.DataFrame(marker_gene_label_results).transpose()
    marker_gene_results_df.index.name = "query_cluster_id"

    # heatmap score differences between 1st and 3rd top match to define low confidence calls
    #     marker_gene_results_df["diff_from_next_match"].hist()

    # determine which mappings are low confidence
    marker_gene_results_df["low_confidence_call"] = (
        marker_gene_results_df["diff_from_next_match"] <= low_confidence_threshold
    )

    # add low_confidence_suffix to end of cluster labels for low-confidence calls
    marker_gene_results_df["top_match"] = marker_gene_results_df[
        "top_match"
    ] + marker_gene_results_df["low_confidence_call"].replace(
        {False: "", True: low_confidence_suffix}
    )

    # rename duplicates so that N unique clusters are mapped to N unique clusters
    # (i.e. don't want to collapse to the number of clusters in the reference)
    marker_gene_results_df["new_cluster"] = helpers.rename_duplicates(
        marker_gene_results_df["top_match"]
    )
    label_map = marker_gene_results_df["new_cluster"].to_dict()

    return (
        marker_gene_results_df[
            ["new_cluster", "low_confidence_call", "highest_score", "next_match"]
        ],
        label_map,
        (
            marker_gene_results_df["low_confidence_call"]
            .value_counts(normalize=True)
            .get(True, default=0.0)
        ),
    )


def _seurat_clr(x):
    """Seurat's implementation of centered log-ratio transformation."""
    # TODO: support sparseness
    s = np.sum(np.log1p(x[x > 0]))
    exp = np.exp(s / len(x))
    return np.log1p(x / exp)


def clr_normalize(mat, axis=0):
    """Centered log ratio transformation for Cite-seq data, normalizing:

    * each protein's count vectors across cells (axis=0, normalizing each column of the cells x proteins matrix, default)
    * or the antibody count vector for each cell (axis=1, normalizing each row of the cells x proteins matrix)

    To use with anndata: ``genetools.scanpy_helpers.clr_normalize(adata, axis)``

    Notes:

    * Output will be densified.

    * We use Seurat's [modified CLR implementation](https://github.com/satijalab/seurat/issues/2624) to handle pseudocounts: ``log1p(x = x / (exp(x = sum(log1p(x = x[x > 0]), na.rm = TRUE) / length(x = x))))``.

     This is almost the same as ``log(x) - 1/D * sum( log(product of x's) )``,
     which is the same as ``log(x) - log ( [ product of x's] ^ (1/D) )``,
     where ``D = len(x)``.

     The general definition is:

     .. code-block:: python

         from scipy.stats.mstats import gmean
         return np.log(x) - np.log(gmean(x))

     But geometric mean only applies to positive numbers (otherwise the inner product will be 0).
     So you want to use pseudocounts or drop 0 counts.
     That's what Seurat's modification does.

    * See also https://github.com/theislab/scanpy/pull/1117 for other approaches.

    * Do you run this normalization cell-wise or gene-wise (i.e. protein-wise)? `See discussion here <https://github.com/satijalab/seurat/issues/871#issuecomment-431414099>`_:

     .. code-block:: text

         Unfortunately there is not a single answer. In some cases, cell-based normalization fails. This is because cell-normalization makes an assumption that the total ADT counts should be constant across cells. That can become a significant issue if you have cell populations in your data, but did not add protein markers for them (this is also an issue for scRNA-seq, but is significantly mitigated because at least you measure many genes).

         However, gene-based normalization can fail when there is significant heterogeneity in sequencing depth, or cell size. The optimal strategy depends on the AB panel, and heterogeneity of your sample.

     In this implementation, protein-wise is axis=0 and cell-wise is axis=1. Seurat's default is protein-wise, i.e. axis=0.

     The default is "protein-wise" (axis=0), i.e. normalize each protein independently.

    :param mat: Counts matrix (cells x proteins)
    :type mat: numpy array or scipy sparse matrix
    :param axis: normalize each antibody independently (axis=0) or normalize each cell independently (axis=1), defaults to 0
    :type axis: int, optional
    :return: Transformed counts matrix
    :rtype: numpy array
    """

    # TODO: handle sparse arrays sparsely
    # def _apply_to_dense_or_sparse_matrix_columns(func, X):
    # if scipy.sparse.issparse(X):
    # return scipy.sparse.vstack([func(X[rowidx, :]) for rowidx in range(X.shape[0])])
    # return np.apply_along_axis(func, axis, X)

    # apply to dense or sparse matrix, along axis. returns dense matrix
    return np.apply_along_axis(
        _seurat_clr, axis, (mat.A if scipy.sparse.issparse(mat) else mat)
    )


# TODO: Implement tf-idf like https://constantamateur.github.io/2020-04-10-scDE/ ?


def make_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    true_label: str,
    pred_label: str,
    label_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Make a confusion matrix. Pairs with ``genetools.plots.plot_confusion_matrix``."""
    # rows ground truth label - columns predicted label
    cm = pd.crosstab(
        np.array(y_true),
        np.array(y_pred),
        rownames=[true_label],
        colnames=[pred_label],
    )

    # reorder so columns and index match
    if label_order is None:
        label_order = cm.index.union(cm.columns)

    resulting_row_order, resulting_col_order = [
        pd.Index(label_order).intersection(source_list).tolist()
        for source_list in [
            cm.index,
            cm.columns,
        ]
    ]

    cm = cm.loc[resulting_row_order][resulting_col_order]

    return cm


def softmax(arr: np.ndarray) -> np.ndarray:
    """softmax a 1d vector or softmax all rows of a 2d array. ensures result sums to 1:

    - if input was a 1d vector, returns a 1d vector summing to 1.
    - if input was a 2d array, returns a 2d array where every row sums to 1.
    """
    orig_ndim = arr.ndim
    if orig_ndim == 1:
        # input array is a M-dim vector that we want to make sum to 1
        # but sklearn softmax expects a matrix NxM, so pass a 1xM matrix.
        # convert single vector into a single-row matrix, e.g. [1,2,3] to [[1, 2, 3]]:
        arr = np.reshape(arr, (1, -1))

    # Convert to probabilities with softmax
    probabilities = sklearn.utils.extmath.softmax(arr.astype(float), copy=False)

    if orig_ndim == 1:
        # Finish workaround for softmax when X has a single row: extract relevant softmax result (first row of matrix)
        probabilities = probabilities[0, :]

    return probabilities


def run_sigmoid_if_binary_and_softmax_if_multiclass(arr: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities using sigmoid if binary, or softmax if multiclass.

    Binary is standard logistic regression. Input should have shape `(n_samples, )` (as computed by decision_function() in sklearn), but the output will follow predict_proba's conventional output shape of `(n_samples, 2)`.

    Multiclass approach is softmax regression as in R glmnet and sklearn. See https://en.wikipedia.org/wiki/Multinomial_logistic_regression#As_a_set_of_independent_binary_regressions
    """
    # Check if binary or multiclass
    if arr.ndim == 1:
        # Apply sigmoid:
        # Positive class probability is expit(x) = 1/(1 + exp(-x)), where x is the logit
        positive_probability = scipy.special.expit(arr)
        return np.vstack((1 - positive_probability, positive_probability)).T
        # Note: we deliberately ignore this other sklearn code path: https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/linear_model/_logistic.py#L1471
        # That's a special case for "binary softmax classifier", but usually we want standard logistic regression. Extra context here: https://github.com/scikit-learn/scikit-learn/pull/11476#issuecomment-414346097
    else:
        return softmax(arr)


@dataclasses.dataclass
class ConfusionMatrixValues:
    true_positives: float
    false_negatives: float
    false_positives: float
    true_negatives: float


def unpack_confusion_matrix(y_true, y_pred, positive_label, negative_label):
    cm = sklearn.metrics.confusion_matrix(
        y_true, y_pred, labels=[positive_label, negative_label]
    )
    # | TP  FN |
    # | FP  TN |
    tp, fn, fp, tn = cm.ravel()
    # Alternatively, if we set labels=[negative_label, positive_label]:
    # | TN  FP |
    # | FN  TP |
    # tn, fp, fn, tp = cm.ravel()

    return ConfusionMatrixValues(
        true_positives=tp, false_negatives=fn, false_positives=fp, true_negatives=tn
    )


def interpolate_roc(y_true, y_score, n_points=1000):
    """
    Interpolate receiver operating characteristic curve.

    Returns:

    * interpolated FPR
    * interpolated TPR
    * original FPR
    * original TPR

    To plot:

    .. code-block:: python

        fig, ax = plt.subplots(figsize=(4, 4))
        plt.plot(
            fpr,
            tpr,
            label="Real",
            color="k",
            alpha=0.5,
            drawstyle="steps-post",
        )
        plt.plot(
            interpolated_fpr,
            interpolated_tpr,
            label="Interpolated",
            color="r",
            alpha=0.5,
            drawstyle="steps-post",
        )
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(bbox_to_anchor=(1, 1))
    """
    # Use piecewise constant interpolation due to stepwise nature of the ROC curve
    # See also https://stats.stackexchange.com/a/187003/297 and https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
    # First define a common set of specificity values along the x axis
    base_fpr = np.linspace(0, 1, n_points + 1)

    fpr, tpr, auc_thresholds = sklearn.metrics.roc_curve(
        y_true, y_score, drop_intermediate=False
    )
    tpr_interpolator = interp1d(
        fpr,
        tpr,
        # Stepwise constant interpolation:
        kind="previous",
        # Ensure that the function handles points outside the original FPR range by extrapolating to 0 at the beginning and 1 at the end, reflecting the behavior of a ROC curve:
        bounds_error=False,
        fill_value=(0.0, 1.0),
    )
    # Interpolate the TPR at these common FPR values
    tpr_interpolated = tpr_interpolator(base_fpr)
    # Interpolation struggles with boundaries. Let's explicitly set the start and end points so the curves match.
    # - When FPR is 0, TPR should also be 0 (no positive cases are predicted, so true positives should also be 0).
    # - When FPR is 1 (or the maximum FPR in our data), TPR should ideally be the maximum TPR from our data (representing the scenario where all cases are predicted as positive).
    # TPR should be 0 when FPR is 0:
    tpr_interpolated[0] = 0
    # Set the last point to the maximum of the original TPR:
    tpr_interpolated[-1] = tpr[-1]

    return base_fpr, tpr_interpolated, fpr, tpr


def interpolate_prc(y_true, y_score, n_points=1000):
    """
    Interpolate precision-recall curve.

    Returns:

    * interpolated recall
    * interpolated precision
    * original recall
    * original precision

    To plot:

    .. code-block:: python

        fig, ax = plt.subplots(figsize=(4, 4))
        plt.plot(
            recall,
            precision,
            label="Real",
            color="k",
            alpha=0.5,
            drawstyle="steps-post",
        )
        plt.plot(
            interpolated_recall,
            interpolated_precision,
            label="Interpolated",
            color="r",
            alpha=0.5,
            drawstyle="steps-post",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(bbox_to_anchor=(1, 1))
    """
    # Use piecewise constant interpolation due to stepwise nature of the PR curve
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_score)

    # Remove bounds that sklearn eadds
    recall_partial = recall[1:-1]
    precision_partial = precision[1:-1]

    # Generate our x axis points
    base_recall = np.linspace(min(recall_partial), max(recall_partial), 1001)
    base_recall = np.flip(base_recall)  # original is in reversed order too
    # Create interpolator. Undo the reverse order of the sklearn outputs
    pr_interpolator = interp1d(
        np.flip(recall_partial),
        np.flip(precision_partial),
        kind="next",
    )

    # Interpolate the precision at our new recall values
    precision_interpolated = pr_interpolator(base_recall)

    # Add boundary points:
    # After performing the interpolation, prepend the starting point (first value from precision, recall=1.0) and append the ending point (precision=1.0, last value from recall).
    # See sklearn.metrics.precision_recall_curve docstring
    precision_with_boundaries = np.concatenate(
        ([precision[0]], precision_interpolated, [1.0])
    )
    recall_with_boundaries = np.concatenate(([1.0], base_recall, [recall[-1]]))

    return recall_with_boundaries, precision_with_boundaries, recall, precision
