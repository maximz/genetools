"""Scanpy common recipes."""

from . import stats

# TODO: enable speeding this up by using highly variable genes only?
def find_all_markers(
    adata,
    cluster_key,
    pval_cutoff=0.05,
    log2fc_min=0.25,
    key_added="rank_genes_groups",
    test="wilcoxon",
    use_raw=True,
):
    """Find differentially expressed marker genes for each group of cells.

    :param adata: Scanpy/anndata object
    :type adata: anndata.AnnData
    :param cluster_key: The adata.obs column name that defines groups for finding distinguishing marker genes.
    :type cluster_key: str
    :param pval_cutoff: Only return markers that have an adjusted p-value below this threshold. Defaults to 0.05. Set to None to disable filtering.
    :type pval_cutoff: float, optional
    :param log2fc_min: Limit testing to genes which show, on average, at least X-fold difference (log-scale) between the two groups of cells. Defaults to 0.25. Set to None to disable filtering.
    :type log2fc_min: float, optional
    :param key_added: The key in adata.uns information is saved to, defaults to "rank_genes_groups"
    :type key_added: str, optional
    :param test: Statistical test to use, defaults to "wilcoxon" (Wilcoxon rank-sum test), see scanpy.tl.rank_genes_groups documentation for other options
    :type test: str, optional
    :param use_raw: Use raw attribute of adata if present, defaults to True
    :type use_raw: bool, optional
    :return: Dataframe with ranked marker genes for each cluster. Important columns: gene, rank, [cluster_key] (same as argument value)
    :rtype: pandas.DataFrame
    """

    import scipy
    import numpy as np
    import pandas as pd
    import scanpy as sc

    # Compute marker genes
    sc.tl.rank_genes_groups(
        adata, groupby=cluster_key, key_added=key_added, use_raw=use_raw, method=test
    )

    # Get each individual cluster's genes
    clusters = adata.uns[key_added]["names"].dtype.names  # this is a np recarray
    cluster_dfs = []
    for cluster_id in clusters:
        cluster_df = sc.get.rank_genes_groups_df(
            adata,
            group=cluster_id,
            key=key_added,
            pval_cutoff=pval_cutoff,
            log2fc_min=log2fc_min,
        )
        cluster_df[cluster_key] = cluster_id
        cluster_df["rank"] = pd.Series(range(cluster_df.shape[0]))
        cluster_dfs.append(cluster_df)

    return (
        pd.concat(cluster_dfs, axis=0)
        .rename(columns={"names": "gene"})
        .reset_index(drop=True)
    )


def clr_normalize(adata, axis=0, inplace=True):
    """Centered log ratio transformation for Cite-seq data, normalizing:

    * each protein's count vectors across cells (axis=0, normalizing each column of the cells x proteins matrix, default)
    * or the antibody count vector for each cell (axis=1, normalizing each row of the cells x proteins matrix)

    This is a wrapper of `genetools.stats.clr_normalize(matrix, axis)`.

    :param adata: Protein counts anndata
    :type adata: anndata.AnnData
    :param axis: normalize each antibody independently (axis=0) or normalize each cell independently (axis=1), defaults to 0
    :type axis: int, optional
    :param inplace: whether to modify input anndata, defaults to True
    :type inplace: bool, optional
    :return: Transformed anndata
    :rtype: anndata.AnnData
    """
    # TODO: expose this as a decorator for genetools.stats.clr_normalize ?
    if not inplace:
        adata = adata.copy()
    adata.X = stats.clr_normalize(adata.X, axis=axis)
    return adata
