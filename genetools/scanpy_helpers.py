"""Scanpy common recipes."""

import scipy
import numpy as np
import pandas as pd
import scanpy as sc

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
