"""Scanpy common recipes."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    # Import optional dependencies
    import anndata
    import scanpy as sc
except ImportError as e:
    raise ImportError(
        "This module requires some optional dependencies. Please install them by running `pip install 'genetools[scanpy]'`"
    ) from e

import sklearn.decomposition
import sklearn.preprocessing
from sklearn.utils import gen_batches

import logging
from genetools import stats

logger = logging.getLogger(__name__)


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


def scale_anndata(
    adata: anndata.AnnData,
    scale_transformer: Optional[sklearn.preprocessing.StandardScaler] = None,
    inplace=False,
    set_raw=False,
    **kwargs,
) -> Tuple[anndata.AnnData, sklearn.preprocessing.StandardScaler]:
    """
    Scale anndata, like with scanpy.pp.scale.
    Accepts pre-computed StandardScaler preprocessing transformer, so you can apply the same scaling to multiple anndatas.

    Args:

    * ``scale_transformer``: pre-defined preprocessing transformer to scale adata.X
    * ``inplace``: whether to modify input adata in place
    * ``set_raw``: whether to set adata.raw equal to input adata

    Returns: ``adata, scale_transformer``
    """
    # TODO: set var and uns parameters too, and support max_value clipping like in in scanpy
    if scale_transformer is None:
        scale_transformer = sklearn.preprocessing.StandardScaler(**kwargs).fit(adata.X)

    if inplace:
        if set_raw:
            adata.raw = adata
        adata.X = scale_transformer.transform(adata.X).astype(adata.X.dtype)
    else:
        # Copy, but be very memory-frugal about it -- try not to allocate memory we won't need (i.e. don't waste RAM copying old adata.X)
        # TODO: consider anndata._mutated_copy(X=X)
        old_adata = adata
        adata = anndata.AnnData(
            X=scale_transformer.transform(adata.X).astype(adata.X.dtype),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=adata.uns.copy(),
            obsm=adata.obsm.copy(),
            varm=adata.varm.copy(),
            layers=adata.layers.copy(),
            raw=(adata.raw.copy() if (adata.raw is not None and not set_raw) else None),
            obsp=adata.obsp.copy(),
            varp=adata.varp.copy(),
        )
        if set_raw:
            adata.raw = old_adata

    return adata, scale_transformer


def scale_train_and_test_anndatas(
    adata_train, adata_test=None, inplace=False, set_raw=False, **kwargs
):
    """
    Scale train anndata (like with scanpy.pp.scale), then apply same scaling to test anndata -- as opposed to scaling them independently.

    If ``adata_test`` isn't supplied, this just scales ``adata_train`` indpendently.
    """
    adata_train_scaled, scale_transformer = scale_anndata(
        adata_train, scale_transformer=None, inplace=inplace, set_raw=set_raw, **kwargs
    )
    adata_test_scaled = None
    if adata_test is not None:
        adata_test_scaled, _ = scale_anndata(
            adata_test,
            scale_transformer=scale_transformer,
            inplace=inplace,
            set_raw=set_raw,
            **kwargs,
        )
    return adata_train_scaled, adata_test_scaled


def pca_anndata(
    adata: anndata.AnnData,
    pca_transformer: Optional[
        Union[sklearn.decomposition.PCA, sklearn.decomposition.IncrementalPCA]
    ] = None,
    n_components=None,
    inplace=True,
    **kwargs,
) -> Tuple[
    anndata.AnnData,
    Union[sklearn.decomposition.PCA, sklearn.decomposition.IncrementalPCA],
]:
    """
    PCA anndata, like with scanpy.pp.pca.
    Accepts pre-computed PCA transformer, so you can apply the same PCA to multiple anndatas.

    Args:

    * ``pca_transformer``: pre-defined preprocessing transformer to run PCA on adata.X
    * ``n_components``: number of PCA components
    * ``inplace``: whether to modify input adata in place

    Returns: ``adata, pca_transformer``
    """
    # TODO: set var and uns parameters too

    is_data_big = adata.X.nbytes > 100e9  # 100GB, threshold measured in bytes
    batch_size = 1000000

    if pca_transformer is None:
        if is_data_big:
            # Special case: For very large datasets, use IncrementalPCA instead of PCA.
            pca_transformer = sklearn.decomposition.IncrementalPCA(
                n_components=n_components, batch_size=batch_size, **kwargs
            )
            # IncrementalPCA automatically fits in batches.
            # But if you call fit(), the built-in batching first casts all the data to float64.
            # We want to avoid having all the data in float64 at the same time (unnecessary memory pressure), so we do our own batching while still in our input dtype (e.g. float16), calling partial_fit() directly.
            # (float64 internally is probably important for the linear algebra algorithm stability and numerical precision.)
            for batch in gen_batches(
                adata.X.shape[0], batch_size, min_batch_size=n_components or 0
            ):
                # Pass in one batch. It will get cast to float64 internally.
                pca_transformer.partial_fit(adata.X[batch])
        else:
            # Use standard PCA for smaller datasets
            pca_transformer = sklearn.decomposition.PCA(
                n_components=n_components, **kwargs
            ).fit(adata.X)

    # Unlike scale_anndata, here we're not being careful to avoid unnecessary copying, since obsm is usually not that big
    adata = adata.copy() if not inplace else adata

    if is_data_big:
        # Special case for very large datasets: batch the PCA transformation.
        # There are two reasons:
        # - the built-in transform() method starts by casting full input to float64, which we want to avoid because it will quickly fill up RAM.
        # - PCA output starts at a higher dtype float64 which could also aggressively fill up RAM before we get the chance to convert back to float16.

        # Process in batches using gen_batches, which preserves order
        transformed_data = []
        for batch in gen_batches(adata.X.shape[0], batch_size):
            batch_transformed = pca_transformer.transform(adata.X[batch])
            transformed_data.append(
                batch_transformed.astype(adata.X.dtype)
            )  # Cast to lower dtype

        # Combine transformed batches and assign to adata
        adata.obsm["X_pca"] = np.vstack(transformed_data)
        del transformed_data
    else:
        # Use standard PCA transform() for smaller datasets
        adata.obsm["X_pca"] = pca_transformer.transform(adata.X).astype(adata.X.dtype)

    return adata, pca_transformer


def pca_train_and_test_anndatas(
    adata_train, adata_test=None, n_components=None, inplace=True, **kwargs
):
    """
    PCA train anndata (like with scanpy.pp.pca), then apply same PCA to test anndata -- as opposed to PCAing them independently.

    If ``adata_test`` isn't supplied, this just scales ``adata_train`` independently.
    """
    adata_train_pcaed, pca_transformer = pca_anndata(
        adata_train,
        pca_transformer=None,
        n_components=n_components,
        inplace=inplace,
        **kwargs,
    )
    adata_test_pcaed = None
    if adata_test is not None:
        adata_test_pcaed, _ = pca_anndata(
            adata_test,
            pca_transformer=pca_transformer,
            n_components=n_components,
            inplace=inplace,
            **kwargs,
        )
    return adata_train_pcaed, adata_test_pcaed


def umap_anndata(
    adata,
    umap_transformer=None,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
    inplace=True,
    use_rapids=False,
    use_pca=False,
    **kwargs,
):
    """
    UMAP anndata, like with scanpy.tl.umap.
    Accepts pre-computed UMAP transformer, so you can apply the same UMAP to multiple anndatas.

    Args:

    * ``umap_transformer``: pre-defined preprocessing transformer to run UMAP on adata.X
    * ``n_components``: number of UMAP components
    * ``inplace``: whether to modify input adata in place

    Anndata should already be scaled.

    Returns: ``adata, umap_transformer``
    """
    if use_rapids:
        # GPU support
        from cuml import UMAP
    else:
        from umap import UMAP

    if use_pca:
        # Allow using adata.obsm["X_pca"] if it exists
        if "X_pca" not in adata.obsm:
            # PCA must be precomputed
            use_pca = False
            logger.warning(
                "X_pca not found in adata.obsm, so not using PCA representation for UMAP despite use_pca=True"
            )

    if umap_transformer is None:
        umap_transformer = UMAP(
            n_neighbors=n_neighbors, n_components=n_components, **kwargs
        ).fit(adata.obsm["X_pca"] if use_pca else adata.X)

    # Unlike scale_anndata, here we're not being careful to avoid unnecessary copying, since obsm is usually not that big
    adata = adata.copy() if not inplace else adata

    # TODO: set var and uns parameters too
    adata.obsm["X_umap"] = umap_transformer.transform(
        adata.obsm["X_pca"] if use_pca else adata.X
    ).astype(adata.obsm["X_pca"].dtype if use_pca else adata.X.dtype)

    return adata, umap_transformer


def umap_train_and_test_anndatas(
    adata_train,
    adata_test=None,
    n_neighbors: Optional[int] = None,
    n_components: Optional[int] = None,
    inplace=True,
    use_rapids=False,
    use_pca=False,
    **kwargs,
):
    """
    UMAP train anndata (like with scanpy.tl.umap), then apply same UMAP to test anndata -- as opposed to PCAing them independently.

    If ``adata_test`` isn't supplied, this just scales ``adata_train`` independently.
    """
    adata_train_umaped, umap_transformer = umap_anndata(
        adata_train,
        umap_transformer=None,
        n_neighbors=n_neighbors,
        n_components=n_components,
        inplace=inplace,
        use_rapids=use_rapids,
        use_pca=use_pca,
        **kwargs,
    )
    adata_test_umaped = None
    if adata_test is not None:
        adata_test_umaped, _ = umap_anndata(
            adata_test,
            umap_transformer=umap_transformer,
            n_neighbors=n_neighbors,
            n_components=n_components,
            inplace=inplace,
            use_rapids=use_rapids,
            use_pca=use_pca,
            **kwargs,
        )
    return adata_train_umaped, adata_test_umaped
