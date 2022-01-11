import pytest
import numpy as np
import scanpy as sc
import pandas as pd
import os
import random
import anndata
import warnings

# TODO: why didn't this solve the reproducibility problem across machines?
random_seed = 12345
np.random.seed(random_seed)
random.seed(random_seed)

# filenames for data persistence
cache_keys = {
    "obsm": {
        "X_diffmap": "data/diffmap.mat.gz",
        "X_pca": "data/pca.mat.gz",
        "X_umap": "data/umap.mat.gz",
    },
    "varm": {"PCs": "data/pcs.mat.gz"},
    "obs": "data/obs.csv.gz",
    "var": "data/var.csv.gz",
    "uns": "data/adata_uns.h5",
    "local_machine_anndata": "data/pbmc3k.h5ad",
}

# Accept command line argument controlling anndata regeneration
# https://docs.pytest.org/en/latest/example/simple.html#pass-different-values-to-a-test-function-depending-on-command-line-options
# see also https://stackoverflow.com/a/42145604/130164


def pytest_addoption(parser):
    # Require this pytest cli flag to regenerate anndata.
    parser.addoption(
        "--regenerate-anndata",
        action="store_true",
        default=False,
        help="Regenerate test anndata",
    )


@pytest.fixture(scope="session")
def adata(request):
    """Fixture that returns an anndata object.

    The processed version can be cached between test runs at ./data/pbmc3k.h5ad. For test speed during local development.
    But it is not versioned. So every first-time user cloning the repo will recreate this object. So will CI.
    Therefore, the locally cached version is optional.
    > To regenerate just the locally cached version: `rm data/pbmc3k.h5ad`

    However, to ensure the data is consistent between test suite executions on different machines,
    we _do_ version the obsm, varm, obs, var, and uns fields of the anndata.
    This is required to run the test suite.

    To regenerate this test data: `rm -r data && make regen-test-data`
    Note: We only allow recreating the anndata when pytest is called with --regenerate-anndata flag.
    If no cached values are available, and --regenerate-anndata is not specified, the tests will error out.

    """

    # Require this pytest cli flag to regenerate anndata.
    regenerate_anndata = request.config.getoption("--regenerate-anndata")

    # cache output of this fixture for local test speed
    if os.path.exists(cache_keys["local_machine_anndata"]) and not regenerate_anndata:
        return sc.read(cache_keys["local_machine_anndata"])

    adata = _make_adata(regenerate_anndata)
    adata.write(cache_keys["local_machine_anndata"], compression="gzip")
    return adata


def _make_adata(regenerate_anndata):
    """Generates anndata object.
    This downloads 5.9 MB of data upon the first call of the function and stores it in ./data/pbmc3k_raw.h5ad.
    """
    # Following https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    adata = sc.datasets.pbmc3k()

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    mito_genes = adata.var_names.str.startswith("MT-")
    adata.obs["percent_mito"] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(
        adata.X, axis=1
    )
    adata.obs["n_counts"] = adata.X.sum(axis=1).A1
    adata = adata[adata.obs.n_genes < 2500, :]
    adata = adata[adata.obs.percent_mito < 0.05, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.diffmap(adata)  # for pseudotime
    sc.tl.umap(adata)
    sc.tl.louvain(adata)

    # these will be out of order
    # TODO: label marker genes and do this the right way with Bindea list
    new_cluster_names = [
        "CD4 T",
        "CD14 Monocytes",
        "B",
        "CD8 T",
        "NK",
        "FCGR3A Monocytes",
        "Dendritic",
        "Megakaryocytes",
    ]
    # TODO:
    # adata.obs['louvain_annotated'] = adata.obs['louvain'].copy()
    # adata.rename_categories("louvain_annotated", new_cluster_names)
    adata.rename_categories("louvain", new_cluster_names)

    # Pull umap into obs
    adata.obs["umap_1"] = adata.obsm["X_umap"][:, 0]
    adata.obs["umap_2"] = adata.obsm["X_umap"][:, 1]

    # Pull a gene value into obs
    adata.obs["CST3"] = adata[:, "CST3"].X

    # Try to replace all this with the saved values, if they exist
    if regenerate_anndata:
        warnings.warn("Cached anndata properties not loaded. Recreating...")
        # Persist the anndata properties.
        _export_adata(adata)
    else:
        # import cached anndata properties
        adata = _import_adata(adata)
    return adata


def _export_adata(adata):
    """Persist anndata properties for test suite reproducibility."""

    # obsm
    for key, fname in cache_keys["obsm"].items():
        np.savetxt(fname, adata.obsm[key], fmt="%.4e")

    # varm
    for key, fname in cache_keys["varm"].items():
        np.savetxt(fname, adata.varm[key], fmt="%.4e")

    # obs
    adata.obs.to_csv(cache_keys["obs"])

    # var
    adata.var.to_csv(cache_keys["var"])

    # uns
    adata_uns_only = anndata.AnnData()
    adata_uns_only.uns = adata.uns
    adata_uns_only.write(cache_keys["uns"], compression="gzip")


def _import_adata(adata):
    """Load persisted anndata properties for test suite reproducibility."""
    # obsm
    for key, fname in cache_keys["obsm"].items():
        adata.obsm[key] = np.loadtxt(fname)

    # varm
    for key, fname in cache_keys["varm"].items():
        adata.varm[key] = np.loadtxt(fname)

    # obs
    adata.obs = pd.read_csv(cache_keys["obs"], index_col=0)

    # var
    adata.var = pd.read_csv(cache_keys["var"], index_col=0)

    # uns
    adata_uns = sc.read(cache_keys["uns"])
    adata.uns = adata_uns.uns

    return adata
