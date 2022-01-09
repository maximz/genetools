"""Pandas/Numpy common recipes."""

import os
import scipy
import numpy as np
import pandas as pd


def rename_duplicates(series, delim="-"):
    """Rename duplicate values to be unique. ['a', 'a'] will become ['a', 'a-1'], for example.

    :param series: series with values to rename
    :type series: pandas.Series
    :param delim: delimeter before duplicate-number index, defaults to "-"
    :type delim: str, optional
    :return: series where original duplicates have been renamed to -1, -2, etc.
    :rtype: pandas.Series
    """
    duplicate_suffix = (
        series.groupby(series).cumcount().astype(str).replace("0", "")
    )  # a number for all but first occurence
    extra_strs = delim + duplicate_suffix
    # remove entries that are just the delim
    extra_strs = extra_strs.replace(delim, "")
    # add to values
    out = series.astype(str) + extra_strs
    # confirm unique (may fail if a-1 happened to match another element that preexisted!)
    assert out.nunique() == out.shape[0]
    return out


def merge_into_left(left, right, **kwargs):
    """Defensively merge [right] series or dataframe into [left] by index, preserving [left]'s index exactly. [right] data will be reordered to match [left] index.

    :param left: left data whose index will be preserved
    :type left: pandas.DataFrame or pandas.Series
    :param right: right data which will be reordered based on left index.
    :type right: pandas.DataFrame or pandas.Series
    :param \**kwargs: passed to pandas.merge
    :return: left-merged DataFrame with [left]'s index
    :rtype: pandas.DataFrame
    """
    # defensively cast to dataframe
    df1 = pd.DataFrame(left)
    df2 = pd.DataFrame(right)
    df = pd.merge(
        df1,
        df2,
        how="left",
        left_index=True,
        right_index=True,
        sort=False,
        validate="1:1",
        **kwargs
    )
    # TODO: asserts are stripped away when code is optimized; replace with if not, raise ValueError('message')
    assert df.shape[0] == df1.shape[0]
    assert df.shape[1] == df1.shape[1] + df2.shape[1]
    df.index = df1.index
    return df


def sample_cells_from_clusters(obs_df, n_cells, cluster_key, cluster_names):
    """[summary]

    :param obs_df: [description]
    :type obs_df: [type]
    :param n_cells: [description]
    :type n_cells: [type]
    :param cluster_key: [description]
    :type cluster_key: [type]
    :param cluster_names: [description]
    :type cluster_names: [type]
    :return: [description]
    :rtype: [type]
    """
    return (
        obs_df[obs_df[cluster_key].isin(cluster_names)]
        .index.to_series()
        .sample(n_cells)
        .values
    )


def horizontal_concat(df_left, df_right):
    """Concatenate df_right horizontally to df_left, with no checks for whether the indexes match, but confirming final shape.

    :param df_left: Left data
    :type df_left: pandas.DataFrame or pandas.Series
    :param df_right: Right data
    :type df_right: pandas.DataFrame or pandas.Series
    :return: Copied dataframe with df_right's columns glued onto the right side of df_left's columns
    :rtype: pandas.DataFrame
    """
    # defensively cast to DataFrame
    df1 = pd.DataFrame(df_left)
    df2 = pd.DataFrame(df_right)

    df = pd.concat([df1, df2], axis=1)
    assert df.shape[0] == df1.shape[0] == df2.shape[0]
    assert df.shape[1] == df1.shape[1] + df2.shape[1]
    return df


def vertical_concat(df_top, df_bottom, reset_index=False):
    """Concatenate df_bottom vertically to df_top, with no checks for whether the columns match, but confirming final shape.

    :param df_top: Top data
    :type df_top: pandas.DataFrame
    :param df_bottom: Bottom data
    :type df_bottom: pandas.DataFrame
    :param reset_index: Reset index values after concat, defaults to False
    :type reset_index: bool, optional
    :return: Copied dataframe with df_bottom's rows glued onto the bottom of df_top's rows
    :rtype: pandas.DataFrame
    """
    # defensively cast to DataFrame
    df1 = pd.DataFrame(df_top)
    df2 = pd.DataFrame(df_bottom)

    df = pd.concat([df1, df2], axis=0)
    if reset_index:
        # so far indexes have just been glued together
        # we can reset it to be unique values
        df = df.reset_index(drop=True)
    assert df.shape[0] == df1.shape[0] + df2.shape[0]
    assert df.shape[1] == df1.shape[1] == df2.shape[1]
    return df


def barcode_split(
    obs_names, separator="-", colname_barcode="barcode", colname_library="library_id"
):
    """Split single cell barcodes such as ATGC-1 into a barcode column with value "ATGC" and a library ID column with value 1.

    Recommended usage with scanpy:
    adata.obs = horizontal_concat(adata.obs, barcode_split(adata.obs_names))

    :param obs_names: Cell barcodes with a library ID suffix.
    :type obs_names: pandas.Series or pandas.Index
    :param separator: library ID separator, defaults to '-'
    :type separator: str, optional
    :param colname_barcode: output column name containing barcode without library ID suffix, defaults to 'barcode'
    :type colname_barcode: str, optional
    :param colname_library: output column name containing library ID suffix as an int, defaults to 'library_id'
    :type colname_library: str, optional
    :return: Two-column dataframe containing barcode prefix and library ID suffix.
    :rtype: pandas.DataFrame
    """
    # defensively cast to a string Series in case an Index was passed, such as adata.obs_names or adata.obs.index
    df = pd.Series(obs_names, dtype="str").str.split(separator, expand=True)
    df.columns = [colname_barcode, colname_library]
    df[colname_library] = df[colname_library].astype(int)
    return df


def get_off_diagonal_values(arr):
    """Get off-diagonal values of a numpy 2d array as a flattened 1d array.

    :param arr: input numpy 2d array
    :type arr: numpy.ndarray
    :return: flattened 1d array of non-diagonal values only
    :rtype: numpy.ndarray
    """
    # See https://stackoverflow.com/a/35746928/130164
    return arr[~np.eye(arr.shape[0], dtype=bool)].flatten()


def make_slurm_command(
    script,
    job_name,
    log_path,
    env=None,
    options={},
    job_group_name="",
    wrap_script=True,
):
    """Generate slurm sbatch command. Should be pipe-able straight to bash.

    Automatic log filenames will take the format:
        - `{{ log_path }}/{{ job_group_name (optional) }}/{{ job_name }}.out` for stdout
        - `{{ log_path }}/{{ job_group_name (optional) }}/{{ job_name }}.err` for stderr

    You can override automatic log filenames by manually supplying "output" and "error" values in the `options` dict.

    :param script: path to an executable script, or inline script (if wrap_script is True)
    :type script: str
    :param job_name: job name, used for naming log files
    :type job_name: str
    :param log_path: destination for log files.
    :type log_path: str
    :param env: any environment variables to pass to script, defaults to None
    :type env: dict, optional
    :param options: any CLI options for sbatch, defaults to {}
    :type options: dict, optional
    :param job_group_name: optional group name for this job and related jobs, used for naming log files, defaults to ""
    :type job_group_name: str, optional
    :param wrap_script: whether the script is inline as opposed to a file on disk, defaults to True
    :type wrap_script: bool, optional
    :return: an sbatch command
    :rtype: str
    """
    log_fname_prefix = os.path.join(log_path, job_group_name, job_name)
    if "output" not in options:
        options["output"] = log_fname_prefix + ".out"
    if "error" not in options:
        options["error"] = log_fname_prefix + ".err"

    options_items = ['--%s="%s"' % (name, val) for name, val in options.items()]
    options_string = " ".join(options_items)

    variable_string = ""
    if env is not None:
        variable_items = ['"%s"="%s"' % (name, val) for name, val in env.items()]
        variable_string = "--export=" + ",".join(variable_items)

    # Very important to wrap the "wrap script" in single quotes,
    # so that the script will pick up the exported variables during execution.
    script_string = "--wrap='%s'" % script if wrap_script else script
    return "sbatch {options} {variables} {script};".format(
        options=options_string, variables=variable_string, script=script_string
    )
