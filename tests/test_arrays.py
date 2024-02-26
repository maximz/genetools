import numpy as np
import pandas as pd
import pytest
import scipy.stats

import genetools.arrays


def test_convert_matrix_to_one_element_per_row():
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    df = genetools.arrays.convert_matrix_to_one_element_per_row(arr)
    df_expected = pd.DataFrame(
        np.array(
            [
                [0, 0, 1],
                [1, 0, 5],
                [0, 1, 2],
                [1, 1, 6],
                [0, 2, 3],
                [1, 2, 7],
                [0, 3, 4],
                [1, 3, 8],
            ]
        ),
        columns=["row_id", "col_id", "value"],
    )
    pd.testing.assert_frame_equal(df, df_expected)


def test_get_trim_both_sides_mask():
    arr = np.array([10, 20, 0, 30, 40, 50])
    weights = np.array([5, 6, 4, 7, 8, 9])
    proportiontocut = 0.2

    mask = genetools.arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )
    assert np.array_equal(arr[mask], [10, 20, 30, 40])
    assert np.array_equal(weights[mask], [5, 6, 7, 8])

    ###
    # bigger. but even
    arr = np.random.randint(low=1, high=100, size=50)
    proportiontocut = 0.1
    mask = genetools.arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )

    ###
    # now odd.
    arr = np.random.randint(low=1, high=100, size=49)
    proportiontocut = 0.1
    mask = genetools.arrays.get_trim_both_sides_mask(
        arr, proportiontocut=proportiontocut
    )
    assert np.array_equal(
        scipy.stats.trimboth(arr, proportiontocut=proportiontocut), arr[mask]
    )

    ###
    # now test a 2d matrix with a weight for each row
    arr = np.c_[
        np.array([0.8, 0.5, 0.6, 0.2, 0.3]), np.array([0.2, 0.5, 0.4, 0.3, 0.8])
    ]
    weights = np.array([3, 4, 1, 2, 5])

    mask = genetools.arrays.get_trim_both_sides_mask(arr, proportiontocut=0.2, axis=0)

    weights_horizontally_cloned = np.tile(
        weights[np.newaxis, :].transpose(), arr.shape[1]
    )
    assert arr.shape == weights_horizontally_cloned.shape == (5, 2)

    column_weighted_averages = np.average(
        a=np.take_along_axis(arr, mask, axis=0),
        weights=np.take_along_axis(weights_horizontally_cloned, mask, axis=0),
        axis=0,
    )
    assert np.array_equal(
        column_weighted_averages,
        [
            np.average([0.3, 0.5, 0.6], weights=[5, 4, 1]),
            np.average([0.3, 0.4, 0.5], weights=[2, 1, 4]),
        ],
    )


def test_add_dummy_variables():
    isotype_groups = ["IGHG", "IGHA", "IGHD-M", "IGHD-M"]
    assert genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    ).shape == (4, 3)


def test_add_dummy_variables_with_some_isotype_groups_missing():
    isotype_groups = ["IGHG", "IGHA", "IGHA", "IGHA"]
    assert genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    ).shape == (4, 3)


def test_add_dummy_variables_with_some_isotype_groups_missing_categorical_input():
    # Categorical input case is special because pd.get_dummies(categorical_series) doesn't allow adding further columns.
    isotype_groups = pd.Series(["IGHG", "IGHA", "IGHA", "IGHA"]).astype("category")
    assert genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    ).shape == (4, 3)


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_some_isotype_groups_missing_disallowed():
    isotype_groups = ["IGHG", "IGHA", "IGHA", "IGHA"]
    genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    )


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_unexpected_isotype_groups_fails():
    isotype_groups = ["IGHG", "IGHA", "IGHE", "IGHD-M"]
    genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=False,
    )


@pytest.mark.xfail(raises=ValueError)
def test_add_dummy_variables_with_unexpected_isotype_groups_fails_regardless_of_setting():
    isotype_groups = ["IGHG", "IGHA", "IGHE", "IGHD-M"]
    genetools.arrays.make_dummy_variables_in_specific_order(
        values=isotype_groups,
        expected_list=["IGHG", "IGHA", "IGHD-M"],
        allow_missing_entries=True,
    )


def test_weighted_median():
    # odd number of entries
    df = pd.DataFrame({"value": [0.3, 0.2, 0.1, 0.2, 0.2], "weight": [100, 1, 1, 1, 1]})
    # even number of entries
    df2 = pd.DataFrame(
        {"value": [0.3, 0.2, 0.1, 0.25, 0.26, 0.4], "weight": [100, 1, 1, 1, 1, 1]}
    )

    # standard median
    assert (
        df["value"].median()
        == genetools.arrays.weighted_median(df["value"], np.ones(df.shape[0]))
        == 0.2
    )
    assert df2["value"].median() == 0.255  # notice: interpolation
    assert (
        genetools.arrays.weighted_median(df2["value"], np.ones(df2.shape[0])) == 0.25
    )  # no interpolation

    # weighted median
    assert genetools.arrays.weighted_median(df["value"], df["weight"]) == 0.3
    assert (
        genetools.arrays.weighted_median(df2["value"], df2["weight"]) == 0.3
    )  # no interpolation


def test_weighted_value_counts():
    df = pd.DataFrame(
        {
            # notice that category C is unused
            "item": pd.Categorical(["A", "A", "B"], categories=["A", "B", "C"]),
            "count": [1, 1, 3],
        }
    )

    # Set up the tests
    assert_kwargs = dict(
        check_index_type=False,  # don't check CategoricalIndex vs standard series index,
        check_categorical=False,  # don't check whether entries are pd.Categorical
        check_names=False,
    )

    # Standard value_counts
    pd.testing.assert_series_equal(
        df["item"].value_counts(),
        pd.Series({"A": 2, "B": 1, "C": 0}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        df["item"].value_counts(normalize=True),
        pd.Series({"A": 2 / 3, "B": 1 / 3, "C": 0}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        df["item"].cat.remove_unused_categories().value_counts(),
        pd.Series({"A": 2, "B": 1}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        df["item"].cat.remove_unused_categories().value_counts(normalize=True),
        pd.Series({"A": 2 / 3, "B": 1 / 3}),
        **assert_kwargs,
    )

    # Weighted value counts
    pd.testing.assert_series_equal(
        genetools.arrays.weighted_value_counts(df, "item", "count"),
        pd.Series({"A": 2, "B": 3, "C": 0}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        genetools.arrays.weighted_value_counts(df, "item", "count", normalize=True),
        pd.Series({"A": 2 / 5, "B": 3 / 5, "C": 0}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        genetools.arrays.weighted_value_counts(df, "item", "count", observed=True),
        pd.Series({"A": 2, "B": 3}),
        **assert_kwargs,
    )
    pd.testing.assert_series_equal(
        genetools.arrays.weighted_value_counts(
            df, "item", "count", normalize=True, observed=True
        ),
        pd.Series({"A": 2 / 5, "B": 3 / 5}),
        **assert_kwargs,
    )


def test_weighted_value_counts_after_groupby():
    # Test weighted_value_counts within groupby
    df = pd.DataFrame(
        {
            "label": ["A", "A", "B", "A", "A"],
            "type": "entry",
            "item": pd.Categorical(
                ["A", "A", "B", "A", "B"], categories=["A", "B", "C"]
            ),
            "count": [2, 1, 3, 1, 1],
        }
    )

    # Set up the tests
    assert_kwargs = dict(
        check_dtype=False,  # don't check for CategoricalDtype
        check_categorical=False,  # don't check whether entries are Categorical
    )

    # standard
    # note, this is broken in pandas v1.4: columns are ['label', 'type', 'level_2', 'item_proportion'] instead of ['label', 'type', 'item', 'item_proportion']. fixed in v1.5.
    pd.testing.assert_frame_equal(
        (
            df.groupby(["label", "type"], observed=True)["item"]
            .value_counts(normalize=True)
            .to_frame(name="item_proportion")
            .reset_index()
        ),
        pd.DataFrame.from_records(
            [
                {"label": "A", "type": "entry", "item": "A", "item_proportion": 0.75},
                {"label": "A", "type": "entry", "item": "B", "item_proportion": 0.25},
                {"label": "A", "type": "entry", "item": "C", "item_proportion": 0.0},
                {"label": "B", "type": "entry", "item": "B", "item_proportion": 1.0},
                {"label": "B", "type": "entry", "item": "A", "item_proportion": 0.0},
                {"label": "B", "type": "entry", "item": "C", "item_proportion": 0.0},
            ]
        ),
        **assert_kwargs,
    )

    # weighted
    # note: we discourage this manual groupby. our preferred alternate syntax (wrapper) is tested below
    pd.testing.assert_frame_equal(
        (
            df.groupby(["label", "type"], observed=True)
            .apply(
                lambda grp: genetools.arrays.weighted_value_counts(
                    grp,
                    "item",
                    "count",
                    normalize=True,
                )
            )
            .stack()  # to be consistent with standard groupby->value_counts output shape
            .to_frame(name="item_proportion")
            .reset_index()
        ),
        pd.DataFrame.from_records(
            [
                {"label": "A", "type": "entry", "item": "A", "item_proportion": 0.8},
                {"label": "A", "type": "entry", "item": "B", "item_proportion": 0.2},
                {"label": "A", "type": "entry", "item": "C", "item_proportion": 0.0},
                {"label": "B", "type": "entry", "item": "A", "item_proportion": 0.0},
                {"label": "B", "type": "entry", "item": "B", "item_proportion": 1.0},
                {"label": "B", "type": "entry", "item": "C", "item_proportion": 0.0},
            ]
        ),
        **assert_kwargs,
    )

    ## Test groupby_apply_weighted_value_counts wrapper behavior on pathological test case where pandas returns a dataframe instead of a series:
    df = pd.DataFrame(
        [
            {"columnA": 1, "category_column": "A", "weight_column": 1},
            {"columnA": 1, "category_column": "B", "weight_column": 2},
            {"columnA": 2, "category_column": "A", "weight_column": 1},
            # It's this row which creates the issue:
            {"columnA": 2, "category_column": "B", "weight_column": 1},
        ]
    )
    df_not_pathological = df.iloc[:-1].copy()

    # First, confirm that standard pandas value_counts is always series
    assert (
        type(df_not_pathological.groupby("columnA")["category_column"].value_counts())
        is pd.Series
    )
    assert type(df.groupby("columnA")["category_column"].value_counts()) is pd.Series

    # Confirm that pandas gives pathological output if you do a manual groupby
    assert (
        type(
            df_not_pathological.groupby(["columnA"], observed=True).apply(
                lambda grp: genetools.arrays.weighted_value_counts(
                    grp,
                    "category_column",
                    "weight_column",
                    normalize=True,
                )
            )
        )
        is pd.Series
    )
    # this here is the problem with the manual approach:
    assert (
        type(
            df.groupby(["columnA"], observed=True).apply(
                lambda grp: genetools.arrays.weighted_value_counts(
                    grp,
                    "category_column",
                    "weight_column",
                    normalize=True,
                )
            )
        )
        is pd.DataFrame
    )

    # Confirm that our special groupby_weighted_value_counts is always series
    assert (
        type(
            genetools.arrays.groupby_apply_weighted_value_counts(
                df_not_pathological,
                ["columnA"],
                observed=True,
                category_column_name="category_column",
                weight_column_name="weight_column",
                normalize=True,
            )
        )
        is pd.Series
    )
    assert (
        type(
            genetools.arrays.groupby_apply_weighted_value_counts(
                df,
                ["columnA"],
                observed=True,
                category_column_name="category_column",
                weight_column_name="weight_column",
                normalize=True,
            )
        )
        is pd.Series
    )
