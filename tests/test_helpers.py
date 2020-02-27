#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest
import numpy as np
import pandas as pd

from genetools import helpers


def test_rename_duplicates():
    actual = helpers.rename_duplicates(pd.Series(["a", "b", "a", "a"]))
    expected = pd.Series(["a", "b", "a-1", "a-2"])
    print(actual)
    assert expected.equals(actual)


def test_horizontal_concat():
    df1 = pd.Series([1, 2, 3])
    df2 = pd.DataFrame({"a": [3, 2, 1], "b": [0, 1, 2]})
    actual = helpers.horizontal_concat(df1, df2)
    expected = pd.DataFrame({0: [1, 2, 3], "a": [3, 2, 1], "b": [0, 1, 2]})
    assert expected.equals(actual)


def test_vertical_concat():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 0]})
    df2 = pd.DataFrame({"a": [3, 2, 1], "b": [0, 1, 2]})
    actual = helpers.vertical_concat(df1, df2)
    expected = pd.DataFrame(
        {"a": [1, 2, 3, 3, 2, 1], "b": [2, 1, 0, 0, 1, 2]}, index=[0, 1, 2, 0, 1, 2]
    )
    print(actual)
    assert expected.equals(actual)


def test_vertical_concat_reset_index():
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [2, 1, 0]})
    df2 = pd.DataFrame({"a": [3, 2, 1], "b": [0, 1, 2]})
    actual = helpers.vertical_concat(df1, df2, reset_index=True)
    expected = pd.DataFrame({"a": [1, 2, 3, 3, 2, 1], "b": [2, 1, 0, 0, 1, 2]})
    print(actual)
    assert expected.equals(actual)


def test_barcode_split():
    # test with a pandas.Index to make sure casting works right
    actual = helpers.barcode_split(
        pd.Index(["A_1", "B_1", "A_2"]),
        separator="_",
        colname_barcode="bar",
        colname_library="lid",
    )
    expected = pd.DataFrame({"bar": ["A", "B", "A"], "lid": [1, 1, 2]})
    assert expected.equals(actual)


def test_merge_into_left():
    left = pd.Series([1, 2, 3], index=[0, 1, 2], name="A")
    right = pd.Series(["C", "A", "D"], index=[2, 0, 3], name="B")
    # test kwargs as well
    actual = helpers.merge_into_left(left, right, suffixes=("_left", "_right"))
    expected = pd.DataFrame({"A": [1, 2, 3], "B": ["A", np.nan, "C"]}, index=[0, 1, 2])
    assert expected.equals(actual)


def test_get_off_diagonal_values():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert np.array_equal(
        helpers.get_off_diagonal_values(arr), np.array([2, 3, 4, 6, 7, 8])
    )
