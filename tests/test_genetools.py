#!/usr/bin/env python

"""Tests for `genetools` package."""

import genetools
import requests
import os
import subprocess


def test_importability():
    """Want to make sure submodules are reachable"""
    # import genetools
    # assert genetools.helpers is not None
    # assert genetools.plots is not None
    # assert genetools.scanpy_helpers is not None
    # assert genetools.stats is not None
    # assert 'genetools.plots' in sys.modules.keys()

    # Hacky way to get a clean python environment without all the pytest env stuff
    # Because the above won't fail
    subprocess.run(["python", "-c", "import genetools; genetools.plots;"], check=True)


def test_scanpy_not_mandatory_import():
    """Confirm that scanpy_helpers does not import scanpy automatically (until methods are called)."""
    # import genetools.scanpy_helpers
    # assert 'scanpy' not in sys.modules.keys()

    # Hacky way to get a clean python environment without all the pytest env stuff
    # Because the above won't fail
    subprocess.run(
        [
            "python",
            "-c",
            "import genetools; import sys; assert 'scanpy' not in sys.modules.keys()",
        ],
        check=True,
    )


def test_version_number_not_yet_on_pypi():
    """For release branches, check that the version number is not yet on PyPI, so we remember to bump the version number.
    Run only on pull request builds against master branch. (Not run on push builds of master branch itself)

    PyPI API docs:
    - https://warehouse.pypa.io/api-reference/json/

    Environment variable docs:
    - https://unhashable.com/getting-the-current-branch-name-during-a-pull-request-in-travis-ci/
    - https://docs.travis-ci.com/user/environment-variables
    """
    is_pr = os.environ.get("TRAVIS_PULL_REQUEST", "false") != "false"
    targets_master = os.environ.get("TRAVIS_BRANCH", "") == "master"

    if is_pr and targets_master:
        # if the release does not exist yet, this version-specific lookup should 404
        assert (
            requests.get(
                "https://pypi.org/pypi/genetools/{}/json".format(genetools.__version__)
            ).status_code
            == 404
        ), "This version number already exists on pypi."
