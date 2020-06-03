#!/usr/bin/env python

"""Tests for `genetools` package."""

import pytest

# import sys
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
