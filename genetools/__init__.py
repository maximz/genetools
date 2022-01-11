"""Top-level package for genetools."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.5.0"

# We want genetools.[submodule].[function] to be accessible simply by importing genetools, without having to import genetools.[submodule]
from . import helpers, plots, scanpy_helpers, stats

__all__ = ["helpers", "plots", "scanpy_helpers", "stats"]
