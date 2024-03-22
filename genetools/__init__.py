"""Top-level package for genetools."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.7.2"

# We want genetools.[submodule].[function] to be accessible simply by importing genetools, without having to import genetools.[submodule]
from . import helpers, plots, stats, arrays

try:
    from . import scanpy_helpers  # noqa: F401

    extras_available = True
except ImportError:
    extras_available = False

__all__ = ["helpers", "plots", "stats", "arrays"]
if extras_available:
    __all__.append("scanpy_helpers")
