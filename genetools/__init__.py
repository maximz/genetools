"""Top-level package for genetools."""

__author__ = """Maxim Zaslavsky"""
__email__ = "maxim@maximz.com"
__version__ = "0.7.6"

# We want genetools.[submodule].[function] to be accessible simply by importing genetools, without having to import genetools.[submodule]
from . import helpers, plots, stats, arrays

try:
    from . import scanpy_helpers  # noqa: F401

    extras_available = True
except Exception as err:
    # Optional Scanpy imports can fail for reasons beyond missing packages,
    # such as numba cache initialization. Keep core genetools importable.
    _scanpy_helpers_import_error = err
    extras_available = False

__all__ = ["helpers", "plots", "stats", "arrays"]
if extras_available:
    __all__.append("scanpy_helpers")


def __getattr__(name):
    if name == "scanpy_helpers" and not extras_available:
        raise ImportError(
            "genetools.scanpy_helpers requires optional Scanpy dependencies. "
            "Install them with `pip install 'genetools[scanpy]'`."
        ) from _scanpy_helpers_import_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
