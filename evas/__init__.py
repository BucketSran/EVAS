try:
    from importlib.metadata import version as _package_version
except ImportError:  # pragma: no cover - Python < 3.8 compatibility fallback
    from importlib_metadata import version as _package_version

try:
    __version__ = _package_version("evas-sim")
except Exception:
    __version__ = "0.4.6"
