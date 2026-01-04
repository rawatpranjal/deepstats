"""Download utilities for benchmark datasets.

This module provides caching and download utilities for fetching
benchmark datasets from remote sources.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


def _get_cache_dir() -> Path:
    """Get the cache directory for deepstats datasets.

    Returns
    -------
    Path
        Path to cache directory (~/.deepstats/data).
    """
    cache_dir = Path.home() / ".deepstats" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _url_to_filename(url: str) -> str:
    """Convert a URL to a safe filename.

    Parameters
    ----------
    url : str
        The URL to convert.

    Returns
    -------
    str
        A safe filename derived from the URL.
    """
    # Use hash of URL for unique filename
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    # Extract filename from URL
    url_filename = url.split("/")[-1].split("?")[0]
    return f"{url_hash}_{url_filename}"


def _download_file(url: str, filename: str | None = None, force: bool = False) -> Path:
    """Download a file from URL with caching.

    Parameters
    ----------
    url : str
        URL to download from.
    filename : str, optional
        Filename to save as. If None, derived from URL.
    force : bool, default=False
        If True, re-download even if cached.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    RuntimeError
        If download fails.
    """
    import ssl

    cache_dir = _get_cache_dir()
    if filename is None:
        filename = _url_to_filename(url)
    filepath = cache_dir / filename

    if filepath.exists() and not force:
        return filepath

    # Download file with SSL fallback for certificate issues
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        # Try with unverified SSL context as fallback
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
        except Exception as e2:
            raise RuntimeError(f"Failed to download {url}: {e2}") from e2

    return filepath


def _load_csv_cached(
    url: str, filename: str | None = None, force: bool = False, **kwargs: Any
) -> pd.DataFrame:
    """Load a CSV file from URL with caching.

    Parameters
    ----------
    url : str
        URL to the CSV file.
    filename : str, optional
        Filename to save as. If None, derived from URL.
    force : bool, default=False
        If True, re-download even if cached.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    filepath = _download_file(url, filename, force)
    return pd.read_csv(filepath, **kwargs)


def _load_npz_cached(
    url: str, filename: str | None = None, force: bool = False
) -> dict:
    """Load a NPZ file from URL with caching.

    Parameters
    ----------
    url : str
        URL to the NPZ file.
    filename : str, optional
        Filename to save as. If None, derived from URL.
    force : bool, default=False
        If True, re-download even if cached.

    Returns
    -------
    dict
        Dictionary of arrays from the NPZ file.
    """
    import numpy as np

    filepath = _download_file(url, filename, force)
    return dict(np.load(filepath, allow_pickle=True))


def clear_cache() -> None:
    """Clear the deepstats data cache."""
    import shutil

    cache_dir = _get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
