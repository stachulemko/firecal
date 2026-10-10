"""HTTP helpers: a retrying session and atomic streaming downloads."""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import USER_AGENT

_local = threading.local()


def session() -> requests.Session:
    """One pooled session per thread (requests.Session is not thread-safe)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(total=5, connect=5, read=5, backoff_factor=1.5,
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET", "HEAD"))
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = USER_AGENT
        _local.session = s
    return s


class NotFound(Exception):
    pass


def download(url: str, dest: Path, timeout: float = 300.0) -> Path:
    """Stream ``url`` to ``dest`` atomically. Raises NotFound on HTTP 404."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with session().get(url, stream=True, timeout=timeout) as r:
        if r.status_code == 404:
            raise NotFound(url)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
            os.replace(tmp, dest)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return dest


def get_json(url: str, params: dict | None = None, timeout: float = 120.0):
    r = session().get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_text(url: str, params: dict | None = None, timeout: float = 120.0) -> str:
    r = session().get(url, params=params, timeout=timeout)
    if r.status_code == 404:
        raise NotFound(url)
    r.raise_for_status()
    return r.text
