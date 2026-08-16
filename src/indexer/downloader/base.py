import hashlib
import json
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from indexer.types import Adventure

DEFAULT_USER_AGENT = (
    "ttrpg-metadata-indexer/1.0 "
    "(+https://github.com/aaqil-a/ttrpg-metadata-indexer)"
)


class JsonDiskCache:
    def __init__(self, cache_dir: Union[str, Path], expire_after: Optional[int] = 604800):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.expire_after = expire_after

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.dir / f"{digest}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        try:
            if self.expire_after is not None and (time.time() - path.stat().st_mtime) > self.expire_after:
                return None
            with path.open(encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(value, fh)
        os.replace(tmp, path)

    def clear(self) -> int:
        """Delete all cached entries (and any stray temp files). Returns count removed."""
        removed = 0
        for pattern in ("*.json", "*.tmp"):
            for path in self.dir.glob(pattern):
                try:
                    path.unlink()
                    removed += pattern == "*.json"
                except OSError:
                    pass
        return removed


class _RateLimitedSession(requests.Session):
    def __init__(self, min_interval: float = 0.0) -> None:
        super().__init__()
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def request(self, *args, **kwargs):
        if self._min_interval > 0:
            with self._lock:
                slot = max(time.monotonic(), self._next_allowed)
                self._next_allowed = slot + self._min_interval
            delay = slot - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return super().request(*args, **kwargs)


class Downloader(ABC):
    @abstractmethod
    def fetch_adventures(self, existing_slugs: Set[str] = set(), max_workers: int = 20) -> List[Adventure]:
        pass

    def get_slug(self, title: str) -> str:
        return re.sub(r'[<>:"/\\|?*]', '-', title.lower())

    def reset_cache(self) -> int:
        cache = getattr(self, "cache", None)
        if isinstance(cache, JsonDiskCache):
            return cache.clear()
        return 0

    def _get_json(
        self,
        session: requests.Session,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        cache: Optional[JsonDiskCache] = None,
        timeout: int = 10,
    ) -> Any:
        key = url if not params else url + "?" + urlencode(sorted(params.items()))
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                return hit

        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        if cache is not None and not (isinstance(data, dict) and "error" in data):
            cache.set(key, data)
        return data

    def _session(
        self,
        total_retries: int = 5,
        backoff_factor: float = 0.5,
        status_forcelist: Optional[List[int]] = None,
        min_interval: float = 0.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> requests.Session:
        if status_forcelist is None:
            status_forcelist = [429, 500, 502, 503, 504]

        session = _RateLimitedSession(min_interval=min_interval)
        session.headers.update({"User-Agent": user_agent})
        retry_strategy = Retry(
            total=total_retries,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
