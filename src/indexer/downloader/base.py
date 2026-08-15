from abc import ABC, abstractmethod
from typing import List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from indexer.types import Adventure


class Downloader(ABC):
    @abstractmethod
    def fetch_adventures(self) -> List[Adventure]:
        pass

    def _session(
        self,
        total_retries: int = 5,
        backoff_factor: float = 0.5,
        status_forcelist: Optional[List[int]] = None,
    ) -> requests.Session:
        if status_forcelist is None:
            status_forcelist = [429, 500, 502, 503, 504]

        session = requests.Session()
        retry_strategy = Retry(
            total=total_retries,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session
