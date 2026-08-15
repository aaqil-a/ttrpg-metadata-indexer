from typing import Any, Dict, List, Tuple

import math

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from indexer.downloader.base import Downloader
from indexer.types import Adventure
import logging

BASE_URL = "https://www.adventurelookup.com/api"
SEED = 42

class AdventureLookupDownloader(Downloader):
    def __init__(self, base_url: str = BASE_URL, seed: int = SEED):
        self.base_url = base_url
        self.seed = seed


    def _fetch_total_and_page_size(self) -> Tuple[int, int]:
        session = self._session()
        r = session.get(f"{self.base_url}/adventures?page=1&seed={self.seed}", timeout=10)
        r.raise_for_status()
        body = r.json()

        total = body.get("total_count", 0)
        if total <= 0:
            raise RuntimeError("Total count returned empty.")

        page_size = len(body.get("adventures", []))
        if page_size <= 0:
            raise RuntimeError("Page size is empty.")

        return total, page_size


    def _parse_adventure(self, data: Dict[str, Any]) -> Adventure:
        known_fields = {
            "slug",
            "title",
            "description",
            "authors",
            "environments",
            "min_starting_level",
            "max_starting_level",
            "common_monsters",
            "boss_monsters"
        }
                
        other_args = {
            key: value
            for key, value in data.items()
            if key not in known_fields
        }

        creatures = data.get("boss_monsters", []) + data.get("common_monsters", [])
        creatures = [creature.lower() for creature in creatures]

        return Adventure(
            slug=data["slug"],
            title=data["title"],
            description=data["description"],
            authors=data["authors"],
            environments=data["environments"],
            start_level=data["min_starting_level"],
            end_level=data["max_starting_level"],
            creatures=creatures,
            downloaded_from="AdventureLookup",
            other_args=other_args,
        )


    def _fetch_page(self, page: int = 1) -> List[Adventure]:
        session = self._session()
        r = session.get(f"{self.base_url}/adventures?page={page}&seed={self.seed}", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [self._parse_adventure(adv) for adv in data.get("adventures", [])]


    def fetch_adventures(self, max_workers: int = 20) -> List[Adventure]:
        total, page_size = self._fetch_total_and_page_size()
        page_num = math.ceil(total / page_size)
        all_adventures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._fetch_page, page)
                for page in range(1, page_num + 1)
            ]

            with tqdm(
                total=page_num,
                desc="Fetching from AdventureLookup",
                unit="page"
            ) as progress:
                for future in as_completed(futures):
                    try:
                        adventures = future.result()
                    except Exception as exc:
                        logging.error("Page fetch raised an exception: %s", exc)
                        adventures = []

                    if adventures:
                        all_adventures += adventures

                    progress.update(1)

        return all_adventures
