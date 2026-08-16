from typing import Any, Dict, List, Set, Tuple

import math

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from indexer.config import load_data_sources
from indexer.downloader.base import Downloader, JsonDiskCache
from indexer.types import Adventure
import logging

class AdventureLookupDownloader(Downloader):
    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        config = load_data_sources(config_path)
        source_config = config.get("adventurelookup", {})
        self.api_url = source_config.get("api_url", "")
        self.seed = source_config.get("seed", 0)
        self.session = self._session()
        self.cache = JsonDiskCache("build/adventurelookup_cache", expire_after=source_config.get("cache_expiry_seconds", 86400))


    def _fetch_total_and_page_size(self) -> Tuple[int, int]:
        body = self._get_json(
            self.session, f"{self.api_url}/adventures",
            params={"page": 1, "seed": self.seed}, cache=self.cache,
        )

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
        data = self._get_json(
            self.session, f"{self.api_url}/adventures",
            params={"page": page, "seed": self.seed}, cache=self.cache,
        )
        return [self._parse_adventure(adv) for adv in data.get("adventures", [])]


    def fetch_adventures(self, existing_slugs: Set[str] = set(), max_workers: int = 20) -> List[Adventure]:
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
