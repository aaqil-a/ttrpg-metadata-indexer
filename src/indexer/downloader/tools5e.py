from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import requests
from tqdm import tqdm
from indexer.downloader.base import Downloader
from indexer.types import Adventure

BASE_URL = "https://raw.githubusercontent.com/5etools-mirror-3/5etools-src/refs/heads/main/data"

DESCRIPTION_TYPES = {
    "section",
    "entries",
    "inset",
    "insetReadaloud",
    "quote",
}

class Tools5eDownloader(Downloader):
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url


    def _get_slug(self, title: str) -> str:
        return title.replace(" ", "-").lower()


    def _extract_description(self, value: Any, max_chars: int) -> str:
        parts: list[str] = []
        char_count = 0

        def add_text(text: str) -> bool:
            nonlocal char_count

            if char_count >= max_chars:
                return False

            text = text.strip()
            if not text:
                return True

            remaining = max_chars - char_count

            if len(text) > remaining:
                text = text[:remaining]

            parts.append(text)
            char_count += len(text)

            return char_count < max_chars

        def walk(node: Any) -> bool:
            if char_count >= max_chars:
                return False

            if isinstance(node, str):
                return add_text(node)

            if isinstance(node, list):
                for item in node:
                    if not walk(item):
                        return False

                return True

            if not isinstance(node, dict):
                return True

            entry_type = node.get("type")

            if entry_type not in DESCRIPTION_TYPES:
                return True

            entries = node.get("entries")

            if entries is not None:
                return walk(entries)

            return True

        walk(value)

        return "\n\n".join(parts)


    def _fetch_description(self, id: str) -> str:
        session = self._session()
        r = session.get(self.base_url + f"/adventure/adventure-{id.lower()}.json", timeout=10)
        r.raise_for_status()
        data = r.json()["data"]
        return self._extract_description(data, max_chars=1500)


    def _parse_adventure(self, data: Dict[str, Any]) -> Adventure:
        other_args = {
            "source": data["source"],
            "published": data["published"],
            "storyline": data["storyline"],
        }

        return Adventure(
            slug=self._get_slug(data["name"]),
            title=data["name"],
            description=self._fetch_description(data["id"]),
            authors=[data.get("author", "")],
            environments=[],
            start_level=data["level"]["start"],
            end_level=data["level"]["end"],
            downloaded_from="5etools",
            other_args=other_args,
        )   


    def fetch_adventures(self, max_workers: int = 20) -> List[Adventure]:
        session = self._session()
        r = session.get(self.base_url + "/adventures.json", timeout=10)
        r.raise_for_status()
        data = r.json()
        adventures = data["adventure"]

        all_adventures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._parse_adventure, adv)
                for adv in adventures
            ]

            with tqdm(
                total=len(adventures),
                desc="Parsing adventures from 5etools",
            ) as progress:
                for future in as_completed(futures):
                    adventure = future.result()
                    all_adventures.append(adventure)

                    progress.update(1)

        return all_adventures
