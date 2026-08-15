from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from collections import Counter
import re

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
    def __init__(self, base_url: str = BASE_URL, max_creatures: int = 15):
        self.base_url = base_url
        self.max_creatures = max_creatures


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


    def _extract_creatures(self, value: Any) -> List[str]:
        counts: Counter[str] = Counter()

        pattern = re.compile(r"\{@creature\s+([^}]+)\}", re.IGNORECASE)

        def walk(node: Any) -> None:
            if isinstance(node, str):
                for m in pattern.finditer(node):
                    inner = m.group(1).strip()
                    name = inner.split("|", 1)[0].strip()
                    if name:
                        counts[name.lower()] += 1
                return

            if isinstance(node, list):
                for item in node:
                    walk(item)

                return

            if not isinstance(node, dict):
                return

            entries = node.get("entries")
            if entries is not None:
                walk(entries)
                return

            for v in node.values():
                walk(v)

        walk(value)

        if not counts:
            return []

        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:self.max_creatures]
        return [name for name, _ in top]


    def _fetch_adventure_data(self, id: str) -> List[Dict[str, Any]]:
        session = self._session()
        r = session.get(self.base_url + f"/adventure/adventure-{id.lower()}.json", timeout=10)
        r.raise_for_status()
        return r.json()["data"]


    def _parse_adventure(self, data: Dict[str, Any]) -> Adventure:
        other_args = {
            "source": data["source"],
            "published": data["published"],
            "storyline": data["storyline"],
        }

        adventure_data = self._fetch_adventure_data(data["id"])
        description = self._extract_description(adventure_data, max_chars=1500)
        creatures = self._extract_creatures(adventure_data)

        return Adventure(
            slug=self._get_slug(data["name"]),
            title=data["name"],
            description=description,
            authors=[data.get("author", "")],
            environments=[],
            start_level=data["level"]["start"],
            end_level=data["level"]["end"],
            downloaded_from="5etools",
            creatures=creatures,
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
