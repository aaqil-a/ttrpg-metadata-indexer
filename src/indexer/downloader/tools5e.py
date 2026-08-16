from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set
from collections import Counter
import re

from tqdm import tqdm
from indexer.config import load_data_sources
from indexer.downloader.base import Downloader, JsonDiskCache
from indexer.types import MAX_DESC_LENGTH, Adventure


class Tools5eDownloader(Downloader):
    def __init__(self, config_path: str | Path | None = None):
        config = load_data_sources(config_path)
        source_config = config.get("5etools", {})
        self.data_path = source_config.get("data_path", "")
        self.description_types = source_config.get("description_types", [])
        self.session = self._session()
        self.cache = JsonDiskCache("build/5etools_cache", expire_after=source_config.get("cache_expiry_seconds", 604800))


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

            if entry_type not in self.description_types:
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

        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [name for name, _ in top]


    def _fetch_adventure_data(self, id: str) -> List[Dict[str, Any]]:
        url = self.data_path + f"/adventure/adventure-{id.lower()}.json"
        return self._get_json(self.session, url, cache=self.cache)["data"]


    def _parse_adventure(self, data: Dict[str, Any]) -> Adventure:
        other_args = {
            "source": data["source"],
            "published": data["published"],
            "storyline": data["storyline"],
        }

        adventure_data = self._fetch_adventure_data(data["id"])
        description = self._extract_description(adventure_data, max_chars=MAX_DESC_LENGTH)
        creatures = self._extract_creatures(adventure_data)

        return Adventure(
            slug=self.get_slug(data["name"]),
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


    def fetch_adventures(self, existing_slugs: Set[str] = set(), max_workers: int = 20) -> List[Adventure]:
        data = self._get_json(self.session, self.data_path + "/adventures.json", cache=self.cache)
        adventures = data["adventure"]

        all_adventures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._parse_adventure, adv)
                for adv in adventures if self.get_slug(adv["name"]) not in existing_slugs
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
