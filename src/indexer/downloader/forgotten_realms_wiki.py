from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pathlib import Path

from tqdm import tqdm
from indexer.config import load_data_sources
from indexer.downloader.base import Downloader, JsonDiskCache
from indexer.types import MAX_DESC_LENGTH, Adventure

INFOBOX_NAMES = ("Book", "Adventurers league", "Adventure")

def extract_infobox(wikitext: str) -> Tuple[str, Dict[str, str], str]:
    name_alt = "|".join(re.escape(n) for n in INFOBOX_NAMES)
    m = re.search(r"\{\{\s*(" + name_alt + r")\b", wikitext, re.IGNORECASE)
    if not m:
        return ("", {}, wikitext)

    template_name = m.group(1)
    i, n, depth = m.end(), len(wikitext), 1
    while i < n and depth > 0:                       
        two = wikitext[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; i += 2
        elif two in ("}}", "]]"):
            depth -= 1; i += 2
        else:
            i += 1

    body = wikitext[m.end():i - 2]                  
    trailing = wikitext[i:].strip()                 
    return template_name, _split_fields(body), trailing


def _split_fields(body: str) -> Dict[str, str]:
    depth, buf, chunks, i, n = 0, [], [], 0, len(body)
    while i < n:
        two = body[i:i + 2]
        if two in ("{{", "[["):
            depth += 1; buf.append(two); i += 2
        elif two in ("}}", "]]"):
            depth -= 1; buf.append(two); i += 2
        elif body[i] == "|" and depth == 0:
            chunks.append("".join(buf)); buf = []; i += 1
        else:
            buf.append(body[i]); i += 1
    chunks.append("".join(buf))

    fields = {}
    for chunk in chunks:
        if "=" not in chunk:
            continue                                
        key, _, value = chunk.partition("=")
        key = key.strip().lower()
        if key:
            fields[key] = value.strip()
    return fields


def clean(v: str) -> str:
    v = re.sub(r"<ref[^>]*>.*?</ref>", "", v, flags=re.S)   
    v = re.sub(r"<ref[^>]*/>", "", v)                       
    v = re.sub(r"<!--.*?-->", "", v, flags=re.S)           
    v = re.sub(r"\{\{[^{}]*\}\}", "", v)                    
    v = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", v) 
    v = re.sub(r"<[^>]+>", " ", v)                          
    v = v.replace("'''", "").replace("''", "")              
    v = html.unescape(v)                                    
    v = re.sub(r"\s+", " ", v)                          
    return v.strip()


def _match_template(text: str, start: int) -> int:
    i, depth, n = start + 2, 1, len(text)
    while i < n and depth > 0:
        two = text[i:i + 2]
        if two in ("{{", "[["):   depth += 1; i += 2
        elif two in ("}}", "]]"): depth -= 1; i += 2
        else:                     i += 1
    return i


def extract_quote(text: str) -> str:
    m = re.search(r"\{\{\s*quote\b", text, re.IGNORECASE)
    if not m:
        return ""
    body = text[m.end():_match_template(text, m.start()) - 2]
    for chunk in _split_fields(body):
        if chunk.strip() and not re.match(r"\s*[A-Za-z_][\w \-]*=", chunk):
            return chunk.strip()
    return ""


def build_description(trailing: str, max_len: int = MAX_DESC_LENGTH) -> str:
    quote = extract_quote(trailing)
    prose = trailing
    m = re.search(r"\{\{\s*quote\b", trailing, re.IGNORECASE)
    if m:                                      
        prose = trailing[:m.start()] + trailing[_match_template(trailing, m.start()):]
    return " ".join(p for p in (clean(prose), clean(quote)) if p).strip()[:max_len]


def extract_levels_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"levels?\s+(\d{1,2})\s*(?:[-–—]|to|through|and)\s*(\d{1,2})", text, re.I)
    if m:
        return int(m[1]), int(m[2])
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s*[-–—]?\s*(?:to|through|[-–—])\s*"
                  r"(\d{1,2})(?:st|nd|rd|th)?[-\s]*level", text, re.I)
    if m:
        return int(m[1]), int(m[2])
    m = re.search(r"level\s+(\d{1,2})\b|(\d{1,2})(?:st|nd|rd|th)?[-\s]*level", text, re.I)
    if m:
        lvl = int(m[1] or m[2])
        return lvl, lvl
    return None, None


class ForgottenRealmsWikiDownloader(Downloader):
    def __init__(
        self,
        config_path: str | Path | None = None,
    ):
        config = load_data_sources(config_path)
        source_config = config.get("forgottenrealmswiki", {})
        self.api_url = source_config.get("api_url", "")
        self.session = super()._session(min_interval=1)
        self.cache = JsonDiskCache("build/frwiki_cache", expire_after=source_config.get("cache_expiry_seconds", 604800))

    def _fetch_adventure_titles(self, cmcontinue: str = "") -> List[str]:
        params: Dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": "Category:Adventures",
            "cmlimit": 500,
            "cmtype": "page",
            "ns": 0,
            "format": "json",
        }

        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        data = self._get_json(self.session, self.api_url, params=params, cache=self.cache)

        titles: List[str] = [cm["title"] for cm in data["query"]["categorymembers"]]

        if "continue" in data and "cmcontinue" in data["continue"]:
            return titles + self._fetch_adventure_titles(data["continue"]["cmcontinue"])

        return titles

    def _get_adventure_sections(self, title: str) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "format": "json",
        }

        data = self._get_json(self.session, self.api_url, params=params, cache=self.cache)
        return data["parse"]["sections"]


    def _fetch_wikitext(self, title: str, section_idx: int) -> str:
        params: Dict[str, Any] = {
            "action": "parse",
            "page": title,
            "prop": "wikitext",
            "section": section_idx,
            "format": "json",
        }

        data = self._get_json(self.session, self.api_url, params=params, cache=self.cache)
        return data["parse"]["wikitext"]["*"]


    def _fill_adventure_metadata(self, title: str, adv: Adventure) -> Adventure: 
        wikitext = self._fetch_wikitext(title, 0)

        _, parsed, trailing = extract_infobox(wikitext)
        if parsed is None:
            logging.warning(f"Failed to fetch metadata for {title}")
        else:
            authors_raw = parsed.get("author") or parsed.get("design", "")
            authors = [a for a in re.split(r",|<br\s*/?>|\band\b", clean(authors_raw)) if a.strip()]
            authors += [clean(e) for e in re.split(r",|<br\s*/?>", parsed.get("editor", "")) if e.strip()]
            adv.authors = authors

            levels = re.findall(r"\d+", parsed.get("levels", ""))
            adv.start_level = int(levels[0]) if levels else None
            adv.end_level = int(levels[-1]) if levels else None

            adv.other_args = {
            k: clean(parsed[k]) for k in (
                "code", "publisher", "game_edition", "setting", "series", "game_edition", "released"
                ) if k in parsed and parsed[k]
            }

        adv.description = build_description(trailing)
        if not (adv.start_level or adv.end_level):
            adv.start_level, adv.end_level = extract_levels_from_text(adv.description)

        return adv

    def _fill_adventure_description(self, title: str, section_idx: int, adv: Adventure) -> Adventure: 
        if len(adv.description) < MAX_DESC_LENGTH:
            additional = self._fetch_wikitext(title, section_idx)
            adv.description = f"{adv.description}\n{clean(additional)}"
            adv.description = adv.description[:MAX_DESC_LENGTH]

        return adv

    def _fill_adventure_creatures(self, title: str, section_idx: int, adv: Adventure) -> Adventure: 
        wikitext = self._fetch_wikitext(title, section_idx)

        seen, out = set(), []
        for name in re.findall(r"\[\[([^\]|#]+)", wikitext):
            c = name.strip().lower()                   
            if c and c not in seen:
                seen.add(c)
                out.append(c)

        adv.creatures = out

        return adv

    def _fetch_adventure_data(self, title: str) -> Adventure:
        adv = Adventure(
            slug=self.get_slug(title),
            title=title,
            description="",
            authors=[],
            environments=[],
            start_level=None,
            end_level=None,
            creatures=[],
            downloaded_from="Forgotten Realms Wiki",
            other_args={}
        )

        sections = self._get_adventure_sections(title)

        desc_idx = -1
        creature_idx = -1
        for section in sections:
            line = section["line"].strip().lower()
            if desc_idx == -1 and line in ("description", "summary", "synopsis"):
                desc_idx = section["index"]
            if creature_idx == -1 and line == "creatures":
                creature_idx = section["index"]

        adv = self._fill_adventure_metadata(title, adv)

        if desc_idx != -1:
            adv = self._fill_adventure_description(title, desc_idx, adv)

        if creature_idx != -1:
            adv = self._fill_adventure_creatures(title, creature_idx, adv)

        return adv

    def fetch_adventures(self, existing_slugs: Set[str] = set(), max_workers: int = 12) -> List[Adventure]:
        titles = self._fetch_adventure_titles()
        all_adventures = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._fetch_adventure_data, title)
                for title in titles if self.get_slug(title) not in existing_slugs
            ]

            with tqdm(
                total=len(titles),
                desc="Fetching from Forgotten Realms Wiki",
            ) as progress:
                for future in as_completed(futures):
                    try:
                        adventure = future.result()
                    except Exception as exc:
                        logging.error("Page fetch raised an exception: %s", exc)
                        adventure = None

                    if adventure:
                        all_adventures.append(adventure)

                    progress.update(1)
        
        return all_adventures
