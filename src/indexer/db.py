import json
import sqlite3
from pathlib import Path
from typing import List

from tqdm import tqdm

from .types import Adventure


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS adventures (
    slug TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    authors TEXT,
    environments TEXT,
    start_level INTEGER,
    end_level INTEGER,
    downloaded_from TEXT,
    other_args TEXT
);
"""


def init_db(db_path: str) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True) if db_file.parent != Path('.') else None
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(DB_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def store_adventures(adventures: List[Adventure], db_path: str) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        sql = (
            "INSERT INTO adventures (slug, title, description, authors, environments, start_level, end_level, downloaded_from, other_args)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(slug) DO UPDATE SET"
            " title=excluded.title, description=excluded.description, authors=excluded.authors,"
            " environments=excluded.environments, start_level=excluded.start_level,"
            " end_level=excluded.end_level, downloaded_from=excluded.downloaded_from, other_args=excluded.other_args"
        )

        for adv in tqdm(adventures, desc="Storing adventures to database"):
            authors_json = json.dumps(adv.authors or [])
            env_json = json.dumps(adv.environments or [])
            other_json = json.dumps(adv.other_args or {})
            cur.execute(
                sql,
                (
                    adv.slug,
                    adv.title,
                    adv.description,
                    authors_json,
                    env_json,
                    adv.start_level,
                    adv.end_level,
                    adv.downloaded_from,
                    other_json,
                ),
            )

        conn.commit()
    finally:
        conn.close()


def get_all_adventures(db_path: str) -> List[Adventure]:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT slug, title, description, authors, environments, start_level, end_level, downloaded_from, other_args FROM adventures"
        )
        rows = cur.fetchall()
        result: List[Adventure] = []
        for r in rows:
            (
                slug,
                title,
                description,
                authors_json,
                env_json,
                start_level,
                end_level,
                downloaded_from,
                other_json,
            ) = r
            authors = json.loads(authors_json) if authors_json else []
            environments = json.loads(env_json) if env_json else []
            other_args = json.loads(other_json) if other_json else {}
            result.append(
                Adventure(
                    slug=slug,
                    title=title,
                    description=description,
                    authors=authors,
                    environments=environments,
                    start_level=start_level,
                    end_level=end_level,
                    downloaded_from=downloaded_from,
                    other_args=other_args,
                )
            )
        return result
    finally:
        conn.close()
