from __future__ import annotations

import json
import sqlite3

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class AdventureRecord:
    slug: str
    title: str
    description: str
    creatures: List[str]
    environments: List[str] = field(default_factory=list)


class BaseClassifier(ABC):
    @staticmethod
    def _parse_json_list(value: Optional[str]) -> list:
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return parsed if isinstance(parsed, list) else []

    @classmethod
    def _fetch_records(cls, conn: sqlite3.Connection) -> List[AdventureRecord]:
        cur = conn.cursor()
        cur.execute("SELECT slug, title, description, creatures, environments FROM adventures")
        records = []
        for slug, title, description, creatures_json, env_json in cur.fetchall():
            records.append(
                AdventureRecord(
                    slug=slug,
                    title=title or "",
                    description=description or "",
                    creatures=cls._parse_json_list(creatures_json),
                    environments=cls._parse_json_list(env_json),
                )
            )
        return records

    @staticmethod
    def _taxonomy_from(records: List[AdventureRecord]) -> List[str]:
        """The allowed label set: every distinct environment already in the DB."""
        labels = set()
        for record in records:
            labels.update(record.environments)
        return sorted(labels)

    @staticmethod
    def _build_text(record: AdventureRecord) -> str:
        """Combine the signals we classify on into a single document."""
        parts = [record.title or ""]
        if record.description:
            parts.append(record.description)
        if record.creatures:
            parts.append(" ".join(record.creatures))
        return "\n".join(part for part in parts if part)

    @abstractmethod
    def predict_records(
        self, records: List[AdventureRecord], taxonomy: List[str]
    ) -> List[List[str]]:
        """Return a predicted environment list for each input record."""

    def infer(
        self,
        db_path: Path,
        update_db: bool = True,
        output_path: Optional[Path] = None,
    ) -> None:
        conn = sqlite3.connect(str(db_path))
        try:
            records = self._fetch_records(conn)
            taxonomy = self._taxonomy_from(records)
            unlabeled = [record for record in records if not record.environments]

            if not unlabeled:
                print("No unlabeled adventures found to infer.")
                return

            predictions_list = self.predict_records(unlabeled, taxonomy)

            cur = conn.cursor()
            results = []
            updated = 0
            for record, predictions in zip(unlabeled, predictions_list):
                predictions = list(predictions)
                results.append({
                    "data": {
                        "slug": record.slug,
                        "title": record.title,
                        "description": record.description,
                    },
                    "inferred_environments": predictions,
                })
                if update_db and predictions:
                    cur.execute(
                        "UPDATE adventures SET environments = ? WHERE slug = ?",
                        (json.dumps(predictions), record.slug),
                    )
                    updated += 1

            if update_db:
                conn.commit()
                print(f"Database updated with inferred environments for {updated} adventures.")

            if output_path:
                with open(output_path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, indent=2, ensure_ascii=False)
                print(f"Wrote {len(results)} inference results to {output_path}")
        finally:
            conn.close()
