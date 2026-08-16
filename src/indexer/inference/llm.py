from __future__ import annotations

import argparse
import json
import os
import sqlite3

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from indexer.config import load_inference_config
from indexer.inference.base import AdventureRecord, BaseClassifier

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT = 30

SYSTEM_PROMPT = (
    "You are a metadata classifier for tabletop RPG adventures. "
    "Given an adventure's title, description and notable creatures, decide which "
    "environments (settings) the adventure takes place in. "
    "Choose ONLY from the allowed environments provided by the user; never invent new labels. "
    "Select every label that clearly applies, and at least one. "
    'Respond with a compact JSON object of the form {"environments": ["Label", ...]}.'
)


def _load_llm_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    return load_inference_config(config_path).get("llm", {})


def _build_user_prompt(taxonomy: List[str], title: str, description: str, creatures: List[str]) -> str:
    parts = [
        "Allowed environments: " + ", ".join(taxonomy),
        "",
        f"Title: {title}",
    ]
    if description:
        parts.append(f"Description: {description}")
    if creatures:
        parts.append("Creatures: " + ", ".join(creatures))
    return "\n".join(parts)


def _extract_environments(content: str, taxonomy: List[str]) -> List[str]:
    """Parse the model response and keep only valid, canonical labels."""
    canonical = {label.lower(): label for label in taxonomy}

    raw: Any = None
    try:
        raw = json.loads(content)
    except Exception:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = content.find(opener), content.rfind(closer)
            if start != -1 and end > start:
                try:
                    raw = json.loads(content[start : end + 1])
                    break
                except Exception:
                    continue

    if isinstance(raw, dict):
        raw = raw.get("environments", [])
    if not isinstance(raw, list):
        return []

    result = []
    for item in raw:
        if not isinstance(item, str):
            continue
        label = canonical.get(item.strip().lower())
        if label and label not in result:
            result.append(label)
    return result


def _make_token_counter(model: str):
    """Return (count_fn, method). Uses tiktoken when installed for an accurate
    count, otherwise falls back to a ~4-chars-per-token heuristic.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(enc.encode(text))), "tiktoken"
    except Exception:
        return (lambda text: (len(text) + 3) // 4), "heuristic (~chars/4)"


_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_REQUEST = 3


def estimate_tokens(
    db_path: Path,
    config_path: Optional[Path] = None,
    output_tokens_per_row: int = 30,
) -> Dict[str, Any]:
    """Estimate token usage for a full LLM inference run without calling the API."""
    config = _load_llm_config(config_path)
    model = config.get("model", DEFAULT_MODEL)
    count, method = _make_token_counter(model)

    conn = sqlite3.connect(str(db_path))
    try:
        records = BaseClassifier._fetch_records(conn)
    finally:
        conn.close()

    taxonomy = BaseClassifier._taxonomy_from(records)
    unlabeled = [record for record in records if not record.environments]

    system_tokens = count(SYSTEM_PROMPT)
    input_tokens = 0
    for record in unlabeled:
        user_prompt = _build_user_prompt(taxonomy, record.title, record.description, record.creatures)
        input_tokens += system_tokens + count(user_prompt)
        input_tokens += 2 * _TOKENS_PER_MESSAGE + _TOKENS_PER_REQUEST

    rows = len(unlabeled)
    output_tokens = rows * output_tokens_per_row

    result: Dict[str, Any] = {
        "model": model,
        "method": method,
        "rows": rows,
        "input_tokens": input_tokens,
        "output_tokens_est": output_tokens,
        "total_tokens_est": input_tokens + output_tokens,
        "per_row_input": input_tokens // rows if rows else 0,
        "output_tokens_per_row": output_tokens_per_row,
    }

    price_in = config.get("price_per_1m_input")
    price_out = config.get("price_per_1m_output")
    if price_in is not None or price_out is not None:
        cost = input_tokens / 1_000_000 * float(price_in or 0)
        cost += output_tokens / 1_000_000 * float(price_out or 0)
        result["cost_usd_est"] = round(cost, 4)

    return result


def print_token_estimate(db_path: Path, config_path: Optional[Path] = None) -> Dict[str, Any]:
    est = estimate_tokens(db_path, config_path)
    if est["rows"] == 0:
        print("No unlabeled adventures found to infer.")
        return est
    print(f"Token estimate for LLM inference ({est['method']}, model={est['model']}):")
    print(f"  unlabeled adventures : {est['rows']}")
    print(f"  input tokens         : ~{est['input_tokens']:,} ({est['per_row_input']:,}/row)")
    print(f"  output tokens (est)  : ~{est['output_tokens_est']:,} (assuming ~{est['output_tokens_per_row']}/row)")
    print(f"  total tokens (est)   : ~{est['total_tokens_est']:,}")
    if "cost_usd_est" in est:
        print(f"  cost (est)           : ~${est['cost_usd_est']:.4f}")
    return est


class LLMClassifier(BaseClassifier):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.base_url = str(config.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
        self.model = config.get("model", DEFAULT_MODEL)
        self.temperature = config.get("temperature", 0)
        self.max_tokens = config.get("max_tokens", 256)
        self.timeout = config.get("timeout", DEFAULT_TIMEOUT)
        self.json_mode = config.get("json_mode", True)
        self.max_workers = config.get("max_workers", DEFAULT_MAX_WORKERS)
        self.extra_params = config.get("extra_params", {})
        api_key_env = config.get("api_key_env", DEFAULT_API_KEY_ENV)
        self.api_key = os.environ.get(api_key_env, "")

    def classify(self, taxonomy: List[str], title: str, description: str, creatures: List[str]) -> List[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(taxonomy, title, description, creatures)},
            ],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_params)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code == 400 and self.json_mode:
            payload.pop("response_format", None)
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        return _extract_environments(content, taxonomy)

    def _classify_record(self, taxonomy: List[str], record: AdventureRecord) -> List[str]:
        try:
            return self.classify(taxonomy, record.title, record.description, record.creatures)
        except Exception as exc:
            print(f"  ! inference failed for {record.slug}: {exc}")
            return []

    def predict_records(
        self, records: List[AdventureRecord], taxonomy: List[str]
    ) -> List[List[str]]:
        results: List[List[str]] = [[] for _ in records]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self._classify_record, taxonomy, record): idx
                for idx, record in enumerate(records)
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(records),
                desc="Inferring environments (LLM)",
            ):
                results[future_to_idx[future]] = future.result()
        return results


def infer(
    db_path: Path,
    update_db: bool = True,
    output_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> None:
    classifier = LLMClassifier(_load_llm_config(config_path))
    classifier.infer(db_path, update_db=update_db, output_path=output_path)


def _cli():
    from indexer import DB_PATH, CONFIG_PATH

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd")

    i = sub.add_parser("infer", help="Infer environments using an OpenAI-compatible LLM")
    i.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read/write inferred labels")
    i.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the config file (reads the inference.llm section)")
    i.add_argument("--no-update", dest="update", action="store_false", help="Do not write predictions back to the database")
    i.add_argument("--output", type=Path, required=False, help="Path to write JSON array of inference results (optional)")

    e = sub.add_parser("estimate", help="Estimate token usage without calling the API")
    e.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read unlabeled adventures from")
    e.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to the config file (reads the inference.llm section)")

    args = parser.parse_args()
    if args.cmd == "infer":
        infer(
            args.db,
            update_db=args.update,
            output_path=getattr(args, "output", None),
            config_path=args.config,
        )
    elif args.cmd == "estimate":
        print_token_estimate(args.db, args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
