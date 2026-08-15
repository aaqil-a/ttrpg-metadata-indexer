import argparse

import os
from pathlib import Path
from typing import List
from indexer.db import get_all_adventures, store_adventures
from indexer.downloader.adventure_lookup import AdventureLookupDownloader
from indexer.downloader.base import Downloader
from indexer.downloader.tools5e import Tools5eDownloader
from indexer.index import create_adventure_files
import indexer.tfidf as tfidf_module
from indexer.tfidf import infer, train

DB_PATH = Path(__file__).resolve().parents[2] / "adventures.db"
MODEL_PATH = Path(__file__).resolve().parents[2] / "model.joblib"
INDEX_DIR = Path(__file__).resolve().parents[2] / "index"

def ingest_adventures(db_path: Path):
    if db_path.is_file():
        os.remove(db_path)

    downloaders: List[Downloader] = [
        AdventureLookupDownloader(),
        Tools5eDownloader()
    ]

    for downloader in downloaders:
        adventures = downloader.fetch_adventures()
        store_adventures(adventures, db_path)
        print(f"Ingested {len(adventures)} adventures from {downloader.__class__.__name__} into {db_path}.")


def infer_labels(db_path: Path, model_path: Path):
    train(db_path, model_path)
    infer(db_path, model_path, True)


def index_adventures(on: List[str], db_path: Path, dir: Path, max_entries: int = 50, hierarchical: bool = False):
    adventures = get_all_adventures(db_path)
    create_adventure_files(adventures, on, dir, max_entries=max_entries, hierarchical=hierarchical)

def main() -> None:
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd")

    ingest = sub.add_parser("ingest", help="Download and store adventures into a sqlite database")
    ingest.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to write (will be overwritten if exists)")

    infer = sub.add_parser("infer", help="Train or run the TF-IDF inference model to infer missing labels")
    infer.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read/write inferred labels")
    infer.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to save/load the trained inference model")

    index = sub.add_parser("index", help="Generate Markdown index files from the database")
    index.add_argument("--on", nargs="+", default=["environments", "start_level", "end_level", "downloaded_from"],
                       help="List of attributes to create indexes on (e.g. environments start_level)")
    index.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read adventures from")
    index.add_argument("--dir", type=Path, default=INDEX_DIR, help="Directory to write generated index files into")
    index.add_argument("--max-entries", type=int, default=50,
                       help="Maximum entries per generated index file (default: 50). Set <=0 to disable splitting.")
    index.add_argument("--no-hierarchical", dest="hierarchical", action="store_false",
                       help="Disable hierarchical directory grouping for index files (default: enabled)")

    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest_adventures(args.db)
    elif args.cmd == "infer":
        infer_labels(args.db, args.model)
    elif args.cmd == "index":
        index_adventures(args.on, args.db, args.dir, max_entries=args.max_entries, hierarchical=bool(getattr(args, 'hierarchical', True)))
    else:
        parser.print_help()


def ingest_cli() -> None:
    parser = argparse.ArgumentParser(prog="ingest")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to write (will be overwritten if exists)")
    args = parser.parse_args()
    ingest_adventures(args.db)


def infer_cli() -> None:
    parser = argparse.ArgumentParser(prog="infer")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read/write inferred labels")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to save/load the trained inference model")
    args = parser.parse_args()
    infer_labels(args.db, args.model)


def index_cli() -> None:
    parser = argparse.ArgumentParser(prog="index")
    parser.add_argument("--on", nargs="+", default=["environments", "downloaded_from"])
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read adventures from")
    parser.add_argument("--dir", type=Path, default=INDEX_DIR, help="Directory to write generated index files into")
    parser.add_argument("--max-entries", type=int, default=50,
                        help="Maximum entries per generated index file (default: 50). Set <=0 to disable splitting.")
    parser.add_argument("--no-hierarchical", dest="hierarchical", action="store_false",
                        help="Disable hierarchical directory grouping for index files (default: enabled)")
    args = parser.parse_args()
    index_adventures(args.on, args.db, args.dir, max_entries=args.max_entries, hierarchical=bool(getattr(args, 'hierarchical', True)))


def tfidf_cli() -> None:
    tfidf_module._cli()

