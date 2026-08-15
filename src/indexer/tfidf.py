from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib


def _fetch_all(conn: sqlite3.Connection) -> List[Tuple[str, str, str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT slug, title, description, environments FROM adventures")
    return cur.fetchall()


def _texts_and_labels(rows: List[Tuple[str, str, str, str]]):
    ids = []
    titles = []
    texts = []
    labels = []

    for slug, title, description, env_json in rows:
        envs = []
        try:
            envs = json.loads(env_json) if env_json else []
        except Exception:
            envs = []

        text = (title or "")
        if description:
            text = text + "\n" + description

        ids.append(slug)
        titles.append(title or "")
        texts.append(text)
        labels.append(envs)

    return ids, titles, texts, labels


def train(db_path: Path, model_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = _fetch_all(conn)
        _, _, texts, labels = _texts_and_labels(rows)

        train_texts = []
        train_labels = []
        for t, l in zip(texts, labels):
            if l:
                train_texts.append(t)
                train_labels.append(l)

        if not train_texts:
            raise RuntimeError("No labeled examples found in the database to train on.")

        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
        X = vectorizer.fit_transform(train_texts)

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(train_labels)

        clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
        clf.fit(X, Y)

        joblib.dump({"vectorizer": vectorizer, "clf": clf, "mlb": mlb}, str(model_path))
        print(f"Trained model saved to {model_path}")
    finally:
        conn.close()


def infer(
    db_path: Path,
    model_path: Path,
    update_db: bool = True,
    threshold: float = 0.3,
    output_path: Optional[Path] = None,
) -> None:
    model = joblib.load(str(model_path))
    vectorizer = model["vectorizer"]
    clf = model["clf"]
    mlb = model["mlb"]

    conn = sqlite3.connect(str(db_path))
    try:
        rows = _fetch_all(conn)
        ids, titles, texts, labels = _texts_and_labels(rows)

        to_infer = []
        to_infer_ids = []
        to_infer_titles = []
        to_infer_descriptions = []
        for slug, title, text, label in zip(ids, titles, texts, labels):
            if not label:
                to_infer_ids.append(slug)
                to_infer.append(text)
                to_infer_titles.append(title)
                desc = ""
                if "\n" in text:
                    desc = text.split("\n", 1)[1]
                to_infer_descriptions.append(desc)

        if not to_infer:
            print("No unlabeled adventures found to infer.")
            return

        X = vectorizer.transform(to_infer)
        probabilities = clf.predict_proba(X)

        cur = conn.cursor()

        results = []
        for slug, title, desc, probs in zip(
            to_infer_ids, to_infer_titles, to_infer_descriptions, probabilities
        ):
            ranked = sorted(zip(mlb.classes_, probs), key=lambda x: x[1], reverse=True)

            predictions = [environment for environment, probability in ranked if probability >= threshold]

            if not predictions and ranked:
                predictions = [ranked[0][0]]

            results.append({
                "data": {"slug": slug, "title": title, "description": desc},
                "inferred_environments": list(predictions),
            })

            if update_db:
                env_json = json.dumps(predictions)
                cur.execute("UPDATE adventures SET environments = ? WHERE slug = ?", (env_json, slug))

        if update_db:
            conn.commit()
            print("Database updated with inferred environments.")

        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            print(f"Wrote {len(results)} inference results to {output_path}")
    finally:
        conn.close()


def _cli():
    from indexer import DB_PATH, MODEL_PATH

    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--db", type=Path, default=DB_PATH)
    t.add_argument("--model", type=Path, default=MODEL_PATH)

    i = sub.add_parser("infer")
    i.add_argument("--db", type=Path, default=DB_PATH)
    i.add_argument("--model", type=Path, default=MODEL_PATH)
    i.add_argument("--no-update", dest="update", action="store_false", help="Do not write predictions back to DB")
    i.add_argument("--output", type=Path, required=False, help="Path to write JSON array of inference results")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.db, args.model)
    elif args.cmd == "infer":
        infer(args.db, args.model, update_db=args.update, output_path=getattr(args, "output", None))
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
