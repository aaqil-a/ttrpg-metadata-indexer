from __future__ import annotations

import argparse
import json
import sqlite3

from pathlib import Path
from typing import Any, List, Tuple, Optional

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import joblib


def _fetch_all(conn: sqlite3.Connection) -> List[Tuple[str, str, str, str, str]]:
    cur = conn.cursor()
    cur.execute("SELECT slug, title, description, creatures, environments FROM adventures")
    return cur.fetchall()


def _parse_json_list(value: Optional[str]) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_text(title: str, description: str, creatures: List[str]) -> str:
    parts = [title or ""]
    if description:
        parts.append(description)
    if creatures:
        parts.append(" ".join(creatures))
    return "\n".join(part for part in parts if part)


def _texts_and_labels(rows: List[Tuple[str, str, str, str, str]]):
    ids = []
    titles = []
    descriptions = []
    texts = []
    labels = []

    for slug, title, description, creatures_json, env_json in rows:
        envs = _parse_json_list(env_json)
        creatures = _parse_json_list(creatures_json)

        text = _build_text(title, description, creatures)
        if not text:
            continue

        ids.append(slug)
        titles.append(title or "")
        descriptions.append(description or "")
        texts.append(text)
        labels.append(envs)

    return ids, titles, descriptions, texts, labels


def _make_vectorizer() -> FeatureUnion:
    word = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words="english",
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        max_features=50_000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.8,
        sublinear_tf=True,
    )
    transformer_list: List[Tuple[str, Any]] = [("word", word), ("char", char)]
    return FeatureUnion(transformer_list)


def _make_classifier() -> OneVsRestClassifier:
    # class_weight="balanced" is essential here: labels range from ~1100 (Dungeon)
    # to ~22 (Shadowfell). Without it, rare classes are never predicted.
    return OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))


def _predict_multilabel(probabilities, mlb, threshold: float):
    predicted = []
    for values in probabilities:
        ranked = sorted(zip(mlb.classes_, values), key=lambda item: item[1], reverse=True)
        selected = [label for label, probability in ranked if probability >= threshold]
        if not selected and ranked:
            selected = [ranked[0][0]]
        predicted.append(selected)
    return predicted


def _tune_threshold(y_true, probabilities) -> Tuple[float, float]:
    """Pick the decision threshold that maximises micro-F1 on held-out data.

    A single fixed threshold is wrong once class_weight="balanced" inflates the
    predicted probabilities, so we sweep and select instead of guessing.
    """
    probabilities = np.asarray(probabilities)
    best_threshold, best_f1 = 0.4, -1.0
    for threshold in np.arange(0.1, 0.61, 0.05):
        predictions = (probabilities >= threshold).astype(int)
        f1 = f1_score(y_true, predictions, average="micro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(threshold)
    return best_threshold, best_f1


def _evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train(
    db_path: Path,
    model_path: Path,
    eval_split: float = 0,
    threshold: Optional[float] = None,
) -> dict:
    if not 0 <= eval_split < 1:
        raise ValueError("eval_split must be between 0 and 1 (exclusive of 1)")

    conn = sqlite3.connect(str(db_path))
    try:
        rows = _fetch_all(conn)
        _, _, _, texts, labels = _texts_and_labels(rows)

        labeled = [(text, label) for text, label in zip(texts, labels) if label]
        if not labeled:
            raise RuntimeError("No labeled examples found in the database to train on.")

        if eval_split > 0:
            train_rows, eval_rows = train_test_split(
                labeled, test_size=eval_split, random_state=42, shuffle=True
            )
        else:
            train_rows, eval_rows = labeled, []

        if not train_rows:
            raise RuntimeError("No training examples remained after the requested data split.")

        if threshold is None and len(train_rows) > 20:
            fit_rows, val_rows = train_test_split(
                train_rows, test_size=0.15, random_state=42, shuffle=True
            )
        else:
            fit_rows, val_rows = train_rows, []

        vectorizer = _make_vectorizer()
        X = vectorizer.fit_transform([text for text, _ in fit_rows])
        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform([label for _, label in fit_rows])
        clf = _make_classifier()
        clf.fit(X, Y)

        if threshold is not None:
            best_threshold = threshold
        elif val_rows:
            X_val = vectorizer.transform([text for text, _ in val_rows])
            probs_val = np.asarray(clf.predict_proba(X_val))
            Y_val = mlb.transform([label for _, label in val_rows])
            best_threshold, best_val_f1 = _tune_threshold(Y_val, probs_val)
            print(f"Selected threshold={best_threshold:.2f} (validation micro-F1={best_val_f1:.4f})")
        else:
            best_threshold = 0.4

        metrics = {}
        if eval_rows:
            X_eval = vectorizer.transform([text for text, _ in eval_rows])
            probs_eval = np.asarray(clf.predict_proba(X_eval))
            Y_eval = mlb.transform([label for _, label in eval_rows])
            Y_pred = (probs_eval >= best_threshold).astype(int)
            metrics = _evaluate_predictions(Y_eval, Y_pred)
            print(
                "Evaluation metrics: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"f1_micro={metrics['f1_micro']:.4f} "
                f"(P={metrics['precision_micro']:.4f}, R={metrics['recall_micro']:.4f}), "
                f"f1_macro={metrics['f1_macro']:.4f} "
                f"(P={metrics['precision_macro']:.4f}, R={metrics['recall_macro']:.4f})"
            )
            print("\nPer-class report:")
            print(
                classification_report(
                    Y_eval, Y_pred, target_names=list(mlb.classes_), zero_division=0
                )
            )

        final_vectorizer = _make_vectorizer()
        X_all = final_vectorizer.fit_transform([text for text, _ in labeled])
        final_mlb = MultiLabelBinarizer()
        Y_all = final_mlb.fit_transform([label for _, label in labeled])
        final_clf = _make_classifier()
        final_clf.fit(X_all, Y_all)

        model = {
            "vectorizer": final_vectorizer,
            "clf": final_clf,
            "mlb": final_mlb,
            "threshold": best_threshold,
        }
        joblib.dump(model, str(model_path))
        print(f"Trained model saved to {model_path} (threshold={best_threshold:.2f})")
        return metrics
    finally:
        conn.close()


def infer(
    db_path: Path,
    model_path: Path,
    update_db: bool = True,
    threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
) -> None:
    model = joblib.load(str(model_path))
    vectorizer = model["vectorizer"]
    clf = model["clf"]
    mlb = model["mlb"]
    active_threshold = threshold if threshold is not None else model.get("threshold", 0.4)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = _fetch_all(conn)
        ids, titles, descriptions, texts, labels = _texts_and_labels(rows)

        to_infer = []
        to_infer_ids = []
        to_infer_titles = []
        to_infer_descriptions = []
        for slug, title, desc, text, label in zip(ids, titles, descriptions, texts, labels):
            if not label:
                to_infer_ids.append(slug)
                to_infer.append(text)
                to_infer_titles.append(title)
                to_infer_descriptions.append(desc)

        if not to_infer:
            print("No unlabeled adventures found to infer.")
            return

        X = vectorizer.transform(to_infer)
        probabilities = clf.predict_proba(X)
        predictions_list = _predict_multilabel(probabilities, mlb, threshold=active_threshold)

        cur = conn.cursor()

        results = []
        for slug, title, desc, predictions in zip(
            to_infer_ids, to_infer_titles, to_infer_descriptions, predictions_list
        ):
            results.append({
                "data": {"slug": slug, "title": title, "description": desc},
                "inferred_environments": list(predictions),
            })

            if update_db:
                env_json = json.dumps(predictions)
                cur.execute("UPDATE adventures SET environments = ? WHERE slug = ?", (env_json, slug))

        if update_db:
            conn.commit()
            print(f"Database updated with inferred environments for {len(results)} adventures.")

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

    t = sub.add_parser("train", help="Train TF-IDF + classifier to infer environments from text")
    t.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database with labeled examples for training")
    t.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to write the trained model (joblib)")
    t.add_argument("--eval-split", type=float, default=0, help="Fraction of labeled data to hold out for evaluation (0 disables evaluation)")
    t.add_argument("--threshold", type=float, default=None, help="Fixed prediction threshold; if omitted it is auto-tuned on a validation split")

    i = sub.add_parser("infer", help="Run inference to predict missing environments using a trained model")
    i.add_argument("--db", type=Path, default=DB_PATH, help="Path to the sqlite database to read/write inferred labels")
    i.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the trained model file (joblib)")
    i.add_argument("--threshold", type=float, default=None, help="Override the model's stored prediction threshold")
    i.add_argument("--no-update", dest="update", action="store_false", help="Do not write predictions back to the database")
    i.add_argument("--output", type=Path, required=False, help="Path to write JSON array of inference results (optional)")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.db, args.model, eval_split=args.eval_split, threshold=args.threshold)
    elif args.cmd == "infer":
        infer(
            args.db,
            args.model,
            update_db=args.update,
            threshold=args.threshold,
            output_path=getattr(args, "output", None),
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
