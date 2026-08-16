from __future__ import annotations

import argparse
import sqlite3

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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

from indexer.config import load_inference_config
from indexer.inference.base import AdventureRecord, BaseClassifier

DEFAULT_MAX_FEATURES = 50_000
DEFAULT_MAX_ITER = 1000
DEFAULT_FALLBACK_THRESHOLD = 0.4


def _load_tfidf_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    return load_inference_config(config_path).get("tfidf", {})


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
    best_threshold, best_f1 = DEFAULT_FALLBACK_THRESHOLD, -1.0
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


class TFIDFClassifier(BaseClassifier):
    """TF-IDF + one-vs-rest logistic regression multilabel environment classifier."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.max_features = config.get("max_features", DEFAULT_MAX_FEATURES)
        self.max_iter = config.get("max_iter", DEFAULT_MAX_ITER)
        self.fallback_threshold = config.get("fallback_threshold", DEFAULT_FALLBACK_THRESHOLD)

        self.vectorizer: Optional[FeatureUnion] = None
        self.clf: Optional[OneVsRestClassifier] = None
        self.mlb: Optional[MultiLabelBinarizer] = None
        self.threshold: float = self.fallback_threshold

    def _make_vectorizer(self) -> FeatureUnion:
        """Word n-grams capture vocabulary; char n-grams capture proper nouns,
        misspellings and morphology in short titles. Using both beats either alone.
        """
        word = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words="english",
            sublinear_tf=True,
        )
        char = TfidfVectorizer(
            max_features=self.max_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True,
        )
        transformer_list: List[Tuple[str, Any]] = [("word", word), ("char", char)]
        return FeatureUnion(transformer_list)

    def _make_classifier(self) -> OneVsRestClassifier:
        return OneVsRestClassifier(
            LogisticRegression(max_iter=self.max_iter, class_weight="balanced")
        )

    def train(self, db_path: Path, eval_split: float = 0, threshold: Optional[float] = None) -> dict:
        if not 0 <= eval_split < 1:
            raise ValueError("eval_split must be between 0 and 1 (exclusive of 1)")

        conn = sqlite3.connect(str(db_path))
        try:
            records = self._fetch_records(conn)
            labeled = [
                (self._build_text(record), record.environments)
                for record in records
                if record.environments
            ]
            labeled = [(text, label) for text, label in labeled if text]
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

            vectorizer = self._make_vectorizer()
            X = vectorizer.fit_transform([text for text, _ in fit_rows])
            mlb = MultiLabelBinarizer()
            Y = mlb.fit_transform([label for _, label in fit_rows])
            clf = self._make_classifier()
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
                best_threshold = self.fallback_threshold

            metrics: dict = {}
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

            final_vectorizer = self._make_vectorizer()
            X_all = final_vectorizer.fit_transform([text for text, _ in labeled])
            final_mlb = MultiLabelBinarizer()
            Y_all = final_mlb.fit_transform([label for _, label in labeled])
            final_clf = self._make_classifier()
            final_clf.fit(X_all, Y_all)

            self.vectorizer = final_vectorizer
            self.clf = final_clf
            self.mlb = final_mlb
            self.threshold = best_threshold
            return metrics
        finally:
            conn.close()

    def save(self, model_path: Path) -> None:
        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "clf": self.clf,
                "mlb": self.mlb,
                "threshold": self.threshold,
            },
            str(model_path),
        )
        print(f"Trained model saved to {model_path} (threshold={self.threshold:.2f})")

    @classmethod
    def load(cls, model_path: Path, config: Optional[Dict[str, Any]] = None) -> "TFIDFClassifier":
        model = joblib.load(str(model_path))
        instance = cls(config)
        instance.vectorizer = model["vectorizer"]
        instance.clf = model["clf"]
        instance.mlb = model["mlb"]
        instance.threshold = model.get("threshold", instance.fallback_threshold)
        return instance

    def predict(self, texts: List[str], threshold: Optional[float] = None) -> List[List[str]]:
        if self.vectorizer is None or self.clf is None or self.mlb is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or load() first.")
        active_threshold = threshold if threshold is not None else self.threshold
        probabilities = self.clf.predict_proba(self.vectorizer.transform(texts))
        return _predict_multilabel(probabilities, self.mlb, threshold=active_threshold)

    def predict_records(
        self, records: List[AdventureRecord], taxonomy: List[str]
    ) -> List[List[str]]:
        return self.predict([self._build_text(record) for record in records])


def train(
    db_path: Path,
    model_path: Path,
    eval_split: float = 0,
    threshold: Optional[float] = None,
    config_path: Optional[Path] = None,
) -> dict:
    classifier = TFIDFClassifier(_load_tfidf_config(config_path))
    metrics = classifier.train(db_path, eval_split=eval_split, threshold=threshold)
    classifier.save(model_path)
    return metrics


def infer(
    db_path: Path,
    model_path: Path,
    update_db: bool = True,
    threshold: Optional[float] = None,
    output_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
) -> None:
    classifier = TFIDFClassifier.load(model_path, config=_load_tfidf_config(config_path))
    if threshold is not None:
        classifier.threshold = threshold
    classifier.infer(db_path, update_db=update_db, output_path=output_path)


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
    i.add_argument("--output", type=Path, required=False, help="Path to write inference results JSON (optional)")

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
