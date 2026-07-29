"""Standalone training script for the SentinelAI risk model.

Usage:
    python train_model.py [labeled_emails.csv] [model.pkl]

Expects a labeled CSV (default: labeled_emails.csv, supplied separately —
this script never fabricates data) with a `label` column containing
phishing/legit (also accepts 1/0, spam/ham, true/false), and either:

  a) precomputed feature columns matching feature_vector.FEATURE_NAMES, or
  b) raw email columns (sender, subject, body) — features are extracted
     on the fly with feature_vector.build_feature_vector(). An optional
     bert_phishing_prob column is used when present; otherwise the BERT
     pipeline runs per row (slower).

Trains a LightGBM classifier, prints feature importances and held-out
precision/recall/F1, and saves the model + schema to model.pkl.
"""

import os
import pickle
import sys

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from feature_vector import FEATURE_NAMES, build_feature_vector

DEFAULT_CSV = "labeled_emails.csv"
DEFAULT_MODEL = "model.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

POSITIVE_LABELS = {"phishing", "spam", "1", "true", "yes"}
NEGATIVE_LABELS = {"legit", "ham", "0", "false", "no"}


def parse_label(value):
    text = str(value).strip().lower()

    if text in POSITIVE_LABELS:
        return 1

    if text in NEGATIVE_LABELS:
        return 0

    raise ValueError(f"Unrecognized label: {value!r}")


def load_features(df):
    """Return an (X, feature_names) DataFrame in schema order."""
    if all(name in df.columns for name in FEATURE_NAMES):
        print("Using precomputed feature columns from CSV.")
        return df[FEATURE_NAMES].astype(float)

    required = {"sender", "subject", "body"}
    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"CSV must contain either all feature columns "
            f"({', '.join(FEATURE_NAMES)}) or raw email columns "
            f"(sender, subject, body). Missing: {', '.join(sorted(missing))}"
        )

    print("Extracting features from raw email columns...")

    has_bert_column = "bert_phishing_prob" in df.columns

    rows = []

    for index, row in df.iterrows():
        bert_prob = (
            float(row["bert_phishing_prob"]) if has_bert_column else None
        )

        features = build_feature_vector(
            str(row["sender"]) if pd.notna(row["sender"]) else "",
            str(row["subject"]) if pd.notna(row["subject"]) else "",
            str(row["body"]) if pd.notna(row["body"]) else "",
            headers=None,
            bert_prob=bert_prob,
        )
        rows.append(features)

        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(df)} rows processed")

    return pd.DataFrame(rows, columns=FEATURE_NAMES).astype(float)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    model_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Labeled dataset not found: {csv_path}\n"
            f"Provide a CSV with a 'label' column (phishing/legit) plus "
            f"feature columns or raw sender/subject/body columns."
        )

    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column.")

    y = df["label"].map(parse_label)
    X = load_features(df)

    print(f"\nDataset: {len(X)} emails "
          f"({int(y.sum())} phishing, {int((1 - y).sum())} legit)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    print("\nFEATURE IMPORTANCES")
    print("-" * 50)

    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for name, importance in importances:
        print(f"{name:26} {importance}")

    y_pred = model.predict(X_test)

    print("\nHELD-OUT EVALUATION "
          f"({len(X_test)} emails, {TEST_SIZE:.0%} split)")
    print("-" * 50)
    print(classification_report(
        y_test, y_pred, target_names=["legit", "phishing"], digits=3
    ))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
    }

    with open(model_path, "wb") as model_file:
        pickle.dump(payload, model_file)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
