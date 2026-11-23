"""Load, clean and split the public phishing/legitimate e-mail corpora.

Data source
-----------
"Phishing Email Curated Datasets" (Champa, Rabbi & Zibran), Zenodo record
8339691, CC-BY-4.0. These are the same six corpora that make up the Kaggle
"Phishing Email Dataset" (CEAS_08, Enron, Ling, Nazario, Nigerian_Fraud,
SpamAssasin). If the files are already in data/raw/ (e.g. you downloaded them
from Kaggle) they are used as-is; otherwise they are downloaded and their MD5
checksums verified against the Zenodo record.

Output
------
data/splits/{train,val,test}.csv  with columns
    row_id, source, sender, subject, body, label, has_sender
data/splits/manifest.json          counts, drops, class balance, sha256 of each split

Label convention: 1 = phishing/spam, 0 = legitimate (ham).

Usage:  python eval/data_loader.py
"""

from __future__ import annotations

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_RAW, DATA_SPLITS, SEED, write_json  # noqa: E402

ZENODO = "https://zenodo.org/api/records/8339691/files/{name}/content"
SOURCES = {
    # file name: md5 from the Zenodo record
    "CEAS_08.csv": "1f0d59191eec892a709995d48cd8decb",
    "Enron.csv": "10489e0fdcc8af6fe62135c689367d40",
    "Ling.csv": "eac6399dc0c7ae455aebb64423dadc8b",
    "Nazario.csv": "4022d055bb7cf8602f30f768f652e91e",
    "Nigerian_Fraud.csv": "65fd47ae7cb4ae762e4be42b11905d63",
    "SpamAssasin.csv": "d921e0b2333bfaa058bd38193e6b3fbd",
}
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
N_FOLDS = 20            # 20 group-folds of ~5%: 14 train / 3 val / 3 test = 70/15/15
TEMPLATE_PREFIX = 200   # chars of normalised body defining a "template cluster"
TEMPLATE_CAP = 10       # max e-mails kept per template cluster (campaign down-sampling)
COLUMNS = ["row_id", "source", "sender", "subject", "body", "label", "has_sender"]


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - integrity check against published checksum
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_raw_files() -> dict:
    """Download any missing corpus file and verify checksums."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, expected in SOURCES.items():
        path = DATA_RAW / name
        if not path.exists():
            print(f"  downloading {name} ...")
            urllib.request.urlretrieve(ZENODO.format(name=name), path)  # noqa: S310 - fixed URL
        actual = _md5(path)
        checks[name] = {"md5": actual, "matches_zenodo": actual == expected}
        if actual != expected:
            print(f"  WARNING: {name} md5 {actual} != Zenodo {expected} (using local file anyway)")
    return checks


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_raw() -> pd.DataFrame:
    frames = []
    for name in SOURCES:
        df = pd.read_csv(DATA_RAW / name)
        source = name.removesuffix(".csv")
        sender = df["sender"] if "sender" in df.columns else pd.Series([None] * len(df))
        frames.append(
            pd.DataFrame(
                {
                    "source": source,
                    "sender": sender.fillna("").astype(str).str.strip(),
                    "subject": df["subject"].fillna("").astype(str),
                    "body": df["body"].fillna("").astype(str),
                    "label": df["label"].astype(int),
                    "source_has_sender_column": "sender" in df.columns,
                    # dataset's own "body contained URLs" flag (absent for Enron/Ling)
                    "urls_flag": df["urls"].astype(float) if "urls" in df.columns else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def url_stripping_stats(raw: pd.DataFrame) -> dict:
    """The curated release removed many URLs from bodies but kept a `urls` flag.
    Production's URL rules only see URLs still present in the text."""
    out = {}
    for src, g in raw.groupby("source"):
        has_http = g["body"].str.contains(r"https?://", regex=True)
        flag = g["urls_flag"]
        # Enron/Ling store URLs tokenised with spaces ("http : / / www . x . com"),
        # which production's extract_urls() regex cannot match.
        spaced = g["body"].str.contains(r"https?\s+:\s*/\s*/", regex=True)
        out[src] = {
            "share_flagged_urls": None if flag.isna().all() else float(flag.mean()),
            "share_body_contains_http_url": float(has_http.mean()),
            "share_body_contains_space_tokenised_url": float(spaced.mean()),
            "share_flagged_but_no_url_in_body": None if flag.isna().all() else float(((flag == 1) & ~has_http).mean()),
        }
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    drops = {"raw_rows": int(len(df))}

    empty = df["body"].str.strip().str.len() == 0
    drops["empty_body"] = int(empty.sum())
    df = df[~empty]

    exact = df.duplicated(subset=["sender", "subject", "body", "label"], keep="first")
    drops["exact_duplicate_rows"] = int(exact.sum())
    df = df[~exact]

    # Same normalised subject+body appearing with BOTH labels: ambiguous ground truth.
    df = df.assign(_key=(df["subject"].map(_norm) + "\x00" + df["body"].map(_norm)))
    conflicting = df.groupby("_key")["label"].transform("nunique") > 1
    drops["conflicting_label_rows"] = int(conflicting.sum())
    df = df[~conflicting]

    # Stricter than "exact": same text up to case/whitespace (e.g. the same spam
    # blast to different recipients). Removing these prevents train/test leakage.
    near = df.duplicated(subset=["_key"], keep="first")
    drops["normalised_text_duplicates"] = int(near.sum())
    df = df[~near]
    naive_base = df.drop(columns="_key").copy()  # state of the first (naive) attempt, kept for the diagnostic

    # Same body with a different subject line is still the same e-mail to a model.
    same_body = df["body"].map(_norm).duplicated(keep="first")
    drops["duplicate_body_different_subject"] = int(same_body.sum())
    df = df[~same_body].drop(columns="_key")

    # Template clusters (identical first 200 normalised chars: spam campaigns,
    # newsletters, mailing-list digests). One CEAS_08 cluster alone has ~2,900
    # spoofed "CNN Daily Top 10" spams. Cap each cluster so no single campaign
    # dominates a split; the split below keeps each cluster on ONE side.
    df = df.assign(template=df["body"].map(lambda t: _norm(t)[:TEMPLATE_PREFIX]))
    rank = df.groupby("template").cumcount()
    over_cap = rank >= TEMPLATE_CAP
    drops["template_cluster_over_cap"] = int(over_cap.sum())
    drops["largest_template_cluster_before_cap"] = int(df.groupby("template").size().max())
    df = df[~over_cap]

    drops["kept_rows"] = int(len(df))
    df = df.reset_index(drop=True)
    df["row_id"] = range(len(df))
    df["has_sender"] = (df["sender"].str.len() > 0).astype(int)
    return df, drops, naive_base


def naive_split_leakage(df: pd.DataFrame) -> dict:
    """Diagnostic only: the leakage a plain random stratified 70/15/15 split would have
    after exact + case/whitespace de-duplication (this was the first attempt)."""
    strata = df["source"] + "_" + df["label"].astype(str)
    trainval, test = train_test_split(df, test_size=SPLIT_FRACTIONS[2], stratify=strata, random_state=SEED)
    tv_strata = trainval["source"] + "_" + trainval["label"].astype(str)
    train, val = train_test_split(trainval, test_size=SPLIT_FRACTIONS[1] / (1 - SPLIT_FRACTIONS[2]), stratify=tv_strata, random_state=SEED)
    return leakage_checks({"train": train, "val": val, "test": test})


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One stratified, group-aware 70/15/15 split.

    Stratified on source x label; grouped on template cluster so near-duplicate
    e-mails never straddle train and test.
    """
    strata = (df["source"] + "_" + df["label"].astype(str)).to_numpy()
    groups = pd.factorize(df["template"])[0]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(df)), strata, groups)):
        fold[idx] = k
    n_test = round(N_FOLDS * SPLIT_FRACTIONS[2])
    n_val = round(N_FOLDS * SPLIT_FRACTIONS[1])
    assign = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))
    return {name: df[assign == name].sort_values("row_id")[COLUMNS].reset_index(drop=True) for name in ("train", "val", "test")}


def describe(df: pd.DataFrame) -> dict:
    return {
        "n": int(len(df)),
        "n_phishing": int(df["label"].sum()),
        "n_legit": int((df["label"] == 0).sum()),
        "phishing_rate": float(df["label"].mean()),
        "n_with_sender": int(df["has_sender"].sum()),
        "body_has_http_url": float(df["body"].str.contains(r"https?://", regex=True).mean()),
    }


def leakage_checks(splits: dict[str, pd.DataFrame]) -> dict:
    """Cross-split overlap. Exact overlap must be 0; prefix overlap is a near-dup proxy."""

    def body_keys(d):
        return set(d["body"].map(_norm))

    def prefix_keys(d, n=200):
        return set(d["body"].map(lambda t: _norm(t)[:n]))

    tr = splits["train"]
    out = {}
    for name in ("val", "test"):
        other = splits[name]
        out[f"{name}_bodies_identical_to_a_train_body"] = int(other["body"].map(_norm).isin(body_keys(tr)).sum())
        out[f"{name}_first200chars_match_a_train_body"] = int(other["body"].map(lambda t: _norm(t)[:200]).isin(prefix_keys(tr)).sum())
        out[f"{name}_n"] = int(len(other))
    return out


def load_splits() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(DATA_SPLITS / f"{name}.csv", keep_default_na=False, dtype={"sender": str, "subject": str, "body": str})
        df["label"] = df["label"].astype(int)
        df["has_sender"] = df["has_sender"].astype(int)
        df["row_id"] = df["row_id"].astype(int)
        out[name] = df
    return out


def main() -> dict:
    checks = ensure_raw_files()
    raw = load_raw()
    per_source_raw = raw.groupby("source").agg(rows=("label", "size"), phishing=("label", "sum"), has_sender_col=("source_has_sender_column", "first"))

    df, drops, naive_base = clean(raw.drop(columns="urls_flag"))
    splits = split(df)
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, part in splits.items():
        path = DATA_SPLITS / f"{name}.csv"
        part.to_csv(path, index=False, lineterminator="\n")
        hashes[name] = _sha256(path)

    manifest = {
        "source": "Zenodo 8339691 'Phishing Email Curated Datasets' (CC-BY-4.0)",
        "raw_file_checksums": checks,
        "seed": SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "stratified_on": "source x label",
        "grouped_on": f"first {TEMPLATE_PREFIX} chars of normalised body (StratifiedGroupKFold, {N_FOLDS} folds)",
        "template_cap": TEMPLATE_CAP,
        "drops": drops,
        "per_source_raw": {s: {k: (bool(v) if k == "has_sender_col" else int(v)) for k, v in r.items()} for s, r in per_source_raw.to_dict("index").items()},
        "url_stripping_raw": url_stripping_stats(raw),
        "per_source_clean": {
            s: describe(g) for s, g in df.groupby("source")
        },
        "overall_clean": describe(df),
        "splits": {name: describe(part) for name, part in splits.items()},
        "split_sha256": hashes,
        "leakage_checks": leakage_checks(splits),
        "leakage_naive_random_split_diagnostic": naive_split_leakage(naive_base),
    }
    write_json(DATA_SPLITS / "manifest.json", manifest)

    print(f"  raw rows {drops['raw_rows']}, kept {drops['kept_rows']}  (dropped: empty body {drops['empty_body']}, "
          f"exact dup {drops['exact_duplicate_rows']}, conflicting label {drops['conflicting_label_rows']}, "
          f"normalised-text dup {drops['normalised_text_duplicates']}, same body/diff subject "
          f"{drops['duplicate_body_different_subject']}, template over cap {drops['template_cluster_over_cap']})")
    for name, part in splits.items():
        d = describe(part)
        print(f"  {name:<5} n={d['n']:>6}  phishing={d['phishing_rate']:.1%}  with sender={d['n_with_sender']}")
    print("  leakage:", manifest["leakage_checks"])
    return manifest


if __name__ == "__main__":
    main()
