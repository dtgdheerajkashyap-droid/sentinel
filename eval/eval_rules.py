"""Experiment 1 — the production rule engine (sentinel_ai_v2.py), unmodified.

The rules are imported, never re-implemented. `score_email` mirrors the per-email
logic of `run_scanner()` exactly:

    urls          = extract_urls(body)                       # body only, like production
    url_raw       = sum(url_heuristic_score(analyze_url(u)[0]) for u in urls)
    sender_raw    = sender_heuristic_score(analyze_sender(sender)[0])
    urgency_risk  = urgency_heuristic_score(analyze_urgency(subject + " " + body)[0])
    calculate_risk(url_raw, sender_raw, urgency_risk, features=None)

`features=None` forces the heuristic path (url max 35 + sender max 25 + urgency
max 15). No model.pkl exists in the repo, so production would take the same path.
No Gmail / OAuth / database function is called.

Reproducibility: production calls `tldextract.extract`, which downloads the live
Public Suffix List on first use. Here the same library is pointed at its bundled
PSL snapshot (pinned by the tldextract version) so results never depend on the
day the script is run. No rule, keyword or weight is changed.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ARTIFACTS,
    REPORTS,
    ROOT,
    full_metrics,
    latency_sample_ids,
    latency_stats,
    set_seeds,
    write_json,
)
from data_loader import load_splits  # noqa: E402


def load_scanner():
    import tldextract

    import sentinel_ai_v2 as sv

    sv.tldextract = types.SimpleNamespace(extract=tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None))
    assert sv.RISK_MODEL is None, "model.pkl present: production would not use the heuristic path"
    return sv


def score_email(sv, sender: str, subject: str, body: str) -> dict:
    urls = sv.extract_urls(body)
    url_features_list = [sv.analyze_url(url)[0] for url in urls]
    sender_features, _ = sv.analyze_sender(sender)
    urgency_features, _ = sv.analyze_urgency(subject + " " + body)

    total_url_raw_score = sum(sv.url_heuristic_score(f) for f in url_features_list)
    sender_raw_score = sv.sender_heuristic_score(sender_features)
    urgency_risk = sv.urgency_heuristic_score(urgency_features)

    final_score, level, url_risk, sender_risk = sv.calculate_risk(
        total_url_raw_score, sender_raw_score, urgency_risk, features=None
    )
    return {
        "n_urls": len(urls),
        "url_raw": total_url_raw_score,
        "sender_raw": sender_raw_score,
        "url_risk": url_risk,
        "sender_risk": sender_risk,
        "urgency_risk": urgency_risk,
        "rule_score": final_score,
        "level": level,
        "sender_trusted": bool(sender_features["is_trusted"]),
    }


def score_split(sv, df: pd.DataFrame) -> pd.DataFrame:
    rows = [score_email(sv, r.sender, r.subject, r.body) for r in df.itertuples(index=False)]
    out = pd.DataFrame(rows)
    out.insert(0, "row_id", df["row_id"].to_numpy())
    out["label"] = df["label"].to_numpy()
    out["has_sender"] = df["has_sender"].to_numpy()
    out["source"] = df["source"].to_numpy()
    return out


def evaluate(scores: pd.DataFrame) -> dict:
    y, s = scores["label"].to_numpy(), scores["rule_score"].to_numpy()
    return {
        "medium_plus_ge30": full_metrics(y, s, 30),
        "high_only_ge70": full_metrics(y, s, 70),
    }


def main() -> dict:
    set_seeds()
    sv = load_scanner()
    splits = load_splits()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    per_split = {}
    for name in ("val", "test"):
        t0 = time.perf_counter()
        sc = score_split(sv, splits[name])
        print(f"  rules scored {name}: {len(sc)} e-mails in {time.perf_counter() - t0:.1f}s")
        sc.to_csv(ARTIFACTS / f"rules_{name}.csv", index=False)
        per_split[name] = sc

    test = per_split["test"]
    sender_mask = test["has_sender"] == 1
    results = {
        "test_overall": evaluate(test),
        "test_sender_available": evaluate(test[sender_mask]),
        "test_per_source_ge30": {
            src: full_metrics(g["label"], g["rule_score"], 30) for src, g in test.groupby("source")
        },
        "diagnostics_test": {
            "share_with_any_extracted_url": float((test["n_urls"] > 0).mean()),
            "share_with_any_extracted_url_by_label": {int(k): float(v) for k, v in test.groupby("label")["n_urls"].apply(lambda s: (s > 0).mean()).items()},
            "share_score_zero_by_label": {int(k): float(v) for k, v in test.groupby("label")["rule_score"].apply(lambda s: (s == 0).mean()).items()},
            "level_counts_by_label": {int(k): g["level"].value_counts().to_dict() for k, g in test.groupby("label")},
            "sender_in_TRUSTED_DOMAINS_by_label": {int(k): int(v) for k, v in test.groupby("label")["sender_trusted"].sum().items()},
            "mean_component_by_label": {
                int(k): {c: float(g[c].mean()) for c in ("url_risk", "sender_risk", "urgency_risk", "rule_score")}
                for k, g in test.groupby("label")
            },
        },
    }

    # Latency: full per-email rule pipeline, single e-mail at a time, after warm-up.
    test_df = splits["test"].set_index("row_id")
    ids = latency_sample_ids(splits["test"])
    for rid in ids[:50]:
        r = test_df.loc[rid]
        score_email(sv, r.sender, r.subject, r.body)
    times = []
    for rid in ids:
        r = test_df.loc[rid]
        t0 = time.perf_counter()
        score_email(sv, r.sender, r.subject, r.body)
        times.append(time.perf_counter() - t0)
    results["latency"] = latency_stats(times)
    results["size_mb"] = (ROOT / "sentinel_ai_v2.py").stat().st_size / 1e6
    results["size_note"] = "sentinel_ai_v2.py source only (rules have no weights); excludes the tldextract PSL snapshot"

    write_json(REPORTS / "partial" / "rules.json", results)
    m30, m70 = results["test_overall"]["medium_plus_ge30"], results["test_overall"]["high_only_ge70"]
    print(f"  rules >=30: P={m30['precision']:.3f} R={m30['recall']:.3f} F1={m30['f1']:.3f} | "
          f">=70: P={m70['precision']:.3f} R={m70['recall']:.3f} F1={m70['f1']:.3f} | ROC-AUC={m30['roc_auc']:.3f}")
    return results


def check_psl_parity() -> dict:
    """Optional, needs network: re-score TEST with production's default (live PSL)
    tldextract and count e-mails whose score differs from the snapshot run."""
    import tldextract

    sv = load_scanner()
    test = load_splits()["test"]
    snapshot = score_split(sv, test)["rule_score"].to_numpy()
    sv.tldextract = types.SimpleNamespace(extract=tldextract.TLDExtract())
    live = score_split(sv, test)["rule_score"].to_numpy()
    out = {"n_test": int(len(test)), "n_scores_differ": int((snapshot != live).sum()), "tldextract": tldextract.__version__}
    write_json(REPORTS / "partial" / "psl_parity.json", out)
    print(f"  PSL parity: {out['n_scores_differ']} of {out['n_test']} test scores differ (snapshot vs live)")
    return out


if __name__ == "__main__":
    if "--check-psl-parity" in sys.argv:
        check_psl_parity()
    else:
        main()
