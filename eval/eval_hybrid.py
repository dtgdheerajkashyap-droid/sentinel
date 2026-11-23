"""Experiment 4 — hybrid: production rule sub-scores + fine-tuned BERT probability.

Slot layout follows risk_enginev2.py (url 35 + sender 25 + bert 40 = 100):

  A (no urgency):  score = url_risk + sender_risk + 40 * P(phishing)            in [0, 100]
  B (rescaled):    score = (url_risk + sender_risk + urgency_risk + 40 * P) * 100/115

url_risk / sender_risk / urgency_risk are exactly the values production's
calculate_risk() / urgency_heuristic_score() return (written by eval_rules.py);
P comes from the fine-tuned model (finetune_bert.py). The decision threshold of
each variant is chosen on VAL (max F1) and applied once to TEST.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bert_utils import SPAM_INDEX, email_text, load_pretrained  # noqa: E402
from common import (  # noqa: E402
    ARTIFACTS,
    REPORTS,
    ROOT,
    dir_size_mb,
    full_metrics,
    latency_sample_ids,
    latency_stats,
    select_threshold_on_val,
    set_seeds,
    write_json,
)
from data_loader import load_splits  # noqa: E402
from eval_rules import load_scanner, score_email  # noqa: E402
from finetune_bert import HPARAMS, OUT_DIR  # noqa: E402

VARIANTS = {
    "hybrid_A_url_sender_bert": lambda d: d["url_risk"] + d["sender_risk"] + 40.0 * d["p_finetuned"],
    "hybrid_B_rescaled_with_urgency": lambda d: (d["url_risk"] + d["sender_risk"] + d["urgency_risk"] + 40.0 * d["p_finetuned"]) * 100.0 / 115.0,
}


def load_components(split: str) -> pd.DataFrame:
    rules = pd.read_csv(ARTIFACTS / f"rules_{split}.csv")
    bert = pd.read_csv(ARTIFACTS / f"finetuned_{split}.csv")[["row_id", "p_finetuned"]]
    df = rules.merge(bert, on="row_id", how="inner", validate="one_to_one")
    assert len(df) == len(rules) == len(bert)
    for name, fn in VARIANTS.items():
        df[name] = fn(df)
    return df


def main() -> dict:
    set_seeds()
    comps = {s: load_components(s) for s in ("val", "test")}
    results = {}
    for name in VARIANTS:
        val, test = comps["val"], comps["test"]
        thr = select_threshold_on_val(val["label"], val[name])
        sender = test["has_sender"] == 1
        results[name] = {
            "val_threshold": thr,
            "val_at_threshold": full_metrics(val["label"], val[name], thr),
            "test_overall": full_metrics(test["label"], test[name], thr),
            "test_sender_available": full_metrics(test.loc[sender, "label"], test.loc[sender, name], thr),
            "test_per_source": {src: full_metrics(g["label"], g[name], thr) for src, g in test.groupby("source")},
        }
    pd.concat([comps["test"].assign(split="test"), comps["val"].assign(split="val")]).to_csv(ARTIFACTS / "hybrid_scores.csv", index=False)

    # Latency: rules + fine-tuned BERT forward pass for one e-mail, timed together.
    sv = load_scanner()
    tok, model = load_pretrained(str(OUT_DIR), revision=None)
    import torch

    test_df = load_splits()["test"].set_index("row_id")
    ids = latency_sample_ids(test_df.reset_index())

    @torch.no_grad()
    def hybrid_one(r):
        rc = score_email(sv, r.sender, r.subject, r.body)
        enc = tok(email_text(r.subject, r.body), truncation=True, max_length=HPARAMS["max_len"], return_tensors="pt")
        p = model(**enc).logits.softmax(-1)[0, SPAM_INDEX].item()
        return rc["url_risk"] + rc["sender_risk"] + 40.0 * p

    rows = [test_df.loc[i] for i in ids]
    for r in rows[:50]:
        hybrid_one(r)
    times = []
    for r in rows:
        t0 = time.perf_counter()
        hybrid_one(r)
        times.append(time.perf_counter() - t0)
    lat = latency_stats(times)
    size = dir_size_mb(OUT_DIR) + (ROOT / "sentinel_ai_v2.py").stat().st_size / 1e6
    for name in VARIANTS:
        results[name]["latency"] = lat
        results[name]["size_mb"] = size

    write_json(REPORTS / "partial" / "hybrid.json", results)
    for name, r in results.items():
        m = r["test_overall"]
        print(f"  {name} (thr {r['val_threshold']:.2f}): P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f}")
    return results


if __name__ == "__main__":
    main()
