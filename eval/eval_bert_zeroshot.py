"""Experiment 2 — pretrained BERT-tiny SMS spam model, zero-shot on e-mail.

Input: subject + body, truncated to 128 tokens. P(phishing) := P(LABEL_1 = spam).
Primary decision rule is the model's own argmax (P >= 0.5), i.e. truly zero-shot.
A second row re-picks only the threshold on VAL to show how much of the gap is
calibration under domain shift vs. ranking quality (ROC-AUC is threshold-free).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bert_utils import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    check_orientation,
    email_text,
    load_pretrained,
    predict_proba,
    time_single,
    token_length_share_over,
)
from common import (  # noqa: E402
    ARTIFACTS,
    REPORTS,
    full_metrics,
    latency_sample_ids,
    latency_stats,
    select_threshold_on_val,
    set_seeds,
    write_json,
)
from data_loader import load_splits  # noqa: E402

MAX_LEN = 128


def model_dir_size_mb() -> float:
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True))
    return sum(p.stat().st_size for p in path.iterdir() if p.name != "README.md" and p.is_file()) / 1e6


def main() -> dict:
    set_seeds()
    tok, model = load_pretrained()
    orientation = check_orientation(model, tok)
    splits = load_splits()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    probs = {}
    for name in ("val", "test"):
        df = splits[name]
        texts = [email_text(s, b) for s, b in zip(df["subject"], df["body"])]
        t0 = time.perf_counter()
        probs[name] = predict_proba(model, tok, texts, MAX_LEN)
        print(f"  zero-shot scored {name}: {len(df)} in {time.perf_counter() - t0:.1f}s")
        pd.DataFrame({"row_id": df["row_id"], "label": df["label"], "p_zeroshot": probs[name]}).to_csv(
            ARTIFACTS / f"zeroshot_{name}.csv", index=False
        )

    yv, yt = splits["val"]["label"].to_numpy(), splits["test"]["label"].to_numpy()
    sender = splits["test"]["has_sender"].to_numpy() == 1
    thr_val = select_threshold_on_val(yv, probs["val"])
    results = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "max_len": MAX_LEN,
        "label_orientation_check": orientation,
        "test_overall_argmax_0.5": full_metrics(yt, probs["test"], 0.5),
        "test_sender_available_argmax_0.5": full_metrics(yt[sender], probs["test"][sender], 0.5),
        "val_tuned_threshold": thr_val,
        "test_overall_val_threshold": full_metrics(yt, probs["test"], thr_val),
        "test_sender_available_val_threshold": full_metrics(yt[sender], probs["test"][sender], thr_val),
        "prob_range_test": [float(probs["test"].min()), float(probs["test"].max())],
    }
    test_texts = [email_text(s, b) for s, b in zip(splits["test"]["subject"], splits["test"]["body"])]
    results["share_test_emails_truncated_at_128_tokens"] = token_length_share_over(tok, test_texts, MAX_LEN)

    by_id = dict(zip(splits["test"]["row_id"], test_texts))
    ids = latency_sample_ids(splits["test"])
    results["latency"] = latency_stats(time_single(model, tok, [by_id[i] for i in ids], MAX_LEN))
    results["size_mb"] = model_dir_size_mb()

    write_json(REPORTS / "partial" / "zeroshot.json", results)
    m = results["test_overall_argmax_0.5"]
    print(f"  zero-shot @0.5: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f}; "
          f"val-thr {thr_val:.3f} -> F1={results['test_overall_val_threshold']['f1']:.3f}")
    return results


if __name__ == "__main__":
    main()
