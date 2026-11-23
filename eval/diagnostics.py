"""Sanity checks for suspiciously high scores (run after finetune_bert.py).

1. Fuzzy near-duplicates: TF-IDF cosine of every TEST e-mail to its nearest TRAIN
   e-mail, and the fine-tuned model's TEST metrics restricted to e-mails with no
   close train neighbour (cosine < 0.8). Exact/prefix duplicates are already 0.
2. Shortcut / corpus artefacts: a plain TF-IDF + logistic-regression reference
   model (threshold on VAL) and its most influential tokens. If a bag-of-words
   model is as good as BERT and leans on corpus-specific names, the benchmark
   is easy for reasons that will not transfer to a real inbox.
3. Corpus identification: how well TF-IDF features predict the *source corpus*.

Nothing here tunes the evaluated models; TEST is only scored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bert_utils import email_text  # noqa: E402
from common import ARTIFACTS, REPORTS, SEED, full_metrics, select_threshold_on_val, set_seeds, write_json  # noqa: E402
from data_loader import load_splits  # noqa: E402

COS_BANDS = (0.95, 0.9, 0.8)


def max_cosine_to_train(x_test, x_train, chunk: int = 500) -> np.ndarray:
    out = np.empty(x_test.shape[0])
    xt = x_train.T.tocsc()
    for start in range(0, x_test.shape[0], chunk):
        sims = x_test[start : start + chunk] @ xt
        out[start : start + chunk] = sims.max(axis=1).toarray().ravel()
    return out


def main() -> dict:
    set_seeds()
    splits = load_splits()
    texts = {k: [email_text(s, b) for s, b in zip(v["subject"], v["body"])] for k, v in splits.items()}
    y = {k: v["label"].to_numpy() for k, v in splits.items()}

    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.95, max_features=200_000, dtype=np.float32)
    X = {"train": vec.fit_transform(texts["train"])}
    X["val"], X["test"] = vec.transform(texts["val"]), vec.transform(texts["test"])

    # 1. near-duplicates
    max_cos = max_cosine_to_train(X["test"], X["train"])
    ft = pd.read_csv(ARTIFACTS / "finetuned_test.csv")
    assert (ft["row_id"].to_numpy() == splits["test"]["row_id"].to_numpy()).all()
    from common import read_json

    thr = read_json(REPORTS / "partial" / "finetuned.json")["val_threshold"]
    novel = max_cos < 0.8
    near = {
        "share_test_max_cosine_ge": {str(c): float((max_cos >= c).mean()) for c in COS_BANDS},
        "median_max_cosine": float(np.median(max_cos)),
        "finetuned_on_test_with_max_cosine_lt_0.8": full_metrics(y["test"][novel], ft["p_finetuned"].to_numpy()[novel], thr),
        "finetuned_on_test_with_max_cosine_ge_0.8": full_metrics(y["test"][~novel], ft["p_finetuned"].to_numpy()[~novel], thr),
    }

    # 2. bag-of-words reference model
    lr = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear", random_state=SEED)
    lr.fit(X["train"], y["train"])
    pv, pt = lr.predict_proba(X["val"])[:, 1], lr.predict_proba(X["test"])[:, 1]
    lr_thr = select_threshold_on_val(y["val"], pv)
    vocab = np.asarray(vec.get_feature_names_out())
    coef = lr.coef_.ravel()
    order = np.argsort(coef)
    tfidf_lr = {
        "val_threshold": lr_thr,
        "test_overall": full_metrics(y["test"], pt, lr_thr),
        "top_phishing_tokens": vocab[order[::-1][:30]].tolist(),
        "top_legit_tokens": vocab[order[:30]].tolist(),
    }

    # 3. corpus identification (multi-class, accuracy on TEST)
    src_clf = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear", random_state=SEED)
    src_clf.fit(X["train"], splits["train"]["source"])
    src_acc = float((src_clf.predict(X["test"]) == splits["test"]["source"].to_numpy()).mean())
    majority = float(splits["test"]["source"].value_counts(normalize=True).iloc[0])

    results = {
        "near_duplicates": near,
        "tfidf_logreg_reference": tfidf_lr,
        "source_identification": {"test_accuracy": src_acc, "majority_class_rate": majority},
    }
    write_json(REPORTS / "partial" / "diagnostics.json", results)
    print(f"  near-dup: test share max-cos>=0.9 {near['share_test_max_cosine_ge']['0.9']:.1%}; fine-tuned F1 on novel (<0.8) "
          f"{near['finetuned_on_test_with_max_cosine_lt_0.8']['f1']:.3f} (n={near['finetuned_on_test_with_max_cosine_lt_0.8']['n']})")
    print(f"  TF-IDF+LR reference: F1={tfidf_lr['test_overall']['f1']:.3f} ROC-AUC={tfidf_lr['test_overall']['roc_auc']:.4f}; "
          f"source-ID accuracy {src_acc:.1%} (majority {majority:.1%})")
    return results


if __name__ == "__main__":
    main()
