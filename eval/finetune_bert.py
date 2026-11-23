"""Experiment 3 — fine-tune the same BERT-tiny checkpoint on the TRAIN split.

Protocol (no test-set tuning):
  * hyper-parameters fixed a priori (below) — none were searched;
  * the epoch is selected by lowest VAL loss;
  * the decision threshold is selected by max F1 on VAL;
  * TEST is scored exactly once, with the selected epoch and threshold.

CPU-friendly: bert-tiny (4.4M params), max_len 256, fixed seeds, deterministic
torch algorithms, fixed thread count. Weights are saved to models/ (git-ignored).
"""

from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bert_utils import SPAM_INDEX, email_text, load_pretrained, predict_proba, time_single  # noqa: E402
from common import (  # noqa: E402
    ARTIFACTS,
    MODELS,
    REPORTS,
    SEED,
    TORCH_THREADS,
    dir_size_mb,
    full_metrics,
    latency_sample_ids,
    latency_stats,
    ranking_metrics,
    select_threshold_on_val,
    set_seeds,
    write_json,
)
from data_loader import load_splits  # noqa: E402

HPARAMS = {
    "max_len": 256,
    "epochs": 4,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "grad_clip": 1.0,
    "seed": SEED,
    "torch_threads": TORCH_THREADS,
}
OUT_DIR = MODELS / "bert_tiny_finetuned"


def encode(tok, df: pd.DataFrame) -> list[list[int]]:
    texts = [email_text(s, b) for s, b in zip(df["subject"], df["body"])]
    return tok(texts, truncation=True, max_length=HPARAMS["max_len"])["input_ids"]


def make_loader(tok, ids, labels, shuffle: bool, generator=None):
    # Map dataset label (1 = phishing) onto the checkpoint's spam index.
    targets = [SPAM_INDEX if y == 1 else 1 - SPAM_INDEX for y in labels]
    items = list(zip(ids, targets))

    def collate(batch):
        enc = tok.pad({"input_ids": [b[0] for b in batch]}, return_tensors="pt")
        enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc

    return DataLoader(items, batch_size=HPARAMS["batch_size"], shuffle=shuffle, collate_fn=collate, generator=generator, num_workers=0)


@torch.no_grad()
def eval_loss(model, loader) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        out = model(**batch)
        total += out.loss.item() * len(batch["labels"])
        n += len(batch["labels"])
    return total / n


def main() -> dict:
    set_seeds()
    tok, model = load_pretrained()
    splits = load_splits()

    t0 = time.perf_counter()
    enc = {name: encode(tok, splits[name]) for name in ("train", "val")}
    print(f"  tokenised train/val in {time.perf_counter() - t0:.1f}s")

    gen = torch.Generator().manual_seed(SEED)
    train_loader = make_loader(tok, enc["train"], splits["train"]["label"].tolist(), shuffle=True, generator=gen)
    val_loader = make_loader(tok, enc["val"], splits["val"]["label"].tolist(), shuffle=False)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": HPARAMS["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=HPARAMS["lr"])
    total_steps = HPARAMS["epochs"] * len(train_loader)
    warmup = math.ceil(HPARAMS["warmup_ratio"] * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup))
    )

    val_texts = [email_text(s, b) for s, b in zip(splits["val"]["subject"], splits["val"]["body"])]
    yv = splits["val"]["label"].to_numpy()
    log, best = [], {"val_loss": float("inf")}
    log.append({"epoch": 0, "train_loss": None, "val_loss": eval_loss(model, val_loader)})
    print(f"  epoch 0 (pretrained)  val_loss={log[0]['val_loss']:.4f}")

    for epoch in range(1, HPARAMS["epochs"] + 1):
        model.train()
        t_ep, run, n = time.perf_counter(), 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["grad_clip"])
            opt.step()
            sched.step()
            opt.zero_grad()
            run += out.loss.item() * len(batch["labels"])
            n += len(batch["labels"])
            if step % 400 == 0:
                print(f"    epoch {epoch} step {step}/{len(train_loader)} train_loss={run / n:.4f}", flush=True)
        val_loss = eval_loss(model, val_loader)
        pv = predict_proba(model, tok, val_texts, HPARAMS["max_len"])
        entry = {
            "epoch": epoch,
            "train_loss": run / n,
            "val_loss": val_loss,
            "val_f1_at_0.5": full_metrics(yv, pv, 0.5)["f1"],
            "val_roc_auc": ranking_metrics(yv, pv)["roc_auc"],
            "epoch_seconds": time.perf_counter() - t_ep,
        }
        log.append(entry)
        print(f"  epoch {epoch}  train_loss={entry['train_loss']:.4f}  val_loss={val_loss:.4f}  "
              f"val_F1@0.5={entry['val_f1_at_0.5']:.4f}  val_AUC={entry['val_roc_auc']:.4f}  ({entry['epoch_seconds']:.0f}s)", flush=True)
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}

    model.load_state_dict(best["state"])
    model.eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR, safe_serialization=True)
    tok.save_pretrained(OUT_DIR)

    probs = {"val": predict_proba(model, tok, val_texts, HPARAMS["max_len"])}
    test_texts = [email_text(s, b) for s, b in zip(splits["test"]["subject"], splits["test"]["body"])]
    thr = select_threshold_on_val(yv, probs["val"])
    probs["test"] = predict_proba(model, tok, test_texts, HPARAMS["max_len"])  # single test pass
    for name in ("val", "test"):
        df = splits[name]
        pd.DataFrame({"row_id": df["row_id"], "label": df["label"], "p_finetuned": probs[name]}).to_csv(
            ARTIFACTS / f"finetuned_{name}.csv", index=False
        )

    yt = splits["test"]["label"].to_numpy()
    sender = splits["test"]["has_sender"].to_numpy() == 1
    by_id = dict(zip(splits["test"]["row_id"], test_texts))
    ids = latency_sample_ids(splits["test"])
    results = {
        "hparams": HPARAMS,
        "epoch_log": log,
        "selected_epoch": best["epoch"],
        "selection_rule": "lowest VAL loss",
        "val_threshold": thr,
        "val_at_threshold": full_metrics(yv, probs["val"], thr),
        "test_overall": full_metrics(yt, probs["test"], thr),
        "test_sender_available": full_metrics(yt[sender], probs["test"][sender], thr),
        "test_overall_at_0.5": full_metrics(yt, probs["test"], 0.5),
        "test_per_source": {
            src: full_metrics(yt[(splits["test"]["source"] == src).to_numpy()], probs["test"][(splits["test"]["source"] == src).to_numpy()], thr)
            for src in sorted(splits["test"]["source"].unique())
        },
        "latency": latency_stats(time_single(model, tok, [by_id[i] for i in ids], HPARAMS["max_len"])),
        "size_mb": dir_size_mb(OUT_DIR),
    }
    write_json(REPORTS / "partial" / "finetuned.json", results)
    m = results["test_overall"]
    print(f"  fine-tuned (epoch {best['epoch']}, thr {thr:.3f}): P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} ROC-AUC={m['roc_auc']:.3f}")
    return results


if __name__ == "__main__":
    main()
