"""BERT-tiny helpers shared by the zero-shot, fine-tuning and hybrid experiments."""

from __future__ import annotations

import time

import numpy as np
import torch

MODEL_ID = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
MODEL_REVISION = "012e2681d92fa3f9be2ea08bd31cfa1db14599d2"  # pinned commit
SPAM_INDEX = 1  # LABEL_1 = spam (sms_spam convention; verified at runtime by check_orientation)
MAX_WORDS = 2000  # input cap applied identically everywhere; >> 256 tokens for real text

# Two spam and two ham messages: the card's widget example plus UCI SMS-style texts.
ORIENTATION_PROBES = [
    ("Camera - You are awarded a SiPix Digital Camera! call 09061221066 fromm landline. Delivery within 28 days.", 1),
    ("WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.", 1),
    ("Ok lar... Joking wif u oni...", 0),
    ("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k?", 0),
]


def email_text(subject: str, body: str) -> str:
    """Model input: subject + newline + body, whitespace-collapsed, first MAX_WORDS words.

    Collapsing whitespace does not change BERT's tokenisation (it splits on any
    whitespace run); the word cap only avoids tokenising multi-MB bodies.
    """
    words = f"{subject}\n{body}".split()
    return " ".join(words[:MAX_WORDS])


def load_pretrained(path_or_id: str = MODEL_ID, revision: str | None = MODEL_REVISION):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    try:
        tok = AutoTokenizer.from_pretrained(path_or_id, local_files_only=True, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(path_or_id, local_files_only=True, **kwargs)
    except OSError:
        tok = AutoTokenizer.from_pretrained(path_or_id, **kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(path_or_id, **kwargs)
    return tok, model.eval()


@torch.no_grad()
def predict_proba(model, tok, texts: list[str], max_len: int, batch_size: int = 64) -> np.ndarray:
    """P(spam/phishing) for each text. Batches are length-sorted for speed; output keeps input order."""
    model.eval()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty(len(texts), dtype=np.float64)
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        probs = model(**enc).logits.softmax(-1)[:, SPAM_INDEX]
        out[idx] = probs.double().numpy()
    return out


@torch.no_grad()
def time_single(model, tok, texts: list[str], max_len: int, warmup: int = 50) -> list[float]:
    """Per-email latency, batch size 1, tokenisation included, after warm-up."""
    model.eval()
    for t in texts[:warmup]:
        model(**tok(t, truncation=True, max_length=max_len, return_tensors="pt"))
    times = []
    for t in texts:
        t0 = time.perf_counter()
        model(**tok(t, truncation=True, max_length=max_len, return_tensors="pt")).logits.softmax(-1)
        times.append(time.perf_counter() - t0)
    return times


def check_orientation(model, tok) -> dict:
    probs = predict_proba(model, tok, [t for t, _ in ORIENTATION_PROBES], max_len=128)
    labels = [y for _, y in ORIENTATION_PROBES]
    ok = all((p > 0.5) == bool(y) for p, y in zip(probs, labels))
    assert ok, f"SPAM_INDEX orientation check failed: {probs}"
    return {"spam_index": SPAM_INDEX, "probe_probs": [float(p) for p in probs], "probe_labels": labels, "passed": ok}


def token_length_share_over(tok, texts: list[str], max_len: int, sample: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(sample, len(texts)), replace=False)
    lengths = [len(tok(texts[i], truncation=False)["input_ids"]) for i in idx]
    return float(np.mean(np.asarray(lengths) > max_len))
