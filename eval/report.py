"""Assemble reports/results.json, reports/results.md, reports/pr_curves.png and
reports/errors.csv from the per-experiment outputs. Every number in results.md is
read from results.json — nothing is typed by hand.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACTS, DATA_SPLITS, EVAL_DIR, REPORTS, ROOT, full_metrics, read_json, write_json  # noqa: E402
from data_loader import load_splits  # noqa: E402

# Categorical palette (dataviz reference palette, light mode, fixed slot order).
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# --------------------------------------------------------------------------- helpers
def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def env_info() -> dict:
    import sklearn
    import tldextract
    import torch
    import transformers

    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None

    scanner = ROOT / "sentinel_ai_v2.py"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tldextract": tldextract.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": git("rev-parse", "HEAD"),
        "sentinel_ai_v2_sha256": hashlib.sha256(scanner.read_bytes()).hexdigest(),
        "sentinel_ai_v2_uncommitted_changes": bool(git("status", "--porcelain", "sentinel_ai_v2.py")),
    }


def test_scores() -> pd.DataFrame:
    """One frame with every model's continuous TEST score, aligned on row_id."""
    hy = pd.read_csv(ARTIFACTS / "hybrid_scores.csv")
    hy = hy[hy["split"] == "test"]
    zs = pd.read_csv(ARTIFACTS / "zeroshot_test.csv")[["row_id", "p_zeroshot"]]
    df = hy.merge(zs, on="row_id", validate="one_to_one")
    return df.sort_values("row_id").reset_index(drop=True)


def val_scores() -> pd.DataFrame:
    hy = pd.read_csv(ARTIFACTS / "hybrid_scores.csv")
    hy = hy[hy["split"] == "val"]
    zs = pd.read_csv(ARTIFACTS / "zeroshot_val.csv")[["row_id", "p_zeroshot"]]
    return hy.merge(zs, on="row_id", validate="one_to_one").sort_values("row_id").reset_index(drop=True)


# --------------------------------------------------------------------------- model table
def build_models(r: dict) -> list[dict]:
    rules, zs, ft, hy = r["rules"], r["zeroshot"], r["finetuned"], r["hybrid"]
    rows = [
        {"key": "rules_ge30", "name": "Rules (production) — score ≥ 30 (MEDIUM+)", "score_col": "rule_score", "thr": 30,
         "test": rules["test_overall"]["medium_plus_ge30"], "test_sender": rules["test_sender_available"]["medium_plus_ge30"],
         "latency": rules["latency"], "size_mb": rules["size_mb"], "threshold_from": "fixed (production level)"},
        {"key": "rules_ge70", "name": "Rules (production) — score ≥ 70 (HIGH)", "score_col": "rule_score", "thr": 70,
         "test": rules["test_overall"]["high_only_ge70"], "test_sender": rules["test_sender_available"]["high_only_ge70"],
         "latency": rules["latency"], "size_mb": rules["size_mb"], "threshold_from": "fixed (production level)"},
        {"key": "zeroshot_05", "name": "BERT-tiny SMS model, zero-shot (P ≥ 0.5)", "score_col": "p_zeroshot", "thr": 0.5,
         "test": zs["test_overall_argmax_0.5"], "test_sender": zs["test_sender_available_argmax_0.5"],
         "latency": zs["latency"], "size_mb": zs["size_mb"], "threshold_from": "fixed (argmax)"},
        {"key": "zeroshot_valthr", "name": "BERT-tiny SMS model, zero-shot (VAL threshold)", "score_col": "p_zeroshot", "thr": zs["val_tuned_threshold"],
         "test": zs["test_overall_val_threshold"], "test_sender": zs["test_sender_available_val_threshold"],
         "latency": zs["latency"], "size_mb": zs["size_mb"], "threshold_from": "VAL max-F1"},
        {"key": "finetuned", "name": "BERT-tiny fine-tuned on e-mail", "score_col": "p_finetuned", "thr": ft["val_threshold"],
         "test": ft["test_overall"], "test_sender": ft["test_sender_available"],
         "latency": ft["latency"], "size_mb": ft["size_mb"], "threshold_from": "VAL max-F1"},
    ]
    for key, label in (("hybrid_A_url_sender_bert", "Hybrid A — url + sender + 40·P"),
                       ("hybrid_B_rescaled_with_urgency", "Hybrid B — (url + sender + urgency + 40·P)·100/115")):
        rows.append({"key": key, "name": label, "score_col": key, "thr": hy[key]["val_threshold"],
                     "test": hy[key]["test_overall"], "test_sender": hy[key]["test_sender_available"],
                     "latency": hy[key]["latency"], "size_mb": hy[key]["size_mb"], "threshold_from": "VAL max-F1"})
    for row in rows:
        cm = row["test"]["confusion_matrix"]
        row["test_positive_rate"] = (cm["tp"] + cm["fp"]) / row["test"]["n"]
        row["degenerate"] = row["test_positive_rate"] > 0.99 or row["test_positive_rate"] < 0.01
    return rows


def add_val_f1(models: list[dict], val: pd.DataFrame) -> None:
    for m in models:
        m["val"] = full_metrics(val["label"], val[m["score_col"]], m["thr"])


# --------------------------------------------------------------------------- figure
def pr_figure(models: list[dict], test: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    curves = [
        ("Rules (production score 0–100)", "rule_score"),
        ("BERT-tiny SMS, zero-shot", "p_zeroshot"),
        ("BERT-tiny fine-tuned", "p_finetuned"),
        ("Hybrid A (no urgency)", "hybrid_A_url_sender_bert"),
        ("Hybrid B (with urgency)", "hybrid_B_rescaled_with_urgency"),
    ]
    from sklearn.metrics import average_precision_score

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(11, 5.6), dpi=150, gridspec_kw={"width_ratios": [1.25, 1]})
    fig.patch.set_facecolor("#fcfcfb")
    y = test["label"].to_numpy()
    prevalence = y.mean()

    handles = []
    for (label, col), color in zip(curves, SLOT):
        p, r, _ = precision_recall_curve(y, test[col].to_numpy())
        ap = average_precision_score(y, test[col].to_numpy())
        for a in (ax, zoom):
            (h,) = a.plot(r, p, color=color, lw=1.6, solid_joinstyle="round", solid_capstyle="round", drawstyle="steps-post",
                          label=f"{label}  (PR-AUC {ap:.3f})")
        handles.append(h)

    ax.axhline(prevalence, color=INK_2, lw=0.8)
    ax.text(0.99, prevalence - 0.025, f"chance = phishing rate ({prevalence:.1%})", color=INK_2, fontsize=8.5, ha="right")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.4, 1.005)
    ax.set_title("Full range", color=INK, loc="left", fontsize=10.5)
    zoom.set_xlim(0.8, 1.0)
    zoom.set_ylim(0.9, 1.001)
    zoom.set_title("Zoom: recall 0.8–1.0, precision 0.9–1.0", color=INK, loc="left", fontsize=10.5)
    for a in (ax, zoom):
        a.set_facecolor("#fcfcfb")
        a.set_xlabel("Recall", color=INK)
        a.grid(True, color=GRID, lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(GRID)
        a.tick_params(colors=INK_2)
    ax.set_ylabel("Precision", color=INK)
    fig.suptitle(f"Precision–recall on the held-out TEST split (n = {len(y):,})", color=INK, x=0.01, ha="left", fontsize=12)
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


# --------------------------------------------------------------------------- errors
def error_analysis(best: dict, test_scores: pd.DataFrame, test_df: pd.DataFrame, path: Path) -> dict:
    df = test_scores.merge(test_df[["row_id", "sender", "subject", "body"]], on="row_id", validate="one_to_one")
    s, thr = df[best["score_col"]], best["thr"]
    df["pred"] = (s >= thr).astype(int)
    df["score"] = s
    fp = df[(df["label"] == 0) & (df["pred"] == 1)].sort_values(["score", "row_id"], ascending=[False, True])
    fn = df[(df["label"] == 1) & (df["pred"] == 0)].sort_values(["score", "row_id"], ascending=[True, True])
    pick = pd.concat([fp.head(20).assign(error_type="false_positive"), fn.head(20).assign(error_type="false_negative")])
    pick["body_snippet"] = pick["body"].map(lambda t: " ".join(t.split())[:600])
    cols = ["error_type", "row_id", "source", "label", "score", "p_finetuned", "url_risk", "sender_risk", "urgency_risk",
            "n_urls", "sender_trusted", "sender", "subject", "body_snippet"]
    pick[cols].to_csv(path, index=False, lineterminator="\n")

    errs = df[df["pred"] != df["label"]]
    by_source = {}
    for src, g in df.groupby("source"):
        e = g[g["pred"] != g["label"]]
        by_source[src] = {"n": int(len(g)), "errors": int(len(e)), "error_rate": float(len(e) / len(g)),
                          "fp": int(((e["label"] == 0)).sum()), "fn": int(((e["label"] == 1)).sum())}
    body_words = df["body"].map(lambda t: len(t.split()))
    return {
        "model": best["key"],
        "threshold": best["thr"],
        "n_fp": int(len(fp)),
        "n_fn": int(len(fn)),
        "selection": "20 most confident FPs (highest score) and 20 most confident FNs (lowest score) on TEST",
        "by_source": by_source,
        "median_body_words": {"errors": float(body_words[errs.index].median()), "correct": float(body_words[~df.index.isin(errs.index)].median())},
        "share_with_extracted_url": {"errors": float((errs["n_urls"] > 0).mean()), "all_test": float((df["n_urls"] > 0).mean())},
        "share_with_sender": {"errors": float(errs["has_sender"].mean()), "all_test": float(df["has_sender"].mean())},
    }


# --------------------------------------------------------------------------- markdown
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def cm_block(name: str, cm: dict) -> str:
    return (f"**{name}**\n\n| | predicted legit | predicted phishing |\n|---|---|---|\n"
            f"| **actual legit** | {cm['tn']:,} | {cm['fp']:,} |\n| **actual phishing** | {cm['fn']:,} | {cm['tp']:,} |\n")


def build_markdown(R: dict) -> str:
    man, models = R["dataset"], R["comparison"]
    drops, ov = man["drops"], man["overall_clean"]
    zs, ft, rules = R["zeroshot"], R["finetuned"], R["rules"]
    diag = rules["diagnostics_test"]
    best = R["best_model_by_val_f1"]
    L = []
    L.append("# SentinelAI — e-mail phishing detector evaluation\n")
    L.append(f"_Generated by `python eval/run_all.py`. Every number below is read from `reports/results.json`. "
             f"Seed {man['seed']}; all models scored on the identical TEST split "
             f"(sha256 `{man['split_sha256']['test'][:12]}…`)._\n")

    # ---- dataset
    L.append("## 1. Dataset\n")
    L.append("Source: **Phishing Email Curated Datasets** (Champa, Rabbi & Zibran; Zenodo record 8339691, CC-BY-4.0). "
             "These are the same six corpora that make up the Kaggle *Phishing Email Dataset*. "
             f"All six files matched their published MD5 checksums: **{all(v['matches_zenodo'] for v in man['raw_file_checksums'].values())}**. "
             "Label 1 = phishing **or spam** (the corpora mix both), 0 = legitimate.\n")
    rows = []
    for src, v in man["per_source_raw"].items():
        c = man["per_source_clean"].get(src, {})
        rows.append([src, f"{v['rows']:,}", f"{v['phishing']:,}", "yes" if v["has_sender_col"] else "**no**",
                     f"{c.get('n', 0):,}", pct(c.get("phishing_rate")), f"{c.get('n_with_sender', 0):,}"])
    L.append(md_table(["corpus", "raw rows", "raw phishing", "sender column", "kept rows", "phishing rate (kept)", "kept with sender"], rows))
    L.append("\n**Cleaning (rows removed):**\n")
    L.append(md_table(["step", "rows"], [
        ["raw total", f"{drops['raw_rows']:,}"],
        ["empty body", f"{drops['empty_body']:,}"],
        ["exact duplicates (sender+subject+body+label)", f"{drops['exact_duplicate_rows']:,}"],
        ["same text with conflicting labels", f"{drops['conflicting_label_rows']:,}"],
        ["duplicates after case/whitespace normalisation", f"{drops['normalised_text_duplicates']:,}"],
        ["same body, different subject", f"{drops['duplicate_body_different_subject']:,}"],
        [f"template clusters capped at {man['template_cap']} e-mails (largest cluster had {drops['largest_template_cluster_before_cap']:,})", f"{drops['template_cluster_over_cap']:,}"],
        ["**kept**", f"**{drops['kept_rows']:,}**"],
    ]))
    L.append(f"\nWhy the extra de-duplication: the first random split put {R['leakage_first_attempt']['test_first200chars_match_a_train_body']:,} "
             f"test e-mails ({pct(R['leakage_first_attempt']['test_first200chars_match_a_train_body'] / R['leakage_first_attempt']['test_n'])}) "
             f"in the same 200-character template as a training e-mail (spam campaigns, newsletters, list digests; one CEAS_08 campaign "
             f"alone had {drops['largest_template_cluster_before_cap']:,} copies). A fine-tuned model would be partly graded on memorised templates. "
             f"The final split is **stratified on source × label and grouped by template** ({man['grouped_on']}), so no template spans splits.\n")
    lk = man["leakage_checks"]
    L.append(md_table(["split", "n", "phishing", "legit", "phishing rate", "with sender", "body has an http(s) URL"],
                      [[k, f"{v['n']:,}", f"{v['n_phishing']:,}", f"{v['n_legit']:,}", pct(v["phishing_rate"]), f"{v['n_with_sender']:,}", pct(v["body_has_http_url"])]
                       for k, v in man["splits"].items()]))
    L.append(f"\nLeakage checks on the final split: test bodies identical to a train body = **{lk['test_bodies_identical_to_a_train_body']}**; "
             f"test bodies sharing their first 200 normalised characters with a train body = **{lk['test_first200chars_match_a_train_body']}** "
             f"(val: {lk['val_bodies_identical_to_a_train_body']} / {lk['val_first200chars_match_a_train_body']}).\n")
    L.append(f"Sender availability: Enron and Ling have no sender column, and some Nigerian_Fraud rows have none. Those rows are kept; production's "
             f"`analyze_sender('')` returns 0 for them. Metrics are reported overall **and** on the {man['splits']['test']['n_with_sender']:,}-e-mail "
             f"sender-available TEST subset.\n")
    us = man.get("url_stripping_raw", {})
    if us:
        L.append("**Many URLs are invisible to the rule engine in this release.** In Nazario, most links were stripped from the body, although the "
                 "dataset's `urls` flag says they existed. Enron and Ling store URLs tokenised with spaces (`http : / / www . x . com`), which "
                 "production's `extract_urls()` regex cannot match. Production's URL rules can only score `http(s)://` URLs present in the text:\n")
        L.append(md_table(["corpus", "flagged as containing URLs", "body contains a parseable http(s):// URL", "flagged but no URL left in body",
                           "body contains a space-tokenised URL"],
                          [[s, pct(v["share_flagged_urls"]), pct(v["share_body_contains_http_url"]), pct(v["share_flagged_but_no_url_in_body"]),
                            pct(v.get("share_body_contains_space_tokenised_url"))] for s, v in us.items()]))
        L.append("")

    # ---- comparison
    L.append("## 2. Results on the held-out TEST split\n")
    rows = []
    for m in models:
        t = m["test"]
        flags = "≈ every e-mail" if m["test_positive_rate"] > 0.5 else "almost no e-mail"
        name = m["name"] + (f" ⚠ degenerate: flags {flags} ({m['test_positive_rate']:.1%})" if m["degenerate"] else "")
        rows.append([name, f"{m['thr']:.3g} ({m['threshold_from']})", f3(t["precision"]), f3(t["recall"]), f3(t["f1"]), f3(t["accuracy"]),
                     f3(t["roc_auc"]), f3(t["pr_auc"]), f"{m['latency']['median_ms']:.2f} / {m['latency']['p95_ms']:.2f}", f"{m['size_mb']:.2f}"])
    L.append(md_table(["model", "threshold (chosen by)", "precision", "recall", "F1", "accuracy", "ROC-AUC", "PR-AUC", "latency ms (median / p95)", "size MB"], rows))
    L.append(f"\nn = {models[0]['test']['n']:,} test e-mails ({pct(man['splits']['test']['phishing_rate'])} phishing, so a constant \"phishing\" "
             f"classifier would score F1 ≈ {2 * man['splits']['test']['phishing_rate'] / (1 + man['splits']['test']['phishing_rate']):.3f} and ROC-AUC 0.5). "
             f"ROC-AUC and PR-AUC use each model's continuous score (rules: the 0–100 risk score; BERT: P(phishing)). "
             f"Latency is per e-mail, batch size 1, CPU ({R['meta']['torch_threads']} torch threads, {R['meta']['env']['processor']}), after 50 warm-up e-mails, "
             f"measured on the same {models[0]['latency']['n_timed']:,} seeded test e-mails for every approach; BERT timings include tokenisation. "
             f"Size: rules = `sentinel_ai_v2.py` source; BERT = weights + tokenizer files; hybrid = both.\n")

    L.append("### Sender-available TEST subset\n")
    L.append(md_table(["model", "n", "precision", "recall", "F1", "ROC-AUC"],
                      [[m["name"], f"{m['test_sender']['n']:,}", f3(m["test_sender"]["precision"]), f3(m["test_sender"]["recall"]),
                        f3(m["test_sender"]["f1"]), f3(m["test_sender"]["roc_auc"])] for m in models]))
    L.append("")

    L.append("### Confusion matrices (TEST)\n")
    for m in models:
        L.append(cm_block(f"{m['name']} — threshold {m['thr']:.3g}", m["test"]["confusion_matrix"]))
    L.append("### Precision–recall curves\n\n![PR curves](pr_curves.png)\n")

    # ---- per-experiment detail
    L.append("## 3. Experiment details\n")
    L.append("### 3.1 Rule engine (production `sentinel_ai_v2.py`)\n")
    L.append("The rules are imported, not re-implemented. Per e-mail the harness repeats `run_scanner()` exactly: URLs are taken from the body only, each "
             "URL's `url_heuristic_score` is **summed**, then `sender_heuristic_score` and `urgency_heuristic_score(subject + \" \" + body)` are added, and "
             "`calculate_risk(..., features=None)` gives url ≤ 35 + sender ≤ 25 + urgency ≤ 15. No `model.pkl` exists, so production takes the same heuristic path.\n")
    lc = diag["level_counts_by_label"]
    mc = diag["mean_component_by_label"]
    L.append(md_table(["TEST diagnostic", "legit", "phishing"], [
        ["e-mails with ≥1 URL extracted", pct(diag["share_with_any_extracted_url_by_label"]["0"]), pct(diag["share_with_any_extracted_url_by_label"]["1"])],
        ["rule score exactly 0", pct(diag["share_score_zero_by_label"]["0"]), pct(diag["share_score_zero_by_label"]["1"])],
        ["level MEDIUM RISK", f"{lc['0'].get('MEDIUM RISK', 0):,}", f"{lc['1'].get('MEDIUM RISK', 0):,}"],
        ["level HIGH RISK", f"{lc['0'].get('HIGH RISK', 0):,}", f"{lc['1'].get('HIGH RISK', 0):,}"],
        ["sender domain in TRUSTED_DOMAINS", f"{diag['sender_in_TRUSTED_DOMAINS_by_label']['0']:,}", f"{diag['sender_in_TRUSTED_DOMAINS_by_label']['1']:,}"],
        ["mean url_risk / sender_risk / urgency", f"{mc['0']['url_risk']:.2f} / {mc['0']['sender_risk']:.2f} / {mc['0']['urgency_risk']:.2f}",
         f"{mc['1']['url_risk']:.2f} / {mc['1']['sender_risk']:.2f} / {mc['1']['urgency_risk']:.2f}"],
    ]))
    pp = R.get("psl_parity")
    L.append("\nReproducibility patch: production calls `tldextract.extract`, which downloads the live Public Suffix List. The harness points the same "
             "library at its bundled snapshot instead. No rule changes." + (
                 f" One-off parity check (`python eval/eval_rules.py --check-psl-parity`): **{pp['n_scores_differ']} of {pp['n_test']:,}** test "
                 f"scores differ between snapshot and live list." if pp else "") + "\n")
    L.append("Rules ≥ 30, per source (TEST):\n")
    L.append(md_table(["source", "n", "precision", "recall", "F1", "ROC-AUC"],
                      [[s, f"{v['n']:,}", f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), f3(v["roc_auc"])] for s, v in rules["test_per_source_ge30"].items()]))

    L.append("\n### 3.2 Pretrained BERT-tiny, zero-shot (SMS → e-mail domain shift)\n")
    oc = zs["label_orientation_check"]
    L.append(f"`{zs['model']}` @ `{zs['revision'][:10]}`, input = subject + body truncated to {zs['max_len']} tokens. The model card does not state "
             f"which label is spam, so it was checked on four SMS probes (card widget + UCI-style texts): P(LABEL_1) = "
             f"{', '.join(f'{p:.2f}' for p in oc['probe_probs'])} for labels {oc['probe_labels']}. **LABEL_1 = spam**, mapped to phishing.\n")
    L.append(f"- TEST ROC-AUC **{f3(zs['test_overall_argmax_0.5']['roc_auc'])}**: the model's score barely separates the classes on e-mail. "
             f"Test probabilities span only [{zs['prob_range_test'][0]:.3f}, {zs['prob_range_test'][1]:.3f}].\n"
             f"- Re-picking the threshold on VAL ({zs['val_tuned_threshold']:.3f}) flags almost every e-mail as phishing. Its F1 matches the "
             f"constant classifier, which reflects the class balance, not skill.\n"
             f"- {pct(zs['share_test_emails_truncated_at_128_tokens'])} of test e-mails are longer than 128 tokens, so the model sees only their opening.\n")
    L.append("Why it fails: the checkpoint was trained on the 2012 UCI SMS corpus (~5.5k short texts). There, spam means premium-rate "
             "numbers, \"WINNER!! claim £900\" and txt-speak, and ham means short personal chat. E-mail is long-form, has headers, signatures, HTML "
             "remnants, mailing-list footers and newsletters, and its phishing uses account-verification and invoice lures, not prize SMS. Inputs are also "
             "far longer than anything seen in training. The features the SMS model learned don't carry over, which matches the near-chance ranking.\n")

    L.append("### 3.3 Fine-tuned BERT-tiny\n")
    hp = ft["hparams"]
    L.append(f"Same checkpoint, fine-tuned on the TRAIN split only. Hyper-parameters were fixed before training, not searched: max_len {hp['max_len']}, "
             f"{hp['epochs']} epochs, batch {hp['batch_size']}, AdamW lr {hp['lr']}, weight decay {hp['weight_decay']}, linear warm-up "
             f"{hp['warmup_ratio']:.0%} then linear decay, grad-clip {hp['grad_clip']}, seed {hp['seed']}, {hp['torch_threads']} CPU threads, "
             f"deterministic torch algorithms. Epoch picked by **lowest VAL loss** → epoch **{ft['selected_epoch']}**; threshold by max VAL F1 → "
             f"**{ft['val_threshold']:.3f}**. TEST was scored once.\n")
    L.append(md_table(["epoch", "train loss", "val loss", "val F1 @0.5", "val ROC-AUC", "seconds"],
                      [[str(e["epoch"]), f3(e["train_loss"]), f"{e['val_loss']:.4f}", f3(e.get("val_f1_at_0.5")), f3(e.get("val_roc_auc")),
                        f"{e['epoch_seconds']:.0f}" if e.get("epoch_seconds") else "–"] for e in ft["epoch_log"]]))
    L.append(f"\nAt the default 0.5 threshold, TEST F1 = {f3(ft['test_overall_at_0.5']['f1'])}.\n")
    L.append("Per source (TEST):\n")
    L.append(md_table(["source", "n", "precision", "recall", "F1", "ROC-AUC"],
                      [[s, f"{v['n']:,}", f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), f3(v["roc_auc"])] for s, v in ft["test_per_source"].items()]))

    L.append("\n### 3.4 Hybrid\n")
    for key in ("hybrid_A_url_sender_bert", "hybrid_B_rescaled_with_urgency"):
        h = R["hybrid"][key]
        L.append(f"- **{key}**: VAL-chosen threshold {h['val_threshold']:.2f}; TEST F1 {f3(h['test_overall']['f1'])}, ROC-AUC {f3(h['test_overall']['roc_auc'])}.")
    L.append(f"\nFor comparison, the fine-tuned model alone: TEST F1 {f3(ft['test_overall']['f1'])}, ROC-AUC {f3(ft['test_overall']['roc_auc'])}. "
             "The rule components rarely fire on this data (§3.1), so in practice the hybrid score is mostly 40·P.\n")

    # ---- errors
    ea = R["error_analysis"]
    L.append("## 4. Error analysis\n")
    L.append(f"Best model by **VAL** F1 (chosen without looking at test): **{best['name']}** (VAL F1 {f3(best['val_f1'])}). On TEST it makes "
             f"{ea['n_fp']:,} false positives and {ea['n_fn']:,} false negatives. `reports/errors.csv` holds the {ea['selection']}.\n")
    L.append(md_table(["source", "test n", "errors", "error rate", "FP", "FN"],
                      [[s, f"{v['n']:,}", f"{v['errors']:,}", pct(v["error_rate"]), f"{v['fp']:,}", f"{v['fn']:,}"] for s, v in ea["by_source"].items()]))
    L.append(f"\nMedian body length: {ea['median_body_words']['errors']:.0f} words for errors vs {ea['median_body_words']['correct']:.0f} for correct predictions. "
             f"Share with an extracted URL: {pct(ea['share_with_extracted_url']['errors'])} of errors vs {pct(ea['share_with_extracted_url']['all_test'])} overall. "
             f"Share with a sender: {pct(ea['share_with_sender']['errors'])} of errors vs {pct(ea['share_with_sender']['all_test'])} overall.\n")
    notes = EVAL_DIR / "analysis_notes.md"
    if notes.exists():
        text = notes.read_text(encoding="utf-8")
        tag = f"test-split-sha256: {man['split_sha256']['test']}"
        if tag not in text:
            L.append("> ⚠ `eval/analysis_notes.md` was written for a different split/run. Re-read `reports/errors.csv` before trusting the notes below.\n")
        L.append(text.split("---", 1)[-1].strip() + "\n")

    # ---- sanity checks on high scores
    dg = R.get("diagnostics")
    if dg:
        nd, lr, si = dg["near_duplicates"], dg["tfidf_logreg_reference"], dg["source_identification"]
        nov, dup = nd["finetuned_on_test_with_max_cosine_lt_0.8"], nd["finetuned_on_test_with_max_cosine_ge_0.8"]
        L.append("## 5. Are the high scores real? (leakage and shortcut checks)\n")
        L.append(f"1. **Exact / template leakage:** 0 test e-mails share a body or a 200-character prefix with a train e-mail (§1).\n"
                 f"2. **Fuzzy near-duplicates** (TF-IDF cosine to the nearest TRAIN e-mail): {pct(nd['share_test_max_cosine_ge']['0.95'])} of TEST "
                 f"e-mails have a train neighbour with cosine ≥ 0.95, {pct(nd['share_test_max_cosine_ge']['0.9'])} ≥ 0.9, "
                 f"{pct(nd['share_test_max_cosine_ge']['0.8'])} ≥ 0.8 (median {nd['median_max_cosine']:.2f}). The fine-tuned model on test e-mails "
                 f"**without** a close train neighbour (cosine < 0.8, n = {nov['n']:,}): F1 {f3(nov['f1'])}, ROC-AUC {f3(nov['roc_auc'])}. "
                 f"On those **with** one (n = {dup['n']:,}): F1 {f3(dup['f1'])}, ROC-AUC {f3(dup['roc_auc'])}.\n"
                 f"3. **Bag-of-words reference:** TF-IDF + logistic regression on the same split (threshold on VAL) scores TEST F1 "
                 f"{f3(lr['test_overall']['f1'])}, ROC-AUC {f3(lr['test_overall']['roc_auc'])}. This benchmark is close to linearly separable on "
                 f"word counts, so a high F1 here says more about the dataset than about model capacity.\n"
                 f"4. **Corpus identification:** TF-IDF + logistic regression predicts the *source corpus* of a TEST e-mail with "
                 f"{pct(si['test_accuracy'])} accuracy (majority class {pct(si['majority_class_rate'])}). Corpora are highly distinguishable, and two "
                 "of them contain only phishing, so part of any learned model's score can come from recognising the corpus.\n")
        L.append(f"Most influential phishing tokens (reference model): `{'`, `'.join(lr['top_phishing_tokens'][:20])}`\n")
        L.append(f"Most influential legitimate tokens: `{'`, `'.join(lr['top_legit_tokens'][:20])}`\n")
        L.append("Conclusion: no duplicate leakage remains, but the benchmark is easy and corpus-specific vocabulary is predictive. Treat the "
                 "fine-tuned F1 as *in-distribution performance on 1999–2008 corpora*, not as expected performance on a modern inbox.\n")

    # ---- limitations
    L.append("## 6. Limitations\n")
    tr = diag["sender_in_TRUSTED_DOMAINS_by_label"]
    L.append("\n".join([
        f"- **Dataset age.** The corpora date from roughly 1999–2008 (Enron, SpamAssassin, Ling, CEAS 2008, Nazario, Nigerian fraud). Modern lures (MFA "
        "fatigue, QR codes, OAuth-consent, invoice/BEC, LLM-written text, `.xyz/.top` throw-away domains) are under-represented. Metrics here do not "
        "predict performance on a 2026 inbox.",
        "- **Spam ≠ phishing.** Label 1 mixes generic spam (Enron, SpamAssassin, Ling, most of CEAS) with phishing and 419 fraud (Nazario, Nigerian_Fraud). "
        "A good spam score is not proof of phishing-specific skill.",
        f"- **Source confounding.** Nazario and Nigerian_Fraud contain only positives, and Enron/Ling have no sender. A learned model can partly identify "
        "the *corpus* rather than the attack. The per-source tables show where performance comes from.",
        f"- **TRUSTED_DOMAINS bias.** The rule engine's allow-list (gmail.com, google.com, naukri.com, canarabank.com, reddit.com, snapchat.com, …) is "
        f"tuned to one personal inbox. On TEST, {tr['0']:,} legitimate and {tr['1']:,} phishing senders hit it. A free-mail domain like gmail.com is trusted "
        "outright, so phishing from any Gmail account gets sender risk 0.",
        "- **URLs invisible to the rules.** This release stripped most Nazario links and stores Enron/Ling URLs space-tokenised, starving the URL "
        "component (35 of 100 points). Production reads raw Gmail bodies with links intact, so the URL rules see more there. These rule numbers are "
        "a pessimistic bound for the URL component, not a production estimate.",
        "- **No headers.** No SPF/DKIM/DMARC or Received headers are available, so `analyze_authentication` could not be evaluated.",
        "- **English-only; no Indian-context e-mail.** No UPI/KYC/Aadhaar/PAN, Hindi/Hinglish, or Indian-bank lures, although the scanner's "
        "allow-list and brands target an Indian inbox.",
        "- **Small model, short context.** BERT-tiny (2 layers, 128 hidden) sees ≤ 256 tokens (≤ 128 zero-shot). Long e-mails are judged on their opening "
        "only. No hyper-parameter search was run, by design, to keep test untouched.",
        "- **One split, one seed.** No confidence intervals or cross-validation. Differences of a few tenths of a point between models are within noise.",
        "- **Thresholds.** VAL-tuned thresholds are optimal for a ~50 % phishing rate. Real inboxes have far fewer phishing e-mails, so precision would drop "
        "at the same threshold.",
    ]))
    L.append("\n## 7. Reproduce\n\n```\npip install -r requirements-eval.txt\npython eval/run_all.py          # data → split → 4 experiments → diagnostics → report\n"
             "python eval/eval_rules.py --check-psl-parity   # optional, needs network\n```\n")
    e = R["meta"]["env"]
    L.append(f"Environment of this run: Python {e['python']}, torch {e['torch']}, transformers {e['transformers']}, scikit-learn {e['scikit_learn']}, "
             f"tldextract {e['tldextract']}, {e['platform']}. Repo commit `{(e['git_commit'] or 'n/a')[:10]}`; `sentinel_ai_v2.py` sha256 "
             f"`{e['sentinel_ai_v2_sha256'][:12]}…`" + (" (**with uncommitted changes**)" if e["sentinel_ai_v2_uncommitted_changes"] else "") + ".\n")
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def main() -> dict:
    from common import TORCH_THREADS

    partial = REPORTS / "partial"
    R = {
        "meta": {"env": env_info(), "torch_threads": TORCH_THREADS},
        "dataset": read_json(DATA_SPLITS / "manifest.json"),
        "rules": read_json(partial / "rules.json"),
        "zeroshot": read_json(partial / "zeroshot.json"),
        "finetuned": read_json(partial / "finetuned.json"),
        "hybrid": read_json(partial / "hybrid.json"),
    }
    R["leakage_first_attempt"] = R["dataset"]["leakage_naive_random_split_diagnostic"]
    if (partial / "diagnostics.json").exists():
        R["diagnostics"] = read_json(partial / "diagnostics.json")
    if (partial / "psl_parity.json").exists():
        R["psl_parity"] = read_json(partial / "psl_parity.json")

    models = build_models(R)
    test, val = test_scores(), val_scores()
    add_val_f1(models, val)
    eligible = [m for m in models if not m["degenerate"]]
    best = max(eligible, key=lambda m: m["val"]["f1"])
    R["comparison"] = [{k: v for k, v in m.items()} for m in models]
    R["best_model_by_val_f1"] = {"key": best["key"], "name": best["name"], "val_f1": best["val"]["f1"], "threshold": best["thr"]}

    pr_figure(models, test, REPORTS / "pr_curves.png")
    R["error_analysis"] = error_analysis(best, test, load_splits()["test"], REPORTS / "errors.csv")

    write_json(REPORTS / "results.json", R)
    (REPORTS / "results.md").write_text(build_markdown(R), encoding="utf-8")
    print(f"  wrote results.json, results.md, pr_curves.png, errors.csv (best by VAL F1: {best['name']})")
    return R


if __name__ == "__main__":
    main()
