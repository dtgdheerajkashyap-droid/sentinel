"""Run the whole SentinelAI e-mail evaluation end to end.

    python eval/run_all.py

1. data_loader         download/verify corpora, clean, one stratified group-aware 70/15/15 split
2. eval_rules          production rule engine (sentinel_ai_v2.py) on VAL + TEST
3. eval_bert_zeroshot  pretrained SMS BERT-tiny, zero-shot
4. finetune_bert       fine-tune BERT-tiny on TRAIN, select epoch/threshold on VAL, score TEST once
5. eval_hybrid         rule sub-scores + 40 * P(fine-tuned), threshold on VAL
6. diagnostics         near-duplicate / shortcut / corpus-identification checks
7. report              results.json, results.md, pr_curves.png, errors.csv

Steps run sequentially in one process so latency measurements never compete
for the CPU. Fixed seeds everywhere; the TEST split is never used for any choice.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore", category=FutureWarning)

import data_loader  # noqa: E402
import diagnostics  # noqa: E402
import eval_bert_zeroshot  # noqa: E402
import eval_hybrid  # noqa: E402
import eval_rules  # noqa: E402
import finetune_bert  # noqa: E402
import report  # noqa: E402
from common import ARTIFACTS, REPORTS, set_seeds  # noqa: E402

STEPS = [
    ("1/7 data", data_loader.main),
    ("2/7 rules", eval_rules.main),
    ("3/7 zero-shot BERT", eval_bert_zeroshot.main),
    ("4/7 fine-tune BERT", finetune_bert.main),
    ("5/7 hybrid", eval_hybrid.main),
    ("6/7 diagnostics", diagnostics.main),
    ("7/7 report", report.main),
]


def main() -> None:
    set_seeds()
    REPORTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    for name, fn in STEPS:
        print(f"\n== {name}", flush=True)
        t0 = time.perf_counter()
        set_seeds()  # each step starts from the same RNG state, regardless of what ran before
        fn()
        print(f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"\nDone in {(time.perf_counter() - t_all) / 60:.1f} min. See reports/results.md")


if __name__ == "__main__":
    main()
