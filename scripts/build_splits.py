#!/usr/bin/env python3
"""Build print-level train/val/test/eval splits (Milestone 2, step 2 of 3).

Reads labels.csv. Drops failure rows with empty failure_at_frame (unlabeled
failures can't be used for training because we can't tell which of their
frames are healthy vs failure). Assigns each remaining print_id to one of
train/val/test/eval using a (printer_id, outcome) stratified shuffle.

Output: splits.json at the repo root, deterministic given --seed.

Usage:

    python scripts/build_splits.py
    python scripts/build_splits.py --seed 13 --train 0.70 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=Path("labels.csv"), type=Path)
    p.add_argument("--out", default=Path("splits.json"), type=Path)
    p.add_argument("--train", default=0.70, type=float)
    p.add_argument("--val",   default=0.15, type=float)
    p.add_argument("--test",  default=0.15, type=float)
    p.add_argument("--eval-fraction-of-test", default=0.10, type=float,
                   help="fraction of test pulled out as eval (target ~10-20 prints, "
                        "hand-curated separately)")
    p.add_argument("--seed",  default=42, type=int)
    args = p.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(f"ERROR: train+val+test must sum to 1.0 (got {args.train + args.val + args.test})")
        return 2

    with open(args.labels, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Drop failures with missing failure_at_frame
    kept = []
    dropped_unlabeled = []
    for r in rows:
        if r["outcome"] == "failure" and not (r.get("failure_at_frame") or "").strip():
            dropped_unlabeled.append(r["print_id"])
            continue
        kept.append(r)

    print(f"Total rows in labels.csv:        {len(rows)}")
    print(f"Dropped (unlabeled failures):    {len(dropped_unlabeled)}")
    if dropped_unlabeled:
        print(f"  {', '.join(dropped_unlabeled[:6])}{'...' if len(dropped_unlabeled) > 6 else ''}")
    print(f"Kept for splitting:              {len(kept)}")
    print()

    pids    = [r["print_id"] for r in kept]
    strata  = [f"{r['printer_id']}|{r['outcome']}" for r in kept]

    stratum_counts = Counter(strata)
    print("Stratum sizes (printer × outcome):")
    for k, n in sorted(stratum_counts.items()):
        warn = " ⚠ small" if n < 4 else ""
        print(f"  {k:<28} {n}{warn}")
    print()

    # Split off (val + test) from train
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=args.val + args.test, random_state=args.seed
    )
    train_idx, rest_idx = next(sss1.split(pids, strata))

    # Split rest into val vs test
    rest_strata = [strata[i] for i in rest_idx]
    test_frac_of_rest = args.test / (args.val + args.test)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_frac_of_rest, random_state=args.seed
    )
    val_local, test_local = next(sss2.split([pids[i] for i in rest_idx], rest_strata))
    val_idx  = [rest_idx[i] for i in val_local]
    test_idx = [rest_idx[i] for i in test_local]

    # Pull eval out of test. Use OUTCOME-only stratification (success vs failure)
    # because some (printer × outcome) strata land with only 1 print in test and
    # StratifiedShuffleSplit requires >= 2 per class. Eval is hand-curated for
    # hard cases anyway, so printer-level representation isn't critical here.
    test_outcomes = [
        next(r for r in kept if r["print_id"] == pids[i])["outcome"]
        for i in test_idx
    ]
    try:
        sss3 = StratifiedShuffleSplit(
            n_splits=1, test_size=args.eval_fraction_of_test, random_state=args.seed
        )
        test_only_local, eval_local = next(
            sss3.split([pids[i] for i in test_idx], test_outcomes)
        )
    except ValueError:
        # Last-ditch fallback: non-stratified shuffle. Should be unreachable for
        # any non-pathological dataset, but doesn't crash on edge cases.
        import random
        rng = random.Random(args.seed)
        n_eval = max(1, round(len(test_idx) * args.eval_fraction_of_test))
        order = list(range(len(test_idx)))
        rng.shuffle(order)
        eval_local = order[:n_eval]
        test_only_local = order[n_eval:]
    eval_idx = [test_idx[i] for i in eval_local]
    test_idx = [test_idx[i] for i in test_only_local]

    splits = {
        "train": sorted(pids[i] for i in train_idx),
        "val":   sorted(pids[i] for i in val_idx),
        "test":  sorted(pids[i] for i in test_idx),
        "eval":  sorted(pids[i] for i in eval_idx),
        "_meta": {
            "seed": args.seed,
            "ratios": {"train": args.train, "val": args.val,
                       "test": args.test, "eval_fraction_of_test": args.eval_fraction_of_test},
            "dropped_unlabeled_failures": dropped_unlabeled,
            "stratified_by": "printer_id × outcome",
        },
    }

    args.out.write_text(json.dumps(splits, indent=2))

    # Summary by split, broken down by class
    row_by_pid = {r["print_id"]: r for r in kept}
    print("Split sizes (healthy / failure):")
    for name in ("train", "val", "test", "eval"):
        ids = splits[name]
        h = sum(1 for pid in ids if row_by_pid[pid]["outcome"] == "success")
        f = sum(1 for pid in ids if row_by_pid[pid]["outcome"] == "failure")
        print(f"  {name:6s}: {len(ids):4d}   ({h} healthy / {f} failure)")
    print()

    # Failure-type breakdown across splits (helps sanity-check stage 2 viability)
    print("Failure-type by split:")
    failure_types = sorted({r["failure_type"] for r in kept if r["outcome"] == "failure"})
    print(f"  {'split':<6}  " + "  ".join(f"{ft:<14}" for ft in failure_types))
    for name in ("train", "val", "test", "eval"):
        counts = Counter(
            row_by_pid[pid]["failure_type"]
            for pid in splits[name]
            if row_by_pid[pid]["outcome"] == "failure"
        )
        line = f"  {name:<6}  " + "  ".join(f"{counts[ft]:<14}" for ft in failure_types)
        print(line)
    print()

    print(f"Saved → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
