"""Select models without test labels (CA-SUM protocol) and report F1 / rho / tau / mAP over the 5 splits.

    python -m vidsum.evaluate --exp casum_googlenet --dataset tvsum [--oracle]

Selection, per split (same as CA-SUM's evaluation/choose_best_model.py):
  1. For every sigma: epoch = argmin of the training loss over epochs [start_epoch, max_epoch).
  2. Improvement = |mean score(epoch) - mean score(untrained)| / |sigma - mean score(untrained)|,
     using only predicted scores, no labels. Values > 1.5 are discarded as overshooting.
  3. Keep the sigma with the highest improvement and report its metrics at the selected epoch.
--oracle additionally reports the best test F1 over all epochs. That number uses test labels for model
selection and is only a diagnostic (it is what the old mine/ runs reported); never put it in the thesis.
"""
import argparse
import json

import numpy as np

from .eval.metrics import evaluate_scores
from .paths import EXPERIMENTS

METRICS = ("f1", "rho", "tau", "map50", "map15")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--start_epoch", type=int, default=20)
    parser.add_argument("--max_epoch", type=int, default=400)
    parser.add_argument("--method", default="argmin", choices=["argmin", "last"])
    parser.add_argument("--oracle", action="store_true", help="Also report test-selected F1 (diagnostic only)")
    return parser.parse_args()


def load_scores(run_dir, epoch):
    with open(run_dir / "scores" / f"epoch_{epoch}.json", encoding="utf-8") as handle:
        return json.load(handle)


def select_epoch(run_dir, start_epoch, max_epoch, method):
    with open(run_dir / "history.json", encoding="utf-8") as handle:
        losses = [e["loss"] for e in json.load(handle)["epochs"]][:max_epoch]
    if method == "last" or len(losses) <= start_epoch:
        return len(losses) - 1
    return int(np.argmin(losses[start_epoch:])) + start_epoch


def improvement_score(run_dir, epoch, sigma):
    untrained = np.mean([np.mean(v) for v in load_scores(run_dir, -1).values()])
    current = np.mean([np.mean(v) for v in load_scores(run_dir, epoch).values()])
    return abs(current - untrained) / abs(sigma - untrained)


def oracle_f1(dataset, run_dir):
    epochs = sorted(int(p.stem.split("_")[1]) for p in (run_dir / "scores").glob("epoch_*.json"))
    return max(evaluate_scores(dataset, load_scores(run_dir, e))["f1"] for e in epochs if e >= 0)


def main():
    args = parse_args()
    base = EXPERIMENTS / args.exp / args.dataset
    sigma_dirs = sorted(base.glob("reg*"), key=lambda p: float(p.name[3:]))
    if not sigma_dirs:
        raise SystemExit(f"No runs under {base}")
    splits = sorted({int(p.name[5:]) for d in sigma_dirs for p in d.glob("split*") if (p / "history.json").exists()})

    rows = []
    for split in splits:
        candidates = []
        for sigma_dir in sigma_dirs:
            run_dir = sigma_dir / f"split{split}"
            if not (run_dir / "history.json").exists():
                continue
            sigma = float(sigma_dir.name[3:])
            epoch = select_epoch(run_dir, args.start_epoch, args.max_epoch, args.method)
            imp = improvement_score(run_dir, epoch, sigma)
            candidates.append((imp if imp <= 1.5 else 0.0, sigma, epoch, run_dir))
        imp, sigma, epoch, run_dir = max(candidates, key=lambda c: c[0])

        row = {"split": split, "sigma": sigma, "epoch": epoch, "improvement": float(imp),
               **evaluate_scores(args.dataset, load_scores(run_dir, epoch))}
        with open(run_dir / "history.json", encoding="utf-8") as handle:
            row["train_seconds"] = json.load(handle).get("train_seconds")
        if args.oracle:
            row["oracle_f1"] = oracle_f1(args.dataset, run_dir)
        rows.append(row)

        shown = "  ".join(f"{m}={row[m]:.3f}" for m in METRICS if m in row)
        extra = f"  oracle_f1={row['oracle_f1']:.2f}" if args.oracle else ""
        print(f"[split {split}] sigma={sigma} epoch={epoch:3d}  {shown}{extra}")

    summary = {m: {"mean": float(np.mean([r[m] for r in rows])), "std": float(np.std([r[m] for r in rows]))}
               for m in (*METRICS, "oracle_f1") if m in rows[0]}
    print("=" * 80)
    print("  ".join(f"{m}: {v['mean']:.3f} ± {v['std']:.3f}" for m, v in summary.items()))

    out = base / "results.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"selection": vars(args), "splits": rows, "summary": summary}, handle, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
