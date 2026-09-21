#!/usr/bin/env python3
"""Plot CARAT accuracy against the malicious-client fraction for 20 clients.

For each dataset, Dirichlet alpha, and fraction, first average the five attack
accuracies within each seed, then plot the mean and sample standard deviation
across seeds 42--46. Every selected run must contain 200 completed rounds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASETS = (
    ("CIFAR100_resnet18", "CIFAR-100 / ResNet18", (20, 55)),
    ("CIFAR10_vgg19", "CIFAR-10 / VGG19", (20, 85)),
)
ALPHAS = (("1", "1.0", "#1F77B4", "o", "-"),
          ("0.5", "0.5", "#E17C24", "s", "--"),
          ("0.1", "0.1", "#4D9972", "^", ":"))
ATTACKS = ("ALIE", "FangAttack", "MinMax", "MinSum", "Mimic")
FRACTIONS = ("0.1", "0.2", "0.3", "0.4")
SEEDS = range(42, 47)
ROUNDS = 200


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Security_clients20_CARAT_only/logs/FedAvg"),
        help="Root of the completed 20-client CARAT logs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures/carat_attacker_fraction.pdf",
    )
    parser.add_argument("--audit-csv", type=Path, help="Optional summary CSV.")
    return parser.parse_args()


def final_accuracy(
    root: Path, dataset: str, alpha: str, fraction: str, attack: str, seed: int
) -> float:
    folder = root / dataset / "non-iid" / f"{attack}__CARAT"
    run_pattern = (
        f"ep200_clients20_lr0.05_adv{fraction}_seed{seed}_exp*_alpha{alpha}_cfg*"
    )
    matches = sorted(folder.glob(f"{run_pattern}/metrics_exp*.csv"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one result for {dataset}/{alpha}/{fraction}/{attack}/{seed}; "
            f"found {len(matches)}"
        )
    path = matches[0]
    if not (path.parent / "task.complete").is_file():
        raise RuntimeError(f"Run lacks completion marker: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != ROUNDS or [int(row["epoch"]) for row in rows] != list(range(ROUNDS)):
        raise RuntimeError(f"Run does not have exactly 200 ordered rounds: {path}")
    accuracy = float(rows[-1]["eval_acc"]) * 100.0
    if not np.isfinite(accuracy):
        raise RuntimeError(f"Non-finite round-200 accuracy: {path}")
    return accuracy


def collect(args: argparse.Namespace) -> list[dict[str, str | int | float]]:
    records = []
    for dataset, _, _ in DATASETS:
        for alpha, _, _, _, _ in ALPHAS:
            for fraction in FRACTIONS:
                per_seed = []
                for seed in SEEDS:
                    per_attack = [
                        final_accuracy(args.results_root, dataset, alpha, fraction, attack, seed)
                        for attack in ATTACKS
                    ]
                    per_seed.append(float(np.mean(per_attack)))
                records.append(
                    {
                        "dataset": dataset,
                        "alpha": alpha,
                        "malicious_fraction": fraction,
                        "malicious_clients": round(20 * float(fraction)),
                        "attack_count": len(ATTACKS),
                        "seed_count": len(SEEDS),
                        "round": ROUNDS,
                        "mean_accuracy_percent": float(np.mean(per_seed)),
                        "sample_std_percent": float(np.std(per_seed, ddof=1)),
                    }
                )
    return records


def plot(records: list[dict[str, str | int | float]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), sharex=True)
    x = np.array([100 * float(fraction) for fraction in FRACTIONS])
    for axis, (dataset, title, limits) in zip(axes, DATASETS):
        for alpha, display, color, marker, linestyle in ALPHAS:
            points = [
                row for row in records if row["dataset"] == dataset and row["alpha"] == alpha
            ]
            mean = np.array([float(row["mean_accuracy_percent"]) for row in points])
            std = np.array([float(row["sample_std_percent"]) for row in points])
            axis.errorbar(
                x,
                mean,
                yerr=std,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=1.2,
                markersize=3.5,
                capsize=2.3,
                elinewidth=0.75,
                label=rf"$\alpha={display}$",
                zorder=3,
            )
        axis.set_title(title, weight="bold", pad=4)
        axis.set_xlim(7, 43)
        axis.set_ylim(*limits)
        axis.set_xticks(x)
        axis.set_yticks(np.arange(limits[0], limits[1] + 1, 10))
        axis.grid(True, color="#B8B8B8", linewidth=0.45, alpha=0.35)
        axis.set_facecolor("white")
        for spine in axis.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.55)
        axis.set_xlabel("Malicious clients (%)")
        axis.tick_params(length=2, width=0.5, pad=2)
    axes[0].set_ylabel("Round-200 test accuracy (%)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.1,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.31, wspace=0.21)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    args = arguments()
    records = collect(args)
    plot(records, args.output)
    if args.audit_csv:
        args.audit_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f"Validated {len(records)} settings x {len(ATTACKS)} attacks x {len(SEEDS)} seeds = 600 runs")
    for row in records:
        print(
            f"{row['dataset']} alpha={row['alpha']} adv={row['malicious_fraction']}: "
            f"{row['mean_accuracy_percent']:.2f} +/- {row['sample_std_percent']:.2f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
