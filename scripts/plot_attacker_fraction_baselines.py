#!/usr/bin/env python3
"""Attacker-fraction sweep with baselines (20 clients, non-i.i.d., seeds 42--46).

Reads the deduplicated result store (one TXT per run, FL_Poison-compatible format), which
already contains the NaN-fixed CARAT reruns. For each dataset, Dirichlet alpha, malicious
fraction and defense, the attack accuracies are averaged within each seed, then the mean and
sample standard deviation over seeds are reported. Mean (no defense) under attack is the
undefended reference.

Mean x Mimic was not part of the original sweep (specs: FL_Security
exps/specs/Fig4_Baselines). Until those runs are merged, pass
--mean-attacks ALIE,FangAttack,MinMax,MinSum to average only four attacks for Mean.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = (("CIFAR100_resnet18", "CIFAR-100 / ResNet18"), ("CIFAR10_vgg19", "CIFAR-10 / VGG19"))
ALPHAS = (("1", r"$\alpha=1.0$"), ("0.5", r"$\alpha=0.5$"), ("0.1", r"$\alpha=0.1$"))
FRACTIONS = ("0.1", "0.2", "0.3", "0.4")
SEEDS = range(42, 47)
# (defense, label, color, marker, linestyle); validated categorical palette, Mean as gray reference
METHODS = (
    ("CARAT", "CARAT", "#2a78d6", "o", "-"),
    ("FLDetector", "FLDetector", "#eb6834", "s", "-"),
    ("FLTrust", "FLTrust", "#1baf7a", "^", "-"),
    ("MultiKrum", "MultiKrum", "#eda100", "D", "-"),
    ("NormClipping", "NormClipping", "#e87ba4", "v", "-"),
    ("Mean", "Mean (no defense)", "#6b6b6b", "x", "--"),
)
EPOCH_RE = re.compile(r"Epoch 199\s.*?Test Acc: ([\d.]+)")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path,
                        default=Path.home() / "scratch/FL_Security_unique_default_results/FedAvg")
    parser.add_argument("--attacks", default="ALIE,FangAttack,MinMax,MinSum,Mimic")
    parser.add_argument("--mean-attacks", default=None,
                        help="attack list for Mean only (default: same as --attacks)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "figures/carat_attacker_fraction_baselines.pdf")
    parser.add_argument("--audit-csv", type=Path)
    return parser.parse_args()


def accuracy(root: Path, dataset: str, alpha: str, frac: str, attack: str, defense: str, seed: int) -> float:
    folder = root / dataset / "non-iid" / f"{attack}__{defense}"
    pattern = f"{dataset}_non-iid_{attack}_{defense}_200_20_*_adv{frac}_seed{seed}_alpha{alpha}_cfg*.txt"
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one run for {pattern} in {folder}, found {len(matches)}")
    found = EPOCH_RE.findall(matches[0].read_text(errors="replace"))
    if not found:
        raise RuntimeError(f"no round-200 accuracy in {matches[0]}")
    value = float(found[-1]) * 100.0
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite accuracy in {matches[0]}")
    return value


def collect(root: Path, attacks: list[str], mean_attacks: list[str]) -> list[dict]:
    records = []
    for dataset, _ in DATASETS:
        for alpha, _ in ALPHAS:
            for defense, *_ in METHODS:
                used = mean_attacks if defense == "Mean" else attacks
                for frac in FRACTIONS:
                    per_seed = [np.mean([accuracy(root, dataset, alpha, frac, a, defense, s) for a in used])
                                for s in SEEDS]
                    records.append(dict(dataset=dataset, alpha=alpha, defense=defense, malicious_fraction=frac,
                                        attacks="+".join(used), seeds=len(per_seed),
                                        mean=float(np.mean(per_seed)), std=float(np.std(per_seed, ddof=1))))
    return records


def plot(records: list[dict], output: Path, attacks: list[str]) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.0, "axes.titlesize": 8.6,
                         "axes.labelsize": 8.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7.8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.3), sharex=True)
    x = np.array([100 * float(f) for f in FRACTIONS])
    for row, (dataset, dtitle) in enumerate(DATASETS):
        for col, (alpha, atitle) in enumerate(ALPHAS):
            ax = axes[row, col]
            for defense, label, color, marker, ls in METHODS:
                pts = [r for r in records if r["dataset"] == dataset and r["alpha"] == alpha and r["defense"] == defense]
                ax.errorbar(x, [p["mean"] for p in pts], yerr=[p["std"] for p in pts], color=color, marker=marker,
                            linestyle=ls, linewidth=2.0 if defense == "CARAT" else 1.2, markersize=4.0,
                            capsize=2.0, elinewidth=0.7, label=label, zorder=4 if defense == "CARAT" else 3)
            ax.set_title(f"{dtitle}, {atitle}", pad=3)
            ax.set_xticks(x)
            ax.set_xlim(7, 43)
            ax.grid(True, color="#B8B8B8", linewidth=0.45, alpha=0.35)
            for spine in ax.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(0.55)
            ax.tick_params(length=2, width=0.5, pad=2)
            if row == 1:
                ax.set_xlabel("Malicious clients (%)")
            if col == 0:
                ax.set_ylabel("Round-200 accuracy (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=len(METHODS), frameon=False, handlelength=2.0, columnspacing=1.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15, wspace=0.22, hspace=0.32)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def main() -> None:
    args = arguments()
    attacks = args.attacks.split(",")
    mean_attacks = args.mean_attacks.split(",") if args.mean_attacks else attacks
    records = collect(args.store_root, attacks, mean_attacks)
    plot(records, args.output, attacks)
    if args.audit_csv:
        with args.audit_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    cells = len(DATASETS) * len(ALPHAS) * len(FRACTIONS) * len(SEEDS)
    n = cells * ((len(METHODS) - 1) * len(attacks) + len(mean_attacks))
    print(f"Validated {n} runs (attacks: {', '.join(attacks)}; Mean: {', '.join(mean_attacks)})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
