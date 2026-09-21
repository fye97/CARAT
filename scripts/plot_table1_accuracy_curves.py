#!/usr/bin/env python3
"""Plot the round-wise results corresponding to Table 1.

The script intentionally uses the same three result archives as the table:

* conventional attacks and the clean Mean reference: legacy text logs;
* Mimic baselines: the clients-20 all-metrics archive;
* CARAT: the clients-20 CARAT-only archive.

Every plotted trajectory is the mean of seeds 42--46.  Shaded regions show
the sample standard deviation across those five seeds.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 43, 44, 45, 46)
ATTACKS = ("ALIE", "FangAttack", "MinMax", "MinSum", "Mimic")
DEFENSES = ("NormClipping", "MultiKrum", "FLTrust", "FLDetector", "CARAT")
ALPHAS = (("1", "1.0"), ("0.5", "0.5"), ("0.1", "0.1"))
ROUNDS = 200


@dataclass(frozen=True)
class DatasetSpec:
    path_name: str
    display_name: str
    y_max: float
    y_ticks: tuple[int, ...]


DATASETS = (
    DatasetSpec("CIFAR100_resnet18", "CIFAR-100 / ResNet18", 60.0, (0, 10, 20, 30, 40, 50, 60)),
    DatasetSpec("CIFAR10_vgg19", "CIFAR-10 / VGG19", 90.0, (0, 15, 30, 45, 60, 75, 90)),
)

METHOD_STYLE = {
    "NoAttack + Mean": dict(color="#202020", linestyle=(0, (6, 3)), linewidth=1.45, zorder=3),
    "NormClipping": dict(color="#F28E2B", linestyle="--", linewidth=1.25, zorder=4),
    "MultiKrum": dict(color="#59A14F", linestyle=":", linewidth=1.45, zorder=4),
    "FLTrust": dict(color="#E15759", linestyle="-.", linewidth=1.25, zorder=4),
    "FLDetector": dict(color="#B07AA1", linestyle=(0, (1, 1.5)), linewidth=1.45, zorder=4),
    "CARAT": dict(color="#1F77B4", linestyle="-", linewidth=1.75, zorder=6),
}

ATTACK_COLOR = {
    "ALIE": "#4C78A8",
    "FangAttack": "#F58518",
    "MinMax": "#E45756",
    "MinSum": "#54A24B",
    "Mimic": "#B279A2",
}

TEXT_LINE = re.compile(
    r"^Epoch\s+(?P<epoch>\d+)\s+.*?Test Acc:\s*"
    r"(?P<accuracy>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Security_unique_default_results/FedAvg"),
        help="Root of the legacy default-result text logs.",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path(
            "/home/fengye/scratch/FL_Security_clients20_all_metrics/FL_Security/logs/FedAvg"
        ),
        help="Root containing the Mimic metrics CSV files.",
    )
    parser.add_argument(
        "--carat-root",
        type=Path,
        default=Path("/home/fengye/scratch/FL_Security_clients20_CARAT_only/logs/FedAvg"),
        help="Root containing the CARAT metrics CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
        help="Directory for the two generated PDF figures.",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        help="Optional CSV containing the final-round mean/std used in the plots.",
    )
    return parser.parse_args()


def read_text_accuracy(path: Path) -> np.ndarray:
    values: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = TEXT_LINE.match(line)
            if match:
                values[int(match.group("epoch"))] = float(match.group("accuracy"))
    return validate_series(path, values)


def read_csv_accuracy(path: Path) -> np.ndarray:
    values: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"epoch", "eval_acc"}.issubset(reader.fieldnames):
            raise RuntimeError(f"Missing epoch/eval_acc columns in {path}")
        for row in reader:
            if row["epoch"] and row["eval_acc"]:
                values[int(float(row["epoch"]))] = float(row["eval_acc"])
    return validate_series(path, values)


def validate_series(path: Path, values: dict[int, float]) -> np.ndarray:
    expected_epochs = set(range(ROUNDS))
    actual_epochs = set(values)
    if actual_epochs != expected_epochs:
        missing = sorted(expected_epochs - actual_epochs)
        extra = sorted(actual_epochs - expected_epochs)
        raise RuntimeError(
            f"Expected epochs 0--{ROUNDS - 1} in {path}; missing={missing[:10]}, extra={extra[:10]}"
        )
    series = np.asarray([values[epoch] for epoch in range(ROUNDS)], dtype=float)
    if not np.isfinite(series).all():
        bad = np.flatnonzero(~np.isfinite(series)).tolist()
        raise RuntimeError(f"Non-finite evaluation accuracy in {path} at epochs {bad[:10]}")
    return series * 100.0


def legacy_path(
    root: Path,
    dataset: str,
    attack: str,
    defense: str,
    alpha_token: str,
    seed: int,
) -> Path:
    folder = root / dataset / "non-iid" / f"{attack}__{defense}"
    pattern = (
        f"{dataset}_non-iid_{attack}_{defense}_200_20_0.05_FedAvg_"
        f"adv0.1_seed{seed}_alpha{alpha_token}_cfg*.txt"
    )
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        rendered = "\n  ".join(str(path) for path in matches[:10]) or "<none>"
        raise RuntimeError(
            f"Expected one legacy log for {dataset}/{attack}/{defense}/alpha={alpha_token}/"
            f"seed={seed}, found {len(matches)}:\n  {rendered}"
        )
    return matches[0]


def metrics_path(
    root: Path,
    dataset: str,
    attack: str,
    defense: str,
    alpha_token: str,
    seed: int,
) -> Path:
    folder = root / dataset / "non-iid" / f"{attack}__{defense}"
    run_pattern = (
        f"ep200_clients20_lr0.05_adv0.1_seed{seed}_exp*_alpha{alpha_token}_cfg*"
    )
    matches = sorted(folder.glob(f"{run_pattern}/metrics_exp*.csv"))
    if len(matches) != 1:
        rendered = "\n  ".join(str(path) for path in matches[:10]) or "<none>"
        raise RuntimeError(
            f"Expected one metrics CSV for {dataset}/{attack}/{defense}/alpha={alpha_token}/"
            f"seed={seed}, found {len(matches)}:\n  {rendered}"
        )
    return matches[0]


def load_runs(
    args: argparse.Namespace,
    dataset: str,
    attack: str,
    method: str,
    alpha_token: str,
) -> np.ndarray:
    runs: list[np.ndarray] = []
    for seed in SEEDS:
        if method == "NoAttack + Mean":
            path = legacy_path(args.legacy_root, dataset, "NoAttack", "Mean", alpha_token, seed)
            runs.append(read_text_accuracy(path))
        elif method == "CARAT":
            path = metrics_path(args.carat_root, dataset, attack, method, alpha_token, seed)
            runs.append(read_csv_accuracy(path))
        elif attack == "Mimic":
            path = metrics_path(args.mimic_root, dataset, attack, method, alpha_token, seed)
            runs.append(read_csv_accuracy(path))
        else:
            path = legacy_path(args.legacy_root, dataset, attack, method, alpha_token, seed)
            runs.append(read_text_accuracy(path))
    result = np.stack(runs)
    if result.shape != (len(SEEDS), ROUNDS):
        raise RuntimeError(f"Unexpected result shape {result.shape} for {dataset}/{attack}/{method}")
    return result


def draw_dataset(
    args: argparse.Namespace, dataset: DatasetSpec
) -> tuple[Path, list[dict[str, str | float | int]]]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.6,
            "axes.titlesize": 10.2,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 5, figsize=(15.8, 10.2), sharex=True, sharey=True)
    rounds = np.arange(1, ROUNDS + 1)
    audit_rows: list[dict[str, str | float | int]] = []
    handles_by_method = {}

    for row_index, (alpha_token, alpha_display) in enumerate(ALPHAS):
        clean_runs = load_runs(
            args, dataset.path_name, "NoAttack", "NoAttack + Mean", alpha_token
        )
        for column_index, attack in enumerate(ATTACKS):
            axis = axes[row_index, column_index]
            accent = ATTACK_COLOR[attack]
            axis.set_facecolor((*plt.matplotlib.colors.to_rgb(accent), 0.045))
            for spine in axis.spines.values():
                spine.set_color(accent)
                spine.set_alpha(0.42)
                spine.set_linewidth(0.9)

            for method in ("NoAttack + Mean", *DEFENSES):
                runs = clean_runs if method == "NoAttack + Mean" else load_runs(
                    args, dataset.path_name, attack, method, alpha_token
                )
                mean = runs.mean(axis=0)
                std = runs.std(axis=0, ddof=1)
                style = METHOD_STYLE[method]
                line, = axis.plot(rounds, mean, label=method, **style)
                fill_alpha = 0.105 if method == "CARAT" else 0.06
                axis.fill_between(
                    rounds,
                    np.clip(mean - std, 0.0, 100.0),
                    np.clip(mean + std, 0.0, 100.0),
                    color=style["color"],
                    alpha=fill_alpha,
                    linewidth=0,
                    zorder=style["zorder"] - 1,
                )
                handles_by_method.setdefault(method, line)
                audit_rows.append(
                    {
                        "dataset": dataset.path_name,
                        "alpha": alpha_display,
                        "attack": attack,
                        "method": method,
                        "seeds": len(SEEDS),
                        "rounds": ROUNDS,
                        "final_mean_percent": float(mean[-1]),
                        "final_sample_std_percent": float(std[-1]),
                    }
                )

            axis.grid(True, color="#B7B7B7", alpha=0.24, linewidth=0.55)
            axis.set_xlim(1, ROUNDS)
            axis.set_ylim(0, dataset.y_max)
            axis.set_xticks((1, 50, 100, 150, 200))
            axis.set_yticks(dataset.y_ticks)
            axis.tick_params(length=2.5, width=0.7)

            if row_index == 0:
                axis.set_title(
                    attack,
                    color=accent,
                    fontweight="bold",
                    pad=7,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": (*plt.matplotlib.colors.to_rgb(accent), 0.10),
                        "edgecolor": (*plt.matplotlib.colors.to_rgb(accent), 0.45),
                        "linewidth": 0.8,
                    },
                )
            if column_index == 0:
                axis.set_ylabel(f"$\\alpha={alpha_display}$\nTest accuracy (\\%)")
            if row_index == len(ALPHAS) - 1:
                axis.set_xlabel("Communication round")

    legend_order = ("NormClipping", "MultiKrum", "FLTrust", "FLDetector", "CARAT", "NoAttack + Mean")
    legend_labels = {
        "NoAttack + Mean": "NoAttack + Mean (clean)",
        "NormClipping": "NormClipping",
        "MultiKrum": "MultiKrum",
        "FLTrust": "FLTrust",
        "FLDetector": "FLDetector",
        "CARAT": "CARAT",
    }
    fig.legend(
        [handles_by_method[name] for name in legend_order],
        [legend_labels[name] for name in legend_order],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        handlelength=3.0,
        columnspacing=1.45,
    )
    fig.suptitle(dataset.display_name, fontsize=11.3, fontweight="bold", y=0.995)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.955, bottom=0.085, wspace=0.12, hspace=0.17)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "cifar100" if dataset.path_name.startswith("CIFAR100") else "cifar10"
    output_path = args.output_dir / f"carat_table1_curves_{suffix}.pdf"
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return output_path, audit_rows


def write_audit(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "dataset",
        "alpha",
        "attack",
        "method",
        "seeds",
        "rounds",
        "final_mean_percent",
        "final_sample_std_percent",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_audit_rows: list[dict[str, str | float | int]] = []
    for dataset in DATASETS:
        output_path, audit_rows = draw_dataset(args, dataset)
        all_audit_rows.extend(audit_rows)
        print(f"wrote {output_path}")
    if args.audit_csv:
        write_audit(args.audit_csv, all_audit_rows)
        print(f"wrote {args.audit_csv}")
    print(
        f"validated {len(all_audit_rows)} plotted configurations: "
        f"{len(SEEDS)} seeds x {ROUNDS} rounds each"
    )


if __name__ == "__main__":
    main()
