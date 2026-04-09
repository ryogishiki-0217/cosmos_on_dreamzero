#!/usr/bin/env python3
"""
Plot training curves from train_vla_distill.py loss_curve.csv.

Example:
  python3 scripts/plot_loss_curve.py \\
    --csv outputs/vla_distill_robocasa/loss_curve.csv \\
    --out outputs/vla_distill_robocasa/loss_curve.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"Empty or invalid CSV: {path}")
        rows = [dict(r) for r in reader]
    return list(fieldnames), rows


def _col_float(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    out: list[float | None] = []
    for r in rows:
        v = r.get(key, "")
        if v is None or v == "":
            out.append(None)
            continue
        try:
            out.append(float(v))
        except ValueError:
            out.append(None)
    return out


def _col_int(rows: list[dict[str, Any]], key: str) -> list[int | None]:
    out: list[int | None] = []
    for r in rows:
        v = r.get(key, "")
        if v is None or v == "":
            out.append(None)
            continue
        try:
            out.append(int(float(v)))
        except ValueError:
            out.append(None)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Plot loss_curve.csv from vla_distill training.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/vla_distill_robocasa/loss_curve.csv"),
        help="Path to loss_curve.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: same dir as csv, name loss_curve.png)",
    )
    p.add_argument(
        "--x",
        choices=("iteration", "index"),
        default="iteration",
        help="X axis: iteration column, or row index (0..N-1) if iterations repeat across runs.",
    )
    args = p.parse_args()

    csv_path = args.csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    fieldnames, rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No data rows in {csv_path}")

    out_path = args.out
    if out_path is None:
        out_path = csv_path.parent / "loss_curve.png"
    else:
        out_path = out_path.resolve()

    if args.x == "iteration":
        x = _col_int(rows, "iteration")
        x_label = "iteration"
    else:
        x = list(range(len(rows)))
        x_label = "row index"

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"Loss curve — {csv_path.name}", fontsize=12)

    # (1) Average losses
    ax = axes[0, 0]
    for key, label, style in [
        ("avg_total_loss", "avg total loss", "-"),
        ("avg_student_edm_loss", "avg student EDM", "-"),
        ("avg_load_balance_loss", "avg load balance", "-"),
    ]:
        if key in fieldnames:
            y = _col_float(rows, key)
            ax.plot(x, y, style, label=label, linewidth=1.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.set_title("Averaged losses (per log window)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) Last-step EDM
    ax = axes[0, 1]
    if "last_step_student_edm_loss" in fieldnames:
        y = _col_float(rows, "last_step_student_edm_loss")
        ax.plot(x, y, color="C0", label="last_step_student_edm_loss", linewidth=1.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("loss")
    ax.set_title("Last-step student EDM")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) Learning rates
    ax = axes[1, 0]
    for key, label in [
        ("lr_adapter_bank", "lr adapter_bank"),
        ("lr_student_backbone", "lr student_backbone"),
    ]:
        if key in fieldnames:
            y = _col_float(rows, key)
            ax.semilogy(x, y, label=label, linewidth=1.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel("lr (log scale)")
    ax.set_title("Learning rates")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (4) Throughput + optional demo metrics
    ax = axes[1, 1]
    if "it_per_sec" in fieldnames:
        y = _col_float(rows, "it_per_sec")
        ax.plot(x, y, color="green", label="it/s", linewidth=1.2)
        ax.set_ylabel("iterations / sec", color="green")
        ax.tick_params(axis="y", labelcolor="green")
    ax2 = ax.twinx() if ("demo_sample_action_l1_loss" in fieldnames or "demo_sample_action_mse_loss" in fieldnames) else None
    if ax2 is not None:
        if "demo_sample_action_l1_loss" in fieldnames:
            y1 = _col_float(rows, "demo_sample_action_l1_loss")
            ax2.plot(x, y1, color="C1", alpha=0.8, label="action_l1", linewidth=1.0)
        if "demo_sample_action_mse_loss" in fieldnames:
            y2 = _col_float(rows, "demo_sample_action_mse_loss")
            ax2.plot(x, y2, color="C3", alpha=0.8, label="action_mse", linewidth=1.0)
        ax2.set_ylabel("demo action metrics")
        ax2.tick_params(axis="y")
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lab1 + lab2, loc="best", fontsize=7)
    else:
        ax.legend(loc="best", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title("Speed & demo metrics (if present)")
    ax.grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
