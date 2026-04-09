#!/usr/bin/env python3
"""Plot loss_curve.csv from vla_distill training (no pandas required).

Usage:
  python scripts/plot_vla_distill_loss_curve.py [loss_curve.csv] [out.png]
  # default CSV: outputs/vla_distill_robocasa_retry/loss_curve.csv under repo root
  # default PNG: <csv_basename>.png next to the CSV (falls back to ~/ if not writable)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import OrderedDict


def load_rows(path: str) -> tuple[list[str], dict[int, dict[str, float]]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = list(r.fieldnames or [])
        by_iter: OrderedDict[int, dict[str, float]] = OrderedDict()
        for row in r:
            it = int(row["iteration"])
            rec: dict[str, float] = {}
            for k, v in row.items():
                if k is None or k == "iteration":
                    continue
                if k == "epoch":
                    rec[k] = float(v)
                else:
                    try:
                        rec[k] = float(v)
                    except ValueError:
                        pass
            by_iter[it] = rec
        return fieldnames, dict(by_iter)


def main() -> None:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_csv = os.path.join(
        repo, "outputs", "vla_distill_robocasa_retry", "loss_curve.csv"
    )
    csv_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else default_csv
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if len(sys.argv) > 2:
        out_png = os.path.abspath(sys.argv[2])
    else:
        out_png = os.path.splitext(csv_path)[0] + ".png"

    _, data = load_rows(csv_path)
    if not data:
        print("No rows in CSV", file=sys.stderr)
        sys.exit(1)

    iters = sorted(data.keys())
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(key: str) -> list[float]:
        return [data[i][key] for i in iters]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(os.path.basename(csv_path))

    ax = axes[0, 0]
    ax.plot(iters, series("avg_total_loss"), label="avg_total_loss", lw=1.2)
    ax.plot(iters, series("avg_student_edm_loss"), label="avg_student_edm_loss", lw=1.2)
    ax.plot(
        iters,
        series("last_step_student_edm_loss"),
        label="last_step_student_edm_loss",
        lw=1.0,
        alpha=0.7,
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(
        iters,
        series("avg_load_balance_loss"),
        color="C3",
        label="avg_load_balance_loss",
        lw=1.2,
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("load balance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(iters, series("demo_sample_action_l1_loss"), label="action_l1", lw=1.2)
    ax.plot(iters, series("demo_sample_action_mse_loss"), label="action_mse", lw=1.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("demo action metric")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(iters, series("it_per_sec"), color="C2", label="it_per_sec", lw=1.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel("it/s")
    ax2 = ax.twinx()
    ax2.plot(
        iters,
        series("lr_adapter_bank"),
        color="C5",
        ls="--",
        label="lr_adapter_bank",
        lw=1.0,
    )
    ax2.set_ylabel("lr (adapter)")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=7, loc="upper right")

    try:
        fig.savefig(out_png, dpi=150)
    except OSError as e:
        if e.errno == 13 and len(sys.argv) <= 2:
            parent = os.path.basename(os.path.dirname(csv_path)) or "loss_curve"
            out_png = os.path.join(
                os.path.expanduser("~"), f"{parent}_loss_curve.png"
            )
            fig.savefig(out_png, dpi=150)
            print(f"Permission denied next to CSV; wrote {out_png}", file=sys.stderr)
        else:
            raise
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

