#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Run RoboCasa evaluation over many tasks (and optional multiple seeds), aligned with
``run_robocasa_eval.py`` / ``ROBOCASA.md`` (per-task rollouts + deterministic seeds).

Typical paper-style reporting (see ROBOCASA.md):
  - Per task: ``num_trials_per_task`` rollouts (default 50).
  - Optional: seeds {195, 196, 197} with ``--deterministic True``, then **average success
    rates across seeds** (report mean ± std over seeds per task, and overall macro mean).

**How "average success ~69%" is usually computed in tables**
  - **Macro average over tasks**: mean of per-task success rates (each task weighted equally),
    often after averaging across seeds first. This matches "average over 24 tasks".
  - **Micro average**: total successes / total episodes across all tasks — only equivalent to
    macro when every task has the same number of trials (here: same ``num_trials_per_task``).

This script reports both macro and micro when all tasks use the same trial count.

Usage (pass through all arguments for ``run_robocasa_eval`` after ``--``):

  python scripts/run_robocasa_full_benchmark.py \\
    --suite cosmos24 \\
    --num-trials-per-task 50 \\
    --seeds 195,196,197 \\
    --output-json robocasa_benchmark_summary.json \\
    -- \\
    -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \\
    --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \\
    --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \\
    ...

If your shell already uses ``python -m ...``, you can instead set ``--eval-module`` and pass
only the module arguments after ``--`` (see ``--help``).

Note: The inline comment "67.1% avg" in ``run_robocasa_eval.py`` is a **single-checkpoint
example**, not a guarantee for your hardware/PyTorch; ROBOCASA.md states results vary slightly
with stack and GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

# 24 single-stage tasks used in Cosmos RoboCasa training (excludes NavigateKitchen); see ROBOCASA.md
COSMOS24_TASKS: tuple[str, ...] = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    "CloseSingleDoor",
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
)


def _try_import_registry_tasks() -> tuple[list[str], list[str]] | None:
    try:
        from robocasa.utils.dataset_registry import (
            MULTI_STAGE_TASK_DATASETS,
            SINGLE_STAGE_TASK_DATASETS,
        )
    except Exception:
        return None
    single = list(SINGLE_STAGE_TASK_DATASETS.keys())
    multi = list(MULTI_STAGE_TASK_DATASETS.keys())
    return single, multi


def resolve_task_names(
    suite: str,
    tasks_override: str | None,
) -> list[str]:
    if tasks_override:
        return [t.strip() for t in tasks_override.split(",") if t.strip()]
    if suite == "cosmos24":
        return list(COSMOS24_TASKS)
    reg = _try_import_registry_tasks()
    if reg is None:
        raise RuntimeError(
            "robocasa is not importable; install robocasa / use --tasks or --suite cosmos24."
        )
    single, multi = reg
    if suite == "single_stage":
        return single
    if suite == "multi_stage":
        return multi
    if suite == "all_registry":
        return single + multi
    raise ValueError(f"Unknown suite: {suite}")


def parse_success_rate(text: str) -> float | None:
    """Parse last 'Success rate: 0.xxxx' line from eval stdout/log."""
    matches = re.findall(r"Success rate:\s+([0-9.]+)\s+", text)
    if not matches:
        matches = re.findall(r"Success rate:\s+([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


def parse_avg_episode_length(text: str) -> float | None:
    matches = re.findall(r"Average episode length:\s+([0-9.]+)", text)
    if not matches:
        return None
    return float(matches[-1])


@dataclass
class RunResult:
    task: str
    seed: int
    returncode: int
    success_rate: float | None
    avg_episode_length: float | None
    stdout_tail: str


def run_one(
    *,
    python_exe: str,
    eval_argv: list[str],
    task_name: str,
    seed: int,
    num_trials: int,
    run_id_prefix: str,
) -> RunResult:
    argv = [python_exe, *eval_argv]
    # Ensure task / seed / trials / note are set last so they override duplicates in eval_argv
    extra = [
        "--task_name",
        task_name,
        "--seed",
        str(seed),
        "--num_trials_per_task",
        str(num_trials),
        "--run_id_note",
        f"{run_id_prefix}--{task_name}--seed{seed}",
    ]
    # Avoid duplicate flags: strip existing task_name/seed/num_trials_per_task/run_id_note from eval_argv
    def _strip_duplicates(src: list[str]) -> list[str]:
        drop = {
            "--task_name",
            "--seed",
            "--num_trials_per_task",
            "--run_id_note",
        }
        out: list[str] = []
        i = 0
        while i < len(src):
            if src[i] in drop:
                i += 2
                continue
            out.append(src[i])
            i += 1
        return out

    clean = _strip_duplicates(argv[1:])
    argv = [python_exe] + clean + extra

    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    sr = parse_success_rate(out)
    le = parse_avg_episode_length(out)
    tail = out[-4000:] if len(out) > 4000 else out
    return RunResult(
        task=task_name,
        seed=seed,
        returncode=proc.returncode,
        success_rate=sr,
        avg_episode_length=le,
        stdout_tail=tail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RoboCasa eval over multiple tasks (and seeds), aggregate success rates."
    )
    parser.add_argument(
        "--suite",
        default="cosmos24",
        choices=("cosmos24", "single_stage", "multi_stage", "all_registry"),
        help="cosmos24 = 24 training single-stage tasks (default). all_registry = single+multi from robocasa.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated task names (overrides --suite).",
    )
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument(
        "--seeds",
        default="195",
        help="Comma-separated seeds (paper often uses 195,196,197).",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for child.")
    parser.add_argument(
        "--eval-module",
        default="cosmos_policy.experiments.robot.robocasa.run_robocasa_eval",
        help="Module to run as: python -m <eval-module> ...",
    )
    parser.add_argument(
        "--run-id-prefix",
        default="fullbench",
        help="Prefix for --run_id_note so logs are easy to grep.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Write full summary JSON to this path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned tasks/seeds and exit.",
    )
    args, passthrough = parser.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    if not passthrough:
        # Default: same as: python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval
        eval_argv = ["-m", args.eval_module]
    elif passthrough[0] == "-m":
        eval_argv = passthrough
    else:
        # Allow: -- --config ...  -> insert -m module
        eval_argv = ["-m", args.eval_module, *passthrough]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tasks = resolve_task_names(args.suite, args.tasks or None)

    if args.dry_run:
        print("tasks:", len(tasks), tasks)
        print("seeds:", seeds)
        print("eval argv:", [args.python] + eval_argv)
        return

    results: list[RunResult] = []
    for task in tasks:
        for seed in seeds:
            print(f"\n=== Task={task} seed={seed} ===", flush=True)
            r = run_one(
                python_exe=args.python,
                eval_argv=eval_argv,
                task_name=task,
                seed=seed,
                num_trials=args.num_trials_per_task,
                run_id_prefix=args.run_id_prefix,
            )
            results.append(r)
            print(
                f"returncode={r.returncode} success_rate={r.success_rate} avg_len={r.avg_episode_length}",
                flush=True,
            )
            if r.returncode != 0:
                print(r.stdout_tail, file=sys.stderr)
            if r.success_rate is None and r.returncode == 0:
                print(
                    "WARNING: could not parse Success rate from output; check log format.",
                    file=sys.stderr,
                )

    # Aggregate
    per_task: dict[str, Any] = {}
    for task in tasks:
        rates = [r.success_rate for r in results if r.task == task and r.success_rate is not None]
        rc_ok = all(r.returncode 0 for r in results if r.task == task)
        per_task[task] = {
            "seeds": {str(r.seed): r.success_rate for r in results if r.task == task},
            "mean_success_rate_over_seeds": float(sum(rates) / len(rates)) if rates else None,
            "std_success_rate_over_seeds": _std(rates) if len(rates) > 1 else 0.0,
            "all_runs_ok": rc_ok,
        }

    # Macro: mean of per-task mean (each task equal weight)
    task_means = [per_task[t]["mean_success_rate_over_seeds"] for t in tasks]
    task_means_f = [x for x in task_means if x is not None]
    macro_over_tasks = float(sum(task_means_f) / len(task_means_f)) if task_means_f else None

    # Micro: pool episodes — same trials per task => micro equals macro over seeds if one seed;
    # for multiple seeds, "micro" per seed slice
    micro_by_seed: dict[str, float | None] = {}
    macro_per_seed_dict: dict[str, float] = {}
    for seed in seeds:
        sr_list = [r.success_rate for r in results if r.seed == seed and r.success_rate is not None]
        if len(sr_list) == len(tasks):
            m = float(sum(sr_list) / len(sr_list))
            micro_by_seed[str(seed)] = m
            macro_per_seed_dict[str(seed)] = m
        else:
            micro_by_seed[str(seed)] = None

    _mps = list(macro_per_seed_dict.values())

    summary = {
        "suite": args.suite,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "num_trials_per_task": args.num_trials_per_task,
        "seeds": seeds,
        "per_task": per_task,
        "macro_mean_success": macro_over_tasks,
        "macro_std_across_seeds": _std(_mps) if len(_mps) > 1 else 0.0,
        "macro_per_seed": macro_per_seed_dict,
        "macro_note": "macro_mean_success: mean over tasks of (mean over seeds per task). "
        "macro_std_across_seeds: std of 'macro_per_seed' values (one macro average per seed).",
        "micro_mean_by_seed": micro_by_seed,
        "micro_note": "Per seed: mean of per-task success rates (equal weight per task). "
        "With equal trials per task, this matches macro for that seed.",
        "failed_runs": [
            asdict(r)
            for r in results
            if r.returncode != 0 or (r.success_rate is None and r.returncode == 0)
        ],
    }

    print("\n" + "=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output_json:
        _out_dir = os.path.dirname(os.path.abspath(args.output_json))
        if _out_dir:
            os.makedirs(_out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nWrote {args.output_json}")


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var**0.5)


if __name__ == "__main__":
    main()

