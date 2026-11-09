"""Benchmark runner for executing complete experimental suites and generating summary tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import CloudDataset, load_and_clean_data
from src.train import train_single_experiment


def get_experiment_filename(
    algo: str,
    data_mode: str,
    feature_set: str,
    episodes: int,
    planning_steps: Optional[int] = None,
) -> str:
    """Standardized results JSON filename."""
    plan_suffix = f"_plan{planning_steps}" if planning_steps is not None else ""
    return f"{algo}_{data_mode}_{feature_set}{plan_suffix}_ep{episodes}.json"


def run_all_benchmarks(
    suite: str = "all",
    episodes: int = 1000,
    seeds: Optional[List[int]] = None,
    eval_episodes_count: int = 100,
    config: Optional[Dict[str, Any]] = None,
    results_dir: Optional[str | Path] = None,
    force_rerun: bool = False,
) -> List[Dict[str, Any]]:
    """Execute benchmarks across suites A, B, C, D, skipping already computed experiments."""
    cfg = config or load_config()
    target_seeds = seeds or cfg.get("seeds", [0, 1, 2, 3, 4])

    if results_dir is None:
        results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Preload dataset once for all dataset experiments
    print("Loading cached dataset for dataset mode experiments...")
    raw_path = cfg.get(
        "dataset_path", "data/raw/vmCloud_data.csv"
    )
    df = load_and_clean_data(raw_path)
    dataset = CloudDataset(
        df,
        train_ratio=0.8,
        network_thresholds=tuple(cfg.get("network_thresholds", [333.33, 666.67])),
        exec_thresholds=tuple(cfg.get("execution_time_thresholds", [33.33, 66.67])),
    )

    experiments_to_run: List[Dict[str, Any]] = []

    # Suite A: Main comparison on feature_set=full (random, threshold, qlearning, sarsa, dynaq(10), dqn)
    if suite in ["all", "main", "A"]:
        for mode in ["synthetic", "dataset"]:
            for algo in ["random", "threshold", "qlearning", "sarsa", "dynaq", "dqn"]:
                experiments_to_run.append(
                    {
                        "suite": "A (Main Comparison)",
                        "algo": algo,
                        "data_mode": mode,
                        "feature_set": "full",
                        "planning_steps": 10 if algo == "dynaq" else None,
                    }
                )

    # Suite B: Controlled State-Space (base 45 states vs full 405 states) & Feature Ablation
    if suite in ["all", "ablation", "B"]:
        for mode in ["synthetic", "dataset"]:
            # Task 2: State-Space Comparison for all 4 learning algorithms
            for algo in ["qlearning", "sarsa", "dynaq", "dqn"]:
                for f_set in ["base", "full"]:
                    experiments_to_run.append(
                        {
                            "suite": "B (State-Space 45 vs 405)",
                            "algo": algo,
                            "data_mode": mode,
                            "feature_set": f_set,
                            "planning_steps": 10 if algo == "dynaq" else None,
                        }
                    )
            # Detailed feature ablation for Q-Learning and Dyna-Q
            for algo in ["qlearning", "dynaq"]:
                for f_set in ["base+network", "base+exec"]:
                    experiments_to_run.append(
                        {
                            "suite": "B (Feature Ablation)",
                            "algo": algo,
                            "data_mode": mode,
                            "feature_set": f_set,
                            "planning_steps": 10 if algo == "dynaq" else None,
                        }
                    )

    # Suite C: Dyna-Q planning steps: 0, 5, 10, 20, 50
    if suite in ["all", "planning", "C"]:
        for mode in ["synthetic", "dataset"]:
            for p_steps in [0, 5, 10, 20, 50]:
                experiments_to_run.append(
                    {
                        "suite": "C (Dyna-Q Planning Steps)",
                        "algo": "dynaq",
                        "data_mode": mode,
                        "feature_set": "full",
                        "planning_steps": p_steps,
                    }
                )

    # Deduplicate experiments by unique key
    deduped_experiments: List[Dict[str, Any]] = []
    seen_keys = set()
    for exp in experiments_to_run:
        key = (exp["algo"], exp["data_mode"], exp["feature_set"], exp["planning_steps"])
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_experiments.append(exp)
    experiments_to_run = deduped_experiments

    completed_summaries: List[Dict[str, Any]] = []

    for exp in experiments_to_run:
        algo = exp["algo"]
        mode = exp["data_mode"]
        f_set = exp["feature_set"]
        p_steps = exp["planning_steps"]
        s_name = exp["suite"]

        fname = get_experiment_filename(algo, mode, f_set, episodes, p_steps)
        fpath = results_dir / fname

        needs_run = True
        if fpath.exists() and not force_rerun:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    res = json.load(f)
                if set(res.get("seeds", [])) == set(target_seeds) and res.get("episodes") == episodes:
                    print(f"[{s_name}] Skipping completed: {fname}")
                    needs_run = False
            except Exception:
                needs_run = True

        if needs_run:
            print(f"\n[{s_name}] Running: {algo} | Mode: {mode} | Features: {f_set} | Plan: {p_steps}")
            res = train_single_experiment(
                algo=algo,
                data_mode=mode,
                feature_set=f_set,
                seeds=target_seeds,
                episodes=episodes,
                planning_steps=p_steps,
                eval_episodes_count=eval_episodes_count,
                config=cfg,
                dataset=dataset,
                output_dir=results_dir,
            )

        res["suite"] = s_name
        completed_summaries.append(res)

    # Generate and print Summary Table per suite
    print("\n" + "=" * 90)
    print("EXPERIMENT SUITE SUMMARY REPORT")
    print("=" * 90)

    summary_records = []
    for r in completed_summaries:
        summary_records.append(
            {
                "Suite": r.get("suite", "Custom"),
                "Algorithm": r["algo"].upper(),
                "Mode": r["data_mode"],
                "Features": r["feature_set"],
                "Plan Steps": r["planning_steps"],
                "Test Reward": f"{r['mean_test_reward']:.2f} +/- {r['std_test_reward']:.2f}",
                "Test SLA Violations": f"{r['mean_test_sla_violations']:.1f}",
                "Test Cost": f"{r['mean_test_cost']:.2f}",
                "Convergence (Ep)": f"{r['mean_convergence_episodes']:.1f}",
            }
        )

    summary_df = pd.DataFrame(summary_records).drop_duplicates()
    print(summary_df.to_string(index=False))

    # Save summary dataframe to CSV
    summary_csv = results_dir / f"summary_table_ep{episodes}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved full summary table to: {summary_csv}")

    return completed_summaries


def parse_args():
    parser = argparse.ArgumentParser(description="Run Enhanced Autoscaling Benchmark Suites")
    parser.add_argument(
        "--suite",
        type=str,
        default="all",
        choices=["all", "main", "ablation", "planning", "A", "B", "C"],
        help="Experiment suite to run",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes per run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Seeds")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test (100 eps, 2 seeds)")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerunning existing runs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke:
        print("Running SMOKE TEST: 100 episodes, 2 seeds ([0, 1])...")
        run_all_benchmarks(
            suite=args.suite,
            episodes=100,
            seeds=[0, 1],
            eval_episodes_count=20,
            force_rerun=args.force_rerun,
        )
    else:
        run_all_benchmarks(
            suite=args.suite,
            episodes=args.episodes,
            seeds=args.seeds,
            force_rerun=args.force_rerun,
        )
