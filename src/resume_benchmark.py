"""Safe Resume Benchmark Script for Option B (Prioritized Completion).

Features:
- Skips all 12 completed Suite A configurations without touching them.
- Runs only pending Suite B (State-Space & Feature Ablations) and Suite C (Dyna-Q Planning Steps).
- Explicitly excludes and blocks any DQN base experiments (postponed per Option B).
- Uses 1,000 training episodes, seeds [0, 1, 2, 3, 4], and 100 evaluation episodes per seed.
- Atomically saves results and never overwrites valid completed files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
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
    plan_suffix = (
        f"_plan{planning_steps}" if algo == "dynaq" and planning_steps is not None else ""
    )
    return f"{algo}_{data_mode}_{feature_set}{plan_suffix}_ep{episodes}.json"


def verify_file_completed(fpath: Path, expected_seeds: List[int], expected_episodes: int) -> bool:
    """Check if file exists and contains full uncorrupted results for all expected seeds."""
    if not fpath.exists():
        return False
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        seeds = data.get("seeds", [])
        runs = data.get("runs", [])
        episodes = data.get("episodes", 0)
        if set(seeds) == set(expected_seeds) and len(runs) == len(expected_seeds) and episodes == expected_episodes:
            return True
        return False
    except Exception:
        return False


def get_pending_experiments() -> List[Dict[str, Any]]:
    """Define exact pending experiment matrix for Suite B and Suite C (Option B)."""
    pending = []

    # ---------------------------------------------------------
    # SUITE B: State-Space (base 45 states) & Feature Ablations
    # Algorithms: qlearning, sarsa, dynaq (DQN base is EXCLUDED)
    # ---------------------------------------------------------
    for mode in ["synthetic", "dataset"]:
        # Q-Learning
        for f_set in ["base", "base+network", "base+exec"]:
            pending.append({
                "suite": "Suite B (State-Space / Ablation)",
                "algo": "qlearning",
                "data_mode": mode,
                "feature_set": f_set,
                "planning_steps": None,
            })
        # SARSA (base 45 states)
        pending.append({
            "suite": "Suite B (State-Space: 45 States)",
            "algo": "sarsa",
            "data_mode": mode,
            "feature_set": "base",
            "planning_steps": None,
        })
        # Dyna-Q (plan=10 for base, base+network, base+exec)
        for f_set in ["base", "base+network", "base+exec"]:
            pending.append({
                "suite": "Suite B (State-Space / Ablation)",
                "algo": "dynaq",
                "data_mode": mode,
                "feature_set": f_set,
                "planning_steps": 10,
            })

    # ---------------------------------------------------------
    # SUITE C: Dyna-Q Planning Steps (k = 0, 5, 20, 50) on 'full'
    # (k=10 is already completed in Suite A and will be reused)
    # ---------------------------------------------------------
    for mode in ["dataset", "synthetic"]:
        for k in [0, 5, 20, 50]:
            pending.append({
                "suite": f"Suite C (Planning k={k})",
                "algo": "dynaq",
                "data_mode": mode,
                "feature_set": "full",
                "planning_steps": k,
            })

    # Strict safety assertion: verify DQN base is NOT in queue
    for exp in pending:
        if exp["algo"] == "dqn":
            raise RuntimeError("CRITICAL ERROR: DQN experiment found in pending queue! Aborting.")

    return pending


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    expected_seeds = [0, 1, 2, 3, 4]
    episodes = 1000
    eval_episodes_count = 100

    print("=" * 80)
    print("STEP 1: AUDITING EXISTING COMPLETED EXPERIMENTS IN SUITE A")
    print("=" * 80)

    suite_a_algos = ["random", "threshold", "qlearning", "sarsa", "dynaq", "dqn"]
    completed_suite_a = 0
    for mode in ["synthetic", "dataset"]:
        for algo in suite_a_algos:
            p_step = 10 if algo == "dynaq" else None
            fname = get_experiment_filename(algo, mode, "full", episodes, p_step)
            fpath = results_dir / fname
            if verify_file_completed(fpath, expected_seeds, episodes):
                completed_suite_a += 1
                with open(fpath, "r", encoding="utf-8") as f:
                    d = json.load(f)
                r = d["mean_test_reward"]
                sla = d["mean_test_sla_violations"]
                print(f"  [ALREADY DONE] {fname:42} | Reward: {r:10.2f} | SLA: {sla:5.1f} | 5 Seeds Verified")
            else:
                print(f"  [MISSING/CORRUPT] {fname}")

    print(f"\nSuite A Status: {completed_suite_a} / 12 configurations complete.")
    if completed_suite_a != 12:
        print("Warning: Not all Suite A configurations are complete! Aborting.")
        return

    # Load dataset for dataset mode
    print("\nLoading dataset cache for pending dataset experiments...")
    raw_path = cfg.get("dataset_path", "data/raw/vmCloud_data.csv")
    df = load_and_clean_data(raw_path)
    dataset = CloudDataset(
        df,
        train_ratio=0.8,
        network_thresholds=tuple(cfg.get("network_thresholds", [333.33, 666.67])),
        exec_thresholds=tuple(cfg.get("execution_time_thresholds", [33.33, 66.67])),
    )

    pending_list = get_pending_experiments()
    print(f"\nIdentified {len(pending_list)} pending experiments for Suite B and Suite C.")
    print("DQN base experiments strictly excluded.")

    # Execution Loop
    for i, exp in enumerate(pending_list, start=1):
        algo = exp["algo"]
        mode = exp["data_mode"]
        f_set = exp["feature_set"]
        p_step = exp["planning_steps"]
        s_name = exp["suite"]

        fname = get_experiment_filename(algo, mode, f_set, episodes, p_step)
        fpath = results_dir / fname

        if verify_file_completed(fpath, expected_seeds, episodes):
            print(f"[{i}/{len(pending_list)}] Skipping already completed: {fname}")
            continue

        print(f"\n[{i}/{len(pending_list)}] Running: {algo.upper()} | Mode: {mode} | Features: {f_set} | Plan: {p_step}")
        res = train_single_experiment(
            algo=algo,
            data_mode=mode,
            feature_set=f_set,
            seeds=expected_seeds,
            episodes=episodes,
            planning_steps=p_step,
            eval_episodes_count=eval_episodes_count,
            config=cfg,
            dataset=dataset,
            output_dir=results_dir,
        )
        print(f"Finished & Saved: {fname} (Test Reward: {res['mean_test_reward']:.2f})")

    print("\n" + "=" * 80)
    print("ALL APPROVED SUITE B & SUITE C EXPERIMENTS COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

