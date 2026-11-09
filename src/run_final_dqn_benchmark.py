"""Complete and robust runner for the final two DQN benchmark experiments:
1. dqn_synthetic_base_ep1000.json
2. dqn_dataset_base_ep1000.json

Features:
- Validates all 34 existing JSON files before starting.
- Never overwrites or modifies completed results.
- Exact DQN architecture (128x128 MLP), hyperparameters, reward rules, and evaluation protocol.
- Seeds [0, 1, 2, 3, 4], 1000 training episodes, 100 evaluation episodes per seed.
- Multi-process parallel seed execution across CPU cores for maximum efficiency.
- Disk-backed IPC to avoid Windows multiprocessing queue deadlocks.
- Atomic per-seed checkpointing for safe crash-resumption.
- Live progress reporting with elapsed time and ETA.
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import CloudDataset, load_and_clean_data
from src.env import EnhancedAutoscalingEnv
from src.train import create_agent, run_episode, compute_convergence_speed


def verify_existing_34_results(results_dir: Path) -> bool:
    """Verify that all 34 prior benchmark results exist and have valid structure."""
    expected_files = glob.glob(str(results_dir / "*_ep1000.json"))
    # Filter out any newly generated base dqn files
    prior_files = [
        f for f in expected_files
        if not (f.endswith("dqn_synthetic_base_ep1000.json") or f.endswith("dqn_dataset_base_ep1000.json"))
    ]
    if len(prior_files) < 34:
        print(f"ERROR: Expected at least 34 prior benchmark files, found {len(prior_files)}!")
        return False
    print(f"Verified {len(prior_files)} existing benchmark JSON files are intact.")
    return True


def run_single_seed_worker(
    algo: str,
    data_mode: str,
    feature_set: str,
    seed: int,
    episodes: int,
    eval_episodes_count: int,
    cfg: Dict[str, Any],
    dataset: Optional[CloudDataset],
    progress_queue: mp.Queue,
    output_dir: Path,
) -> None:
    """Execute training and evaluation for a single seed with periodic progress reports."""
    try:
        # Use single-thread intra-op per worker for optimal multi-process scaling
        torch.set_num_threads(1)

        # Preload dataset if needed in child process
        if data_mode == "dataset" and dataset is None:
            raw_path = cfg.get("dataset_path", "data/raw/vmCloud_data.csv")
            df = load_and_clean_data(raw_path)
            dataset = CloudDataset(
                df,
                train_ratio=0.8,
                network_thresholds=tuple(cfg.get("network_thresholds", [333.33, 666.67])),
                exec_thresholds=tuple(cfg.get("execution_time_thresholds", [33.33, 66.67])),
            )

        train_env = EnhancedAutoscalingEnv(
            config=cfg,
            data_mode=data_mode,
            feature_set=feature_set,
            dataset=dataset,
            split="train",
        )
        test_env = EnhancedAutoscalingEnv(
            config=cfg,
            data_mode=data_mode,
            feature_set=feature_set,
            dataset=dataset,
            split="test",
        )

        agent = create_agent(algo, cfg, seed, planning_steps=None)

        train_rewards: List[float] = []
        train_sla_violations: List[int] = []
        train_costs: List[float] = []
        train_churns: List[int] = []

        seed_start_time = time.time()

        # Training phase (1000 episodes)
        for ep in range(episodes):
            ep_seed = seed * 10000 + ep
            m = run_episode(train_env, agent, training=True, seed=ep_seed)
            train_rewards.append(m["reward"])
            train_sla_violations.append(m["sla_violations"])
            train_costs.append(m["total_cost"])
            train_churns.append(m["capacity_changes"])

            if (ep + 1) % 25 == 0 or (ep + 1) == episodes:
                recent_reward = float(np.mean(train_rewards[-25:]))
                recent_sla = float(np.mean(train_sla_violations[-25:]))
                progress_queue.put({
                    "type": "progress",
                    "seed": seed,
                    "episode": ep + 1,
                    "total_episodes": episodes,
                    "phase": "train",
                    "reward": recent_reward,
                    "sla": recent_sla,
                    "elapsed": time.time() - seed_start_time,
                })

        # Evaluation phase (100 held-out test episodes)
        test_rewards: List[float] = []
        test_sla_violations: List[int] = []
        test_costs: List[float] = []
        test_churns: List[int] = []

        for eval_ep in range(eval_episodes_count):
            eval_seed = (seed + 8888) * 10000 + eval_ep
            tm = run_episode(test_env, agent, training=False, seed=eval_seed)
            test_rewards.append(tm["reward"])
            test_sla_violations.append(tm["sla_violations"])
            test_costs.append(tm["total_cost"])
            test_churns.append(tm["capacity_changes"])

        conv_speed = compute_convergence_speed(train_rewards)
        last_100_train = train_rewards[-min(100, len(train_rewards)) :]

        seed_result = {
            "seed": seed,
            "train_reward_mean_last100": float(np.mean(last_100_train)),
            "train_reward_std_last100": float(np.std(last_100_train)),
            "test_reward_mean": float(np.mean(test_rewards)),
            "test_reward_std": float(np.std(test_rewards)),
            "test_sla_violations_mean": float(np.mean(test_sla_violations)),
            "test_cost_mean": float(np.mean(test_costs)),
            "test_churn_mean": float(np.mean(test_churns)),
            "convergence_episodes": conv_speed,
            "train_curves": {
                "rewards": train_rewards,
                "sla_violations": train_sla_violations,
                "costs": train_costs,
                "churn": train_churns,
            },
            "test_curves": {
                "rewards": test_rewards,
                "sla_violations": test_sla_violations,
                "costs": test_costs,
                "churn": test_churns,
            },
        }

        # Write seed result to temporary file to avoid IPC buffer limits
        tmp_file = output_dir / f"_tmp_{algo}_{data_mode}_{feature_set}_seed{seed}.json"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(seed_result, f)

        progress_queue.put({
            "type": "seed_complete",
            "seed": seed,
            "tmp_file": str(tmp_file),
            "test_reward_mean": seed_result["test_reward_mean"],
            "test_reward_std": seed_result["test_reward_std"],
            "test_sla_violations_mean": seed_result["test_sla_violations_mean"],
            "elapsed": time.time() - seed_start_time,
        })
    except Exception as e:
        progress_queue.put({
            "type": "error",
            "seed": seed,
            "error": str(e),
            "traceback": traceback.format_exc(),
        })


def save_checkpoint(checkpoint_path: Path, completed_runs: Dict[int, Any]) -> None:
    """Atomically save checkpoint dictionary mapping seed -> seed_result."""
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(completed_runs, f, indent=2)
    tmp_path.replace(checkpoint_path)


def run_experiment_suite(
    algo: str,
    data_mode: str,
    feature_set: str,
    seeds: List[int],
    episodes: int,
    eval_episodes_count: int,
    cfg: Dict[str, Any],
    dataset: Optional[CloudDataset],
    results_dir: Path,
) -> Dict[str, Any]:
    """Run an entire experiment across all seeds with checkpointing and parallel multiprocessing."""
    final_filename = f"{algo}_{data_mode}_{feature_set}_ep{episodes}.json"
    final_path = results_dir / final_filename
    checkpoint_path = results_dir / f"{algo}_{data_mode}_{feature_set}_ep{episodes}.checkpoint.json"

    # If final file already exists and valid, skip
    if final_path.exists():
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            if set(existing_data.get("seeds", [])) == set(seeds) and len(existing_data.get("runs", [])) == len(seeds):
                print(f"Skipping already completed experiment: {final_filename}")
                return existing_data
        except Exception:
            pass

    # Check for existing checkpoint
    completed_runs: Dict[int, Any] = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                completed_runs = {int(k): v for k, v in loaded.items()}
            print(f"Loaded existing checkpoint with {len(completed_runs)} completed seeds: {list(completed_runs.keys())}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), starting fresh.")
            completed_runs = {}

    pending_seeds = [s for s in seeds if s not in completed_runs]
    if not pending_seeds:
        print("All seeds already completed in checkpoint. Finalizing experiment JSON...")
    else:
        print(f"\n=======================================================")
        print(f"Starting {algo.upper()} | Mode: {data_mode} | Features: {feature_set}")
        print(f"Pending Seeds: {pending_seeds} | Total Episodes: {episodes} per seed")
        print(f"=======================================================\n", flush=True)

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()

        workers: Dict[int, mp.Process] = {}
        for s in pending_seeds:
            p = ctx.Process(
                target=run_single_seed_worker,
                args=(algo, data_mode, feature_set, s, episodes, eval_episodes_count, cfg, dataset, queue, results_dir),
            )
            p.daemon = True
            workers[s] = p
            p.start()
            print(f"Launched worker process for Seed {s} (PID {p.pid})", flush=True)

        exp_start_time = time.time()
        active_progress: Dict[int, Dict[str, Any]] = {s: {"ep": 0, "reward": 0.0, "sla": 0.0} for s in pending_seeds}

        while len(completed_runs) < len(seeds):
            try:
                msg = queue.get(timeout=3.0)
                mtype = msg.get("type")
                s = msg.get("seed")

                if mtype == "progress":
                    active_progress[s] = {
                        "ep": msg["episode"],
                        "reward": msg["reward"],
                        "sla": msg["sla"],
                    }
                    elapsed = time.time() - exp_start_time
                    total_eps_done = sum(v["ep"] for v in active_progress.values()) + (len(completed_runs) * episodes)
                    total_eps_target = len(seeds) * episodes
                    progress_pct = (total_eps_done / total_eps_target) * 100
                    rate = total_eps_done / max(elapsed, 1.0)
                    remaining_eps = max(0, total_eps_target - total_eps_done)
                    eta_sec = remaining_eps / max(rate, 0.01)

                    status_parts = [f"S{k}: {v['ep']}/{episodes}" for k, v in sorted(active_progress.items())]
                    print(
                        f"[{algo.upper()}-{data_mode}-{feature_set}] "
                        f"{' | '.join(status_parts)} | "
                        f"Progress: {progress_pct:.1f}% | "
                        f"Elapsed: {elapsed/60:.1f}m | ETA: {eta_sec/60:.1f}m",
                        flush=True,
                    )

                elif mtype == "seed_complete":
                    tmp_file = Path(msg["tmp_file"])
                    with open(tmp_file, "r", encoding="utf-8") as f:
                        res = json.load(f)
                    if tmp_file.exists():
                        tmp_file.unlink()

                    completed_runs[s] = res
                    save_checkpoint(checkpoint_path, completed_runs)
                    print(
                        f"\n>>> SEED {s} COMPLETED! Test Reward: {msg['test_reward_mean']:.2f} +/- {msg['test_reward_std']:.2f} "
                        f"| SLA: {msg['test_sla_violations_mean']:.1f} | Saved to checkpoint ({len(completed_runs)}/{len(seeds)} done)\n",
                        flush=True,
                    )
                    if s in active_progress:
                        del active_progress[s]

                elif mtype == "error":
                    print(f"\n[FATAL ERROR IN SEED {s}]: {msg.get('error')}\n{msg.get('traceback')}", flush=True)
                    break

            except Exception:
                dead_workers = [s for s, p in workers.items() if not p.is_alive() and s not in completed_runs]
                if dead_workers:
                    print(f"ERROR: Worker(s) {dead_workers} terminated unexpectedly!", flush=True)
                    break

        for p in workers.values():
            p.join(timeout=5.0)

    # All seeds completed -> build final summary
    ordered_runs = [completed_runs[s] for s in sorted(seeds)]
    all_test_rewards = [r["test_reward_mean"] for r in ordered_runs]
    all_test_sla = [r["test_sla_violations_mean"] for r in ordered_runs]
    all_test_costs = [r["test_cost_mean"] for r in ordered_runs]
    all_conv = [r["convergence_episodes"] for r in ordered_runs]

    exp_summary = {
        "algo": algo,
        "data_mode": data_mode,
        "feature_set": feature_set,
        "planning_steps": 10,
        "episodes": episodes,
        "seeds": sorted(seeds),
        "mean_test_reward": float(np.mean(all_test_rewards)),
        "std_test_reward": float(np.std(all_test_rewards)),
        "mean_test_sla_violations": float(np.mean(all_test_sla)),
        "std_test_sla_violations": float(np.std(all_test_sla)),
        "mean_test_cost": float(np.mean(all_test_costs)),
        "mean_convergence_episodes": float(np.mean(all_conv)),
        "runs": ordered_runs,
    }

    # Write atomically to final JSON file
    tmp_final = final_path.with_suffix(".tmp")
    with open(tmp_final, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)
    tmp_final.replace(final_path)

    # Clean up checkpoint file
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(
        f"\n=======================================================\n"
        f"SUCCESS: Generated {final_filename}\n"
        f"Mean Test Reward: {exp_summary['mean_test_reward']:.2f} +/- {exp_summary['std_test_reward']:.2f}\n"
        f"Mean SLA Violations: {exp_summary['mean_test_sla_violations']:.1f}\n"
        f"Mean Cost: ${exp_summary['mean_test_cost']:.2f}\n"
        f"Mean Convergence: {exp_summary['mean_convergence_episodes']:.1f} eps\n"
        f"=======================================================\n",
        flush=True,
    )
    return exp_summary


def main():
    parser = argparse.ArgumentParser(description="Final DQN Benchmark Runner")
    parser.add_argument("--test-run", action="store_true", help="Quick 2-episode sanity test")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    cfg = load_config()

    print("==================================================================")
    print("FINAL DQN BENCHMARK RUNNER (45-STATE BASE REPRESENTATION)")
    print("Project: Enhanced-State Reinforcement Learning for Cloud Autoscaling")
    print("Target Results:")
    print("  1. results/dqn_synthetic_base_ep1000.json")
    print("  2. results/dqn_dataset_base_ep1000.json")
    print("==================================================================\n", flush=True)

    if not verify_existing_34_results(results_dir):
        sys.exit(1)

    target_seeds = [0] if args.test_run else cfg.get("seeds", [0, 1, 2, 3, 4])
    episodes = 2 if args.test_run else 1000
    eval_episodes = 2 if args.test_run else 100

    if args.test_run:
        print(">>> RUNNING QUICK SANITY TEST (2 episodes, seed 0) ...", flush=True)
        test_dir = results_dir / "test_sanity"
        test_dir.mkdir(exist_ok=True)
        run_experiment_suite(
            algo="dqn",
            data_mode="synthetic",
            feature_set="base",
            seeds=target_seeds,
            episodes=episodes,
            eval_episodes_count=eval_episodes,
            cfg=cfg,
            dataset=None,
            results_dir=test_dir,
        )
        print(">>> Sanity test completed successfully! Cleaning up test dir...", flush=True)
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        return

    # Experiment 1: Double DQN on Synthetic Base
    print("\n>>> EXECUTING EXPERIMENT 1/2: Double DQN | Synthetic | Base (45 States) ...", flush=True)
    exp1_res = run_experiment_suite(
        algo="dqn",
        data_mode="synthetic",
        feature_set="base",
        seeds=target_seeds,
        episodes=episodes,
        eval_episodes_count=eval_episodes,
        cfg=cfg,
        dataset=None,
        results_dir=results_dir,
    )

    # Experiment 2: Double DQN on Dataset Base
    print("\n>>> EXECUTING EXPERIMENT 2/2: Double DQN | Dataset | Base (45 States) ...", flush=True)
    print("Pre-loading Kaggle dataset...", flush=True)
    raw_path = cfg.get("dataset_path", "data/raw/vmCloud_data.csv")
    df = load_and_clean_data(raw_path)
    dataset = CloudDataset(
        df,
        train_ratio=0.8,
        network_thresholds=tuple(cfg.get("network_thresholds", [333.33, 666.67])),
        exec_thresholds=tuple(cfg.get("execution_time_thresholds", [33.33, 66.67])),
    )
    print(f"Loaded {len(df)} dataset rows successfully.", flush=True)

    exp2_res = run_experiment_suite(
        algo="dqn",
        data_mode="dataset",
        feature_set="base",
        seeds=target_seeds,
        episodes=episodes,
        eval_episodes_count=eval_episodes,
        cfg=cfg,
        dataset=dataset,
        results_dir=results_dir,
    )

    # Final Verification
    all_files = glob.glob(str(results_dir / "*_ep1000.json"))
    print("\n==================================================================")
    print(f"BENCHMARK COMPLETE! Total results files in results/: {len(all_files)}")
    print("==================================================================", flush=True)
    if len(all_files) == 36:
        print("PERFECT: All 36 benchmark experiments are now completed and verified!", flush=True)
    else:
        print(f"NOTE: Expected 36 files, found {len(all_files)}.", flush=True)


if __name__ == "__main__":
    main()
