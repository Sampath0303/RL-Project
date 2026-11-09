"""Training script for single algorithm runs with CLI, metrics, evaluation, and JSON serialization."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config
from src.data import CloudDataset, load_and_clean_data
from src.env import EnhancedAutoscalingEnv
from src.agents.baselines import RandomPolicy, ThresholdPolicy
from src.agents.tabular import QLearningAgent, SARSAAgent
from src.agents.dyna_q import DynaQAgent
from src.agents.dqn import DQNAgent


def create_agent(
    algo: str,
    config: Dict[str, Any],
    seed: int,
    planning_steps: Optional[int] = None,
) -> Any:
    """Factory to instantiate agent matching requested algorithm and hyperparameters."""
    algo_key = algo.lower().replace("-", "").replace("_", "")
    hyper = config.get("hyperparameters", {})

    if algo_key in ["random", "randompolicy"]:
        return RandomPolicy(n_actions=3, seed=seed)

    elif algo_key in ["threshold", "thresholdpolicy"]:
        return ThresholdPolicy(n_actions=3, seed=seed)

    elif algo_key in ["qlearning", "qlearningagent", "q"]:
        tab = hyper.get("tabular", {})
        return QLearningAgent(
            num_states=405,
            num_actions=3,
            alpha=tab.get("alpha", 0.1),
            gamma=tab.get("gamma", 0.99),
            epsilon=tab.get("epsilon_start", 1.0),
            epsilon_min=tab.get("epsilon_min", 0.01),
            epsilon_decay=tab.get("epsilon_decay", 0.995),
            seed=seed,
        )

    elif algo_key in ["sarsa", "sarsaagent"]:
        tab = hyper.get("tabular", {})
        return SARSAAgent(
            num_states=405,
            num_actions=3,
            alpha=tab.get("alpha", 0.1),
            gamma=tab.get("gamma", 0.99),
            epsilon=tab.get("epsilon_start", 1.0),
            epsilon_min=tab.get("epsilon_min", 0.01),
            epsilon_decay=tab.get("epsilon_decay", 0.995),
            seed=seed,
        )

    elif algo_key in ["dynaq", "dynaqagent", "dyna"]:
        dq_cfg = hyper.get("dyna_q", {})
        p_steps = (
            planning_steps
            if planning_steps is not None
            else dq_cfg.get("planning_steps", 10)
        )
        return DynaQAgent(
            num_states=405,
            num_actions=3,
            alpha=dq_cfg.get("alpha", 0.1),
            gamma=dq_cfg.get("gamma", 0.99),
            epsilon=dq_cfg.get("epsilon_start", 1.0),
            epsilon_min=dq_cfg.get("epsilon_min", 0.01),
            epsilon_decay=dq_cfg.get("epsilon_decay", 0.995),
            planning_steps=p_steps,
            seed=seed,
        )

    elif algo_key in ["dqn", "dqnagent"]:
        dqn_cfg = hyper.get("dqn", {})
        return DQNAgent(
            state_dim=5,
            action_dim=3,
            hidden_dims=tuple(dqn_cfg.get("hidden_dims", [128, 128])),
            learning_rate=dqn_cfg.get("learning_rate", 5e-4),
            gamma=dqn_cfg.get("gamma", 0.99),
            tau=dqn_cfg.get("tau", 1e-3),
            epsilon=dqn_cfg.get("epsilon_start", 1.0),
            epsilon_min=dqn_cfg.get("epsilon_min", 0.01),
            epsilon_decay=dqn_cfg.get("epsilon_decay", 0.995),
            buffer_size=dqn_cfg.get("buffer_size", 100000),
            batch_size=dqn_cfg.get("batch_size", 64),
            double_dqn=dqn_cfg.get("double_dqn", False),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def run_episode(
    env: EnhancedAutoscalingEnv,
    agent: Any,
    training: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute one complete episode and return performance metrics."""
    state, info = env.reset(seed=seed)
    total_reward = 0.0
    sla_violations = 0
    total_cost = 0.0
    capacity_changes = 0
    steps = 0
    done = False

    is_sarsa = isinstance(agent, SARSAAgent)
    action = agent.select_action(state, training=training)

    while not done:
        next_state, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        sla_violations += int(step_info.get("sla_violation", 0))
        total_cost += float(step_info.get("total_cost", 0.0))
        capacity_changes += int(step_info.get("capacity_changes", 0))
        steps += 1

        next_action = agent.select_action(next_state, training=training) if not done else 0

        if training and hasattr(agent, "update"):
            if is_sarsa:
                agent.update(state, action, reward, next_state, done, next_action)
            else:
                agent.update(state, action, reward, next_state, done)

        state = next_state
        action = next_action

    if training and hasattr(agent, "decay_epsilon"):
        agent.decay_epsilon()

    return {
        "reward": float(total_reward),
        "steps": int(steps),
        "sla_violations": int(sla_violations),
        "sla_rate": float(sla_violations / max(steps, 1)),
        "total_cost": float(total_cost),
        "capacity_changes": int(capacity_changes),
    }


def compute_convergence_speed(train_rewards: List[float], window: int = 20) -> int:
    """Calculate the episode number where performance first reaches 90% of final mean performance."""
    if len(train_rewards) < window * 2:
        return len(train_rewards)

    # Smooth curve using rolling mean
    smoothed = np.convolve(train_rewards, np.ones(window) / window, mode="valid")
    final_perf = float(np.mean(train_rewards[-window:]))
    initial_perf = float(np.mean(train_rewards[:window]))

    # Target threshold is 90% of the gain from initial to final
    target_perf = initial_perf + 0.90 * (final_perf - initial_perf)

    if final_perf <= initial_perf:
        return len(train_rewards)

    indices = np.where(smoothed >= target_perf)[0]
    if len(indices) > 0:
        return int(indices[0] + window)
    return len(train_rewards)


def train_single_experiment(
    algo: str,
    data_mode: str,
    feature_set: str,
    seeds: List[int],
    episodes: int,
    planning_steps: Optional[int] = None,
    eval_episodes_count: int = 100,
    config: Optional[Dict[str, Any]] = None,
    dataset: Optional[CloudDataset] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Train algorithm across multiple seeds, evaluate greedily on held-out test episodes,

    and write results JSON to results/.
    """
    cfg = config or load_config()

    # Preload dataset if in dataset mode
    if data_mode == "dataset" and dataset is None:
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

    # Results directory
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_runs: List[Dict[str, Any]] = []

    for seed in seeds:
        # Environments
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

        agent = create_agent(algo, cfg, seed, planning_steps=planning_steps)

        train_rewards: List[float] = []
        train_sla_violations: List[int] = []
        train_costs: List[float] = []
        train_churns: List[int] = []

        # Training phase
        for ep in range(episodes):
            ep_seed = seed * 10000 + ep
            m = run_episode(train_env, agent, training=True, seed=ep_seed)
            train_rewards.append(m["reward"])
            train_sla_violations.append(m["sla_violations"])
            train_costs.append(m["total_cost"])
            train_churns.append(m["capacity_changes"])

        # Evaluation phase (training=False) on 100 held-out test episodes
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

        # Final 100 episodes train slice
        last_100_train = train_rewards[-min(100, len(train_rewards)) :]

        seed_runs.append(
            {
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
        )

    # Aggregation across seeds
    all_test_rewards = [r["test_reward_mean"] for r in seed_runs]
    all_test_sla = [r["test_sla_violations_mean"] for r in seed_runs]
    all_test_costs = [r["test_cost_mean"] for r in seed_runs]
    all_conv = [r["convergence_episodes"] for r in seed_runs]

    exp_summary = {
        "algo": algo,
        "data_mode": data_mode,
        "feature_set": feature_set,
        "planning_steps": planning_steps if planning_steps is not None else 10,
        "episodes": episodes,
        "seeds": seeds,
        "mean_test_reward": float(np.mean(all_test_rewards)),
        "std_test_reward": float(np.std(all_test_rewards)),
        "mean_test_sla_violations": float(np.mean(all_test_sla)),
        "std_test_sla_violations": float(np.std(all_test_sla)),
        "mean_test_cost": float(np.mean(all_test_costs)),
        "mean_convergence_episodes": float(np.mean(all_conv)),
        "runs": seed_runs,
    }

    # Unique JSON filename
    plan_suffix = (
        f"_plan{planning_steps}" if planning_steps is not None else ""
    )
    json_filename = f"{algo}_{data_mode}_{feature_set}{plan_suffix}_ep{episodes}.json"
    save_path = output_dir / json_filename
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(exp_summary, f, indent=2)

    print(
        f"Saved: {save_path.name} | Test Reward: {exp_summary['mean_test_reward']:.2f} +/- "
        f"{exp_summary['std_test_reward']:.2f} | SLA: {exp_summary['mean_test_sla_violations']:.1f}"
    )

    return exp_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train Cloud Autoscaling RL Agent")
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["random", "threshold", "qlearning", "sarsa", "dynaq", "dqn"],
        help="Algorithm to run",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="Random seeds")
    parser.add_argument(
        "--feature-set",
        type=str,
        default="full",
        choices=["base", "base+network", "base+exec", "full"],
        help="Feature space set",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        default="synthetic",
        choices=["synthetic", "dataset"],
        help="Workload data mode",
    )
    parser.add_argument(
        "--planning-steps",
        type=int,
        default=None,
        help="Dyna-Q planning steps (optional)",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save JSON results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    train_single_experiment(
        algo=args.algo,
        data_mode=args.data_mode,
        feature_set=args.feature_set,
        seeds=args.seeds,
        episodes=args.episodes,
        planning_steps=args.planning_steps,
        config=cfg,
        output_dir=args.output_dir,
    )
