"""Statistical analysis script for Cloud Autoscaling RL Benchmark.

Calculates:
- Mean, Standard Deviation, and 95% Confidence Intervals
- Paired t-tests across matching seeds [0, 1, 2, 3, 4]
- Effect sizes (Cohen's d_z for paired samples)
- Formal hypothesis testing for:
  1. Contribution 1: Enhanced 5-feature (405 states) vs Original 3-feature (45 states)
  2. Contribution 2: Dyna-Q (model-based planning) vs Q-Learning, SARSA, DQN
  3. Planning sensitivity: Dyna-Q plan=10 vs plan=0, 5, 20, 50
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


def find_ep1000_file(
    results_dir: Path,
    algo: str,
    mode: str,
    feature_set: str,
    planning_steps: Optional[int] = None,
    episodes: int = 1000,
) -> Optional[Path]:
    """Find specific experiment JSON file with exact component matching."""
    plan_pattern = rf"_plan{planning_steps}" if planning_steps is not None else (r"(_plan10|_plan\d+)?" if algo == "dynaq" else r"(_plan\d+)?")
    regex = re.compile(rf"^{re.escape(algo)}_{re.escape(mode)}_{re.escape(feature_set)}{plan_pattern}_ep{episodes}\.json$")
    candidates = []
    for p in results_dir.glob("*.json"):
        if regex.match(p.name):
            candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (1 if "_plan10" in p.stem else 0, p.stat().st_mtime), reverse=True)
    return candidates[0]


def load_seed_metrics(json_path: Path) -> Dict[str, np.ndarray]:
    """Load per-seed metrics array from experiment JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", [])
    rewards = np.array([r["test_reward_mean"] for r in runs], dtype=np.float64)
    sla = np.array([r["test_sla_violations_mean"] for r in runs], dtype=np.float64)
    costs = np.array([r["test_cost_mean"] for r in runs], dtype=np.float64)
    conv = np.array([r["convergence_episodes"] for r in runs], dtype=np.float64)
    seeds = [r["seed"] for r in runs]
    return {
        "rewards": rewards,
        "sla": sla,
        "costs": costs,
        "conv": conv,
        "seeds": seeds,
        "algo": data.get("algo"),
        "mode": data.get("data_mode"),
        "feature_set": data.get("feature_set"),
        "planning_steps": data.get("planning_steps"),
    }


def compute_ci(arr: np.ndarray, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Calculate mean, standard deviation, and half-width of 95% Confidence Interval."""
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0, 0.0
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    h = se * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return mean, std, float(h)


def paired_test(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Run paired two-tailed t-test and compute Cohen's d_z."""
    diff = a - b
    mean_diff = float(np.mean(diff))
    n = len(diff)
    if n < 2 or np.all(diff == 0):
        return {
            "mean_diff": mean_diff,
            "t_stat": 0.0,
            "p_val": 1.0,
            "cohen_d": 0.0,
            "significant": False,
        }
    std_diff = float(np.std(diff, ddof=1))
    t_stat, p_val = stats.ttest_rel(a, b)
    cohen_d = mean_diff / (std_diff + 1e-12)
    return {
        "mean_diff": mean_diff,
        "t_stat": float(t_stat),
        "p_val": float(p_val),
        "cohen_d": float(cohen_d),
        "significant": bool(p_val < 0.05),
    }


def analyze_all(results_dir: Path, episodes: int = 1000) -> str:
    out = []
    out.append(f"# Rigorous Statistical Analysis Report ({episodes} Episodes, 5 Seeds [0, 1, 2, 3, 4])\n")

    # =========================================================================
    # PART 1: CONTRIBUTION 1 - CONTROLLED STATE-SPACE COMPARISON (45 vs 405)
    # =========================================================================
    out.append("## 1. Contribution 1: State-Space Representation Comparison\n")
    out.append("Comparing Original 3-Feature (45 states) vs Enhanced 5-Feature (405 states) across matched seeds.\n")
    out.append("| Data Mode | Algorithm | Original 45-State (Mean \u00b1 95% CI) | Enhanced 405-State (Mean \u00b1 95% CI) | Mean Diff | Paired t-stat | p-value | Cohen's d_z | Significant (p<0.05)? |")
    out.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    algos = ["qlearning", "sarsa", "dynaq", "dqn"]
    for mode in ["synthetic", "dataset"]:
        for algo in algos:
            p_step = 10 if algo == "dynaq" else None
            f_base = find_ep1000_file(results_dir, algo, mode, "base", p_step, episodes)
            f_full = find_ep1000_file(results_dir, algo, mode, "full", p_step, episodes)
            if not f_base or not f_full:
                if algo == "dqn":
                    out.append(f"| **{mode.capitalize()}** | **DQN** | *PENDING (Postponed)* | *{mean_f:,.1f} \u00b1 {ci_f:.1f}* | *N/A (Pending)* | *N/A* | *N/A* | *N/A* | *PENDING* |" if f_full and (m_full := load_seed_metrics(f_full)["rewards"]) is not None and (mean_f := float(np.mean(m_full))) and (ci_f := compute_ci(m_full)[2]) else f"| **{mode.capitalize()}** | **DQN** | *PENDING (Postponed)* | *PENDING* | *N/A* | *N/A* | *N/A* | *N/A* | *PENDING* |")
                continue
            m_base = load_seed_metrics(f_base)["rewards"]
            m_full = load_seed_metrics(f_full)["rewards"]

            mean_b, std_b, ci_b = compute_ci(m_base)
            mean_f, std_f, ci_f = compute_ci(m_full)
            t_res = paired_test(m_full, m_base)

            sig_str = "**YES (Significant)**" if t_res["significant"] else "NO"
            out.append(
                f"| **{mode.capitalize()}** | **{algo.upper()}** | "
                f"{mean_b:,.1f} \u00b1 {ci_b:.1f} | {mean_f:,.1f} \u00b1 {ci_f:.1f} | "
                f"{t_res['mean_diff']:+,.1f} | {t_res['t_stat']:.3f} | {t_res['p_val']:.4f} | "
                f"{t_res['cohen_d']:.2f} | {sig_str} |"
            )
    out.append("\n")

    # =========================================================================
    # PART 2: CONTRIBUTION 2 - ALGORITHM COMPARISON (Dyna-Q vs Baselines)
    # =========================================================================
    out.append("## 2. Contribution 2: Dyna-Q (Model Planning) vs Baselines (Enhanced 405-State Space)\n")
    out.append("Paired comparisons between Dyna-Q (plan=10) and alternative policies across matching seeds.\n")
    out.append("| Data Mode | Comparison Pair | Baseline (Mean \u00b1 95% CI) | Dyna-Q (Mean \u00b1 95% CI) | Mean Gain (\u0394) | Paired t-stat | p-value | Cohen's d_z | Significant (p<0.05)? |")
    out.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    baseline_pairs = ["qlearning", "sarsa", "dqn", "threshold"]
    for mode in ["synthetic", "dataset"]:
        f_dynaq = find_ep1000_file(results_dir, "dynaq", mode, "full", 10, episodes)
        if not f_dynaq:
            continue
        m_dynaq = load_seed_metrics(f_dynaq)["rewards"]
        mean_dq, std_dq, ci_dq = compute_ci(m_dynaq)

        for base_algo in baseline_pairs:
            f_base = find_ep1000_file(results_dir, base_algo, mode, "full", None, episodes)
            if not f_base:
                continue
            m_base = load_seed_metrics(f_base)["rewards"]
            mean_b, std_b, ci_b = compute_ci(m_base)
            t_res = paired_test(m_dynaq, m_base)
            sig_str = "**YES (Significant)**" if t_res["significant"] else "NO"

            out.append(
                f"| **{mode.capitalize()}** | Dyna-Q vs **{base_algo.upper()}** | "
                f"{mean_b:,.1f} \u00b1 {ci_b:.1f} | {mean_dq:,.1f} \u00b1 {ci_dq:.1f} | "
                f"{t_res['mean_diff']:+,.1f} | {t_res['t_stat']:.3f} | {t_res['p_val']:.4f} | "
                f"{t_res['cohen_d']:.2f} | {sig_str} |"
            )
    out.append("\n")

    # =========================================================================
    # PART 3: SENSITIVITY TO PLANNING STEPS IN DYNA-Q
    # =========================================================================
    out.append("## 3. Dyna-Q Planning Step Sensitivity (k = 0, 5, 10, 20, 50)\n")
    out.append("| Data Mode | Planning Steps (k) | Mean Test Reward \u00b1 95% CI | Test SLA Violations \u00b1 Std | Resource Cost | Convergence (eps) | Paired Gain vs k=0 | p-value (vs k=0) |")
    out.append("| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |")

    for mode in ["synthetic", "dataset"]:
        f_k0 = find_ep1000_file(results_dir, "dynaq", mode, "full", 0, episodes)
        m_k0 = load_seed_metrics(f_k0)["rewards"] if f_k0 else None

        for k in [0, 5, 10, 20, 50]:
            f_k = find_ep1000_file(results_dir, "dynaq", mode, "full", k, episodes)
            if not f_k:
                continue
            metrics = load_seed_metrics(f_k)
            r_mean, r_std, r_ci = compute_ci(metrics["rewards"])
            sla_m, sla_s, _ = compute_ci(metrics["sla"])
            cost_m, _, _ = compute_ci(metrics["costs"])
            conv_m, _, _ = compute_ci(metrics["conv"])

            if m_k0 is not None and k != 0:
                t_k = paired_test(metrics["rewards"], m_k0)
                diff_str = f"{t_k['mean_diff']:+,.1f}"
                p_str = f"{t_k['p_val']:.4f}"
            else:
                diff_str = "Baseline"
                p_str = "-"

            out.append(
                f"| **{mode.capitalize()}** | **{k}** | {r_mean:,.1f} \u00b1 {r_ci:.1f} | "
                f"{sla_m:.1f} \u00b1 {sla_s:.1f} | {cost_m:,.1f} | {conv_m:.0f} | {diff_str} | {p_str} |"
            )
    out.append("\n")

    report_str = "\n".join(out)
    return report_str


def main():
    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    
    # Try 1000 episodes first, fallback to available
    episodes = 1000
    report = analyze_all(results_dir, episodes=episodes)
    print(report)

    out_file = results_dir / f"statistical_analysis_ep{episodes}.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved statistical report to: {out_file}")


if __name__ == "__main__":
    main()

