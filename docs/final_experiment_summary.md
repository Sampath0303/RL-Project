# Experimental Architecture & Benchmark Protocol: Enhanced Cloud Autoscaling RL

---

## 1. Experimental Benchmark Architecture

This document defines the complete experimental evaluation architecture, benchmark matrix, metric definitions, and statistical protocols for the **Enhanced-State Cloud Autoscaling RL** investigation.

To eliminate common pitfalls in reinforcement learning benchmarks—such as single-seed bias, unrepresentative evaluation horizons, and confounding hyperparameters—all experiments adhere to a standardized, multi-seed evaluation protocol:
- **Seed Budget**: 5 independent, fixed random seeds: $\mathcal{S}_{\text{seeds}} = \{0, 1, 2, 3, 4\}$.
- **Training Budget**: 1,000 full training episodes per seed ($1,000 \times 200 = 200,000$ environment transitions per seed; $1,000,000$ transitions per configuration).
- **Evaluation Budget**: 100 held-out evaluation episodes per seed executed strictly greedily ($\epsilon = 0$) to measure true policy quality without exploration noise.
- **Evaluation Splits**:
  - *Synthetic Workload*: Evaluated on a continuous held-out temporal evaluation horizon.
  - *Kaggle Dataset*: Evaluated on the strictly separated 20% temporal test split (137,430 rows).

---

## 2. The Complete 36-Configuration Benchmark Matrix

The evaluation spans 36 distinct configurations organized into three targeted experimental suites:

```
====================================================================================================
SUITE A: CORE ALGORITHM BENCHMARK (Full 405-State Representation)
Goal: Benchmark model-free tabular, deep RL, model-based RL, and heuristic baselines on the full state space.
Episodes: 1,000 | Seeds: [0, 1, 2, 3, 4] | Eval Episodes: 100
====================================================================================================
Config 01: Random Policy           | Synthetic Workload | Full State (405)
Config 02: Random Policy           | Kaggle Dataset     | Full State (405)
Config 03: Threshold Rule Baseline | Synthetic Workload | Full State (405)
Config 04: Threshold Rule Baseline | Kaggle Dataset     | Full State (405)
Config 05: Q-Learning              | Synthetic Workload | Full State (405)
Config 06: Q-Learning              | Kaggle Dataset     | Full State (405)
Config 07: SARSA                   | Synthetic Workload | Full State (405)
Config 08: SARSA                   | Kaggle Dataset     | Full State (405)
Config 09: Dyna-Q (plan = 10)      | Synthetic Workload | Full State (405)
Config 10: Dyna-Q (plan = 10)      | Kaggle Dataset     | Full State (405)
Config 11: Double DQN              | Synthetic Workload | Full State (405 continuous)
Config 12: Double DQN              | Kaggle Dataset     | Full State (405 continuous)

====================================================================================================
SUITE B: STATE-SPACE & FEATURE ABLATION STUDY
Goal: Isolate the exact causal impact of each telemetry feature (Base vs. +Net vs. +Exec vs. Full).
Episodes: 1,000 | Seeds: [0, 1, 2, 3, 4] | Eval Episodes: 100
====================================================================================================
Config 13: Q-Learning              | Synthetic Workload | Base State (45 states)
Config 14: Q-Learning              | Synthetic Workload | Base + Network (135 states)
Config 15: Q-Learning              | Synthetic Workload | Base + Execution Time (135 states)
Config 16: SARSA                   | Synthetic Workload | Base State (45 states)
Config 17: Dyna-Q (plan = 10)      | Synthetic Workload | Base State (45 states)
Config 18: Dyna-Q (plan = 10)      | Synthetic Workload | Base + Network (135 states)
Config 19: Dyna-Q (plan = 10)      | Synthetic Workload | Base + Execution Time (135 states)
Config 20: Double DQN              | Synthetic Workload | Base State (45 states)
Config 21: Q-Learning              | Kaggle Dataset     | Base State (45 states)
Config 22: Q-Learning              | Kaggle Dataset     | Base + Network (135 states)
Config 23: Q-Learning              | Kaggle Dataset     | Base + Execution Time (135 states)
Config 24: SARSA                   | Kaggle Dataset     | Base State (45 states)
Config 25: Dyna-Q (plan = 10)      | Kaggle Dataset     | Base State (45 states)
Config 26: Dyna-Q (plan = 10)      | Kaggle Dataset     | Base + Network (135 states)
Config 27: Dyna-Q (plan = 10)      | Kaggle Dataset     | Base + Execution Time (135 states)
Config 28: Double DQN              | Kaggle Dataset     | Base State (45 states)

====================================================================================================
SUITE C: DYNA-Q PLANNING SENSITIVITY STUDY
Goal: Evaluate the effect of background simulated planning steps k in {0, 5, 10, 20, 50}.
Episodes: 1,000 | Seeds: [0, 1, 2, 3, 4] | Eval Episodes: 100 | State Space: Full (405)
(Note: k = 10 is re-used from Suite A to avoid redundant computation)
====================================================================================================
Config 29: Dyna-Q (plan = 0)       | Synthetic Workload | Full State (405)
Config 30: Dyna-Q (plan = 5)       | Synthetic Workload | Full State (405)
Config 31: Dyna-Q (plan = 20)      | Synthetic Workload | Full State (405)
Config 32: Dyna-Q (plan = 50)      | Synthetic Workload | Full State (405)
Config 33: Dyna-Q (plan = 0)       | Kaggle Dataset     | Full State (405)
Config 34: Dyna-Q (plan = 5)       | Kaggle Dataset     | Full State (405)
Config 35: Dyna-Q (plan = 20)      | Kaggle Dataset     | Full State (405)
Config 36: Dyna-Q (plan = 50)      | Kaggle Dataset     | Full State (405)
```

---

## 3. Rigorous Metric Definitions

For every seed run $i \in \{0, 1, 2, 3, 4\}$ of an experiment, performance is measured across five orthogonal dimensions over the 100 evaluation episodes ($e = 1, \dots, 100$, each of duration $T = 200$ steps):

### 3.1 Mean Evaluation Reward ($\bar{R}_{\text{eval}}$)
The average undiscounted cumulative return per evaluation episode:
$$\bar{R}_{\text{eval}}^{(i)} = \frac{1}{100} \sum_{e=1}^{100} \sum_{t=0}^{199} R_{e, t}$$
Overall reported metric across 5 seeds:
$$\mu_R = \frac{1}{5} \sum_{i=0}^4 \bar{R}_{\text{eval}}^{(i)}, \quad \sigma_R = \sqrt{\frac{1}{4} \sum_{i=0}^4 \left(\bar{R}_{\text{eval}}^{(i)} - \mu_R\right)^2}$$

### 3.2 SLA Violation Count ($\bar{V}_{\text{SLA}}$)
An SLA violation occurs at any time step where instantaneous CPU utilization exceeds the 80% saturation threshold ($U_{e, t} > 80\%$):
$$V_{e} = \sum_{t=0}^{199} \mathbb{I}(U_{e, t} > 80.0)$$
$$\bar{V}_{\text{SLA}}^{(i)} = \frac{1}{100} \sum_{e=1}^{100} V_e$$
A lower value indicates superior Quality of Service (QoS) and fewer dropped requests.

### 3.3 Resource Infrastructure Cost ($\bar{C}_{\text{cost}}$)
Measures cumulative active server-hour billing. Each active VM instance incurs $0.5$ cost units per time step:
$$\bar{C}_{\text{cost}}^{(i)} = \frac{1}{100} \sum_{e=1}^{100} \sum_{t=0}^{199} \left( 0.5 \times C_{e, t+1} \right)$$
A lower value reflects greater energy and monetary efficiency.

### 3.4 Reconfiguration Churn ($\bar{K}_{\text{churn}}$)
Quantifies cluster instability by measuring the frequency of non-zero scaling actions ($|a_{e, t}| > 0$):
$$\bar{K}_{\text{churn}}^{(i)} = \frac{1}{100} \sum_{e=1}^{100} \sum_{t=0}^{199} \mathbb{I}(a_{e, t} \ne 0)$$
Lower churn indicates a stable, steady-state policy that avoids wasteful oscillation.

### 3.5 Convergence Speed ($T_{\text{conv}}$)
Defined as the earliest training episode $E^* \in [1, 1000]$ after which the rolling 50-episode moving average of evaluation reward remains within a 5% tolerance band of the final asymptotic policy return:
$$T_{\text{conv}}^{(i)} = \min \left\{ E \in [50, 1000] : \left| \text{MA}_{50}(E) - \bar{R}_{\text{final}} \right| \le 0.05 \cdot \left| \bar{R}_{\text{final}} \right| \quad \forall e \ge E \right\}$$

---

## 4. Formal Statistical Testing Protocol

To ensure conclusions are scientifically defensible rather than artifacts of stochastic variance:

### 4.1 Paired Two-Tailed Student's $t$-Test
When comparing two algorithms or representations $A$ and $B$, evaluations are strictly paired across matching random seeds:
$$D_i = X_A^{(i)} - X_B^{(i)}, \quad \text{for } i \in \{0, 1, 2, 3, 4\}$$
Sample mean difference:
$$\bar{D} = \frac{1}{5} \sum_{i=0}^4 D_i$$
Sample standard deviation of differences:
$$s_D = \sqrt{\frac{1}{4} \sum_{i=0}^4 \left( D_i - \bar{D} \right)^2}$$
Standard error of the mean difference:
$$\text{SE}_{\bar{D}} = \frac{s_D}{\sqrt{5}}$$
Test statistic:
$$t = \frac{\bar{D}}{\text{SE}_{\bar{D}}} = \frac{\bar{D}}{s_D / \sqrt{5}}, \quad \text{with degrees of freedom } \nu = 5 - 1 = 4$$
Significance criterion: Two-tailed $p$-value $< 0.05$ (corresponding to $|t| > 2.776$ at $\alpha = 0.05$).

### 4.2 95% Confidence Intervals (CI)
For every reported mean $\mu$, the two-sided 95% confidence interval is computed via:
$$\text{CI}_{95\%} = \mu \pm t_{0.025, 4} \times \frac{s}{\sqrt{5}} = \mu \pm 2.776 \times \frac{s}{\sqrt{5}}$$

### 4.3 Standardized Effect Size: Cohen's $d_z$
For paired samples, effect size is measured by Cohen's $d_z$:
$$d_z = \frac{\bar{D}}{s_D}$$
- Small Effect: $|d_z| \ge 0.2$
- Medium Effect: $|d_z| \ge 0.5$
- Large Effect: $|d_z| \ge 0.8$
- Very Large Effect: $|d_z| \ge 1.2$

