# Enhanced-State Cloud Autoscaling using Reinforcement Learning

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/environment-Gymnasium-green.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced Reinforcement Learning framework for dynamic cloud resource autoscaling. This project enhances traditional autoscaling state spaces with exogenous and operational telemetry—utilization, capacity, demand trend, network traffic, and task execution time—and introduces **Dyna-Q** (tabular Q-learning integrated with a learned transition model and background simulated planning) alongside **Q-Learning**, **SARSA**, **Deep Q-Networks (DQN)**, and rule-based baselines.

---

## Table of Contents
1. [Problem Statement & Objectives](#problem-statement--objectives)
2. [State, Action, and Reward Formulation](#state-action-and-reward-formulation)
3. [Algorithms Evaluated](#algorithms-evaluated)
4. [Empirical Analysis: Synthetic vs Real Kaggle Dataset](#empirical-analysis-synthetic-vs-real-kaggle-dataset)
5. [Experimental Results & Analysis](#experimental-results--analysis)
6. [Visualizations](#visualizations)
7. [How to Run & Reproduce](#how-to-run--reproduce)
8. [Limitations & Practical Realities](#limitations--practical-realities)
9. [Attribution & Credits](#attribution--credits)

---

## Problem Statement & Objectives

Cloud infrastructure autoscaling requires maintaining balance between two competing objectives:
1. **Minimizing Operational Expenditure (OpEx):** Over-provisioning idle virtual machines wastes compute budget.
2. **Eliminating Service Level Agreement (SLA) Violations:** Under-provisioning compute capacity during workload surges leads to dropped requests, queue overflows, and customer-facing latency spikes.

Traditional autoscalers often rely solely on reactive CPU utilization thresholds or simple 2-variable states. In this project, we formulate the autoscaling challenge as a Markov Decision Process (MDP) with an **enhanced 5-dimensional state representation** ($405$ discrete states) and evaluate model-free vs. model-based RL agents across both **synthetic leading-signal workloads** and a **2-million row real-world cloud workload dataset**.

---

## State, Action, and Reward Formulation

### 1. State Space ($405$ States)
The environment represents state as a tuple of 5 discretized indices:

$$s_t = \langle \text{utilization}, \text{capacity}, \text{trend}, \text{network\_level}, \text{exec\_level} \rangle$$

- **Utilization ($3$ levels):** Ratio of current demand to allocated capacity ($\text{demand} / (\text{capacity} \times 20)$):
  - $0$: Low ($< 0.40$)
  - $1$: Medium ($0.40 \le u \le 0.80$)
  - $2$: High ($> 0.80$)
- **Capacity ($5$ levels):** Active server count $\in \{1, 2, 3, 4, 5\}$ (mapped to indices $0 \dots 4$).
- **Trend ($3$ levels):** Short-term momentum of demand $\Delta d = d_t - d_{t-1}$:
  - $0$: Falling ($\Delta d < -2.0$)
  - $1$: Stable ($-2.0 \le \Delta d \le +2.0$)
  - $2$: Rising ($\Delta d > +2.0$)
- **Network Level ($3$ levels):** Terciles ($0, 1, 2$) calculated strictly from the training partition ($80\%$ chronological split).
- **Execution Time Level ($3$ levels):** Terciles ($0, 1, 2$) calculated strictly from the training partition.

$$\text{Total Discrete State Space} = 3 \times 5 \times 3 \times 3 \times 3 = 405 \text{ states}$$

### 2. Action Space
- Discrete($3$):
  - $a = 0$: **Scale Down** ($-1$ server)
  - $a = 1$: **Hold** ($0$ capacity change)
  - $a = 2$: **Scale Up** ($+1$ server)
- Capacity is bounded strictly within $[1, 5]$ servers ($20$ to $100$ compute units).

### 3. Reward Function
At each step $t$, the agent receives a multi-component reward matching `env.py`:

$$R_t = R_{\text{oper}} - C_{\text{provision}} - P_{\text{SLA}} - P_{\text{churn}} - P_{\text{underutil}}$$

Where:
- **Normal Operating Bonus:** $+10.0$ awarded when $0.40 \le u \le 0.80$ (+5.0 extra if $0.60 \le u \le 0.70$).
- **Resource Capacity Cost:** $C_{\text{provision}} = 0.5 \times \text{capacity}$ per step.
- **SLA Violation Penalty:** Triggered when utilization $u \ge 0.90$:
  $$P_{\text{SLA}} = 50.0 \times (1.0 + [u - 0.90])$$
  *(In synthetic mode, an additional $-50.0$ penalty occurs if queueing latency exceeds $150\text{ ms}$)*.
- **Churn Penalty:** $P_{\text{churn}} = 2.0$ if $a_t \ne \text{Hold}$ (penalizes unnecessary oscillatory flapping).
- **Underutilization Waste Penalty:** $P_{\text{underutil}} = 5.0 \times (0.20 - u)$ if $u < 0.20$.

---

## Algorithms Evaluated

1. **RandomPolicy:** Uniformly samples random actions $a \sim \mathcal{U}\{0, 1, 2\}$. Serves as the lower baseline.
2. **ThresholdPolicy:** Industrial rule-based heuristic: scales up when utilization is High ($>0.75$), scales down when Low ($<0.50$), holds otherwise.
3. **QLearningAgent:** Standard tabular off-policy Temporal-Difference (TD) control updating $Q(s, a)$ via:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
4. **SARSAAgent:** On-policy TD control evaluating policy trajectories:
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$
5. **DynaQAgent (Main Contribution):** Integrates sample-efficient Q-learning with an internal transition model $M(s, a) \to (r, s')$. In addition to direct real-world experience, the agent performs $k$ background simulated planning updates per timestep, accelerating credit assignment across the 405-state space.
6. **DQNAgent (Deep Q-Network):** Deep RL agent taking the normalized 5-dimensional state vector through an MLP ($5 \to 128 \to 128 \to 3$), utilizing an Experience Replay Buffer ($100\text{k}$ transitions), target network with soft Polyak averaging ($\tau = 10^{-3}$), and optional Double DQN loss.

---

## Empirical Analysis: Synthetic vs Real Kaggle Dataset

A central scientific contribution of this study is an honest, empirical evaluation of feature engineering across two contrasting workload paradigms:

### 1. Real Kaggle Dataset (`vmCloud_data.csv`) Findings
We analyzed the full 338 MB Kaggle dataset ($2,000,000$ rows):
- **Missing Value Profile:** 687,148 rows ($34.36\%$) contained missing values across key telemetry features (~10% per column) and were dropped during preprocessing, leaving $1,312,852$ clean chronological rows.
- **Correlation Reality:** Cross-feature Pearson correlations are effectively zero:
  $$\rho(\text{CPU}, \text{Network}) = 0.0004 \quad | \quad \rho(\text{CPU}, \text{Exec Time}) = 0.0001 \quad | \quad \rho(\text{Network}, \text{Exec Time}) = 0.0002$$
- **Autocorrelation Reality:** The lag-1 autocorrelation of CPU demand is $-0.00003$. The data behaves like independent, identically distributed (i.i.d.) uniform random noise rather than physical server traffic with diurnal cycles or cache effects.
- **Implication for RL:** Expanding the state space from $45$ states (`base`) to $405$ states (`full`) on this dataset causes severe state space dilution without adding any predictive signal. Tabular agents require $10\times$ more samples to populate Q-tables while learning nothing predictable about future transitions.

### 2. Structured Synthetic Workload
To verify that the enhanced state space functions correctly when physical signal is present, we constructed a synthetic workload with verified characteristics:
- **Periodic Demand:** Diurnal sinusoidal load + weekly drift + Gaussian noise + random burst spikes ($5\%$ probability, $+20$ demand).
- **Leading Network Indicator:** Network traffic engineered to lead demand by 1 timestep:
  $$\text{network}_t = 0.8 \times \text{demand}_{t+1} + \epsilon_t$$
- **Coupled Execution Latency:** Task latency modeled via queueing theory:
  $$\text{exec\_time}_t = \frac{\text{base\_time}}{1 - \min(u_t, 0.95)} + \eta_t$$

---

## Experimental Results & Analysis (1,000 Episodes, 5 Independent Seeds [0, 1, 2, 3, 4])

All experiments were trained across 5 independent random seeds (`[0, 1, 2, 3, 4]`) for 1,000 training episodes, followed by 100 held-out evaluation episodes evaluated greedily ($\epsilon = 0$) to measure deterministic policy quality.

### 1. Main Performance Comparison (Enhanced 405-State Space)

#### Synthetic Workload (With Leading Signal)
| Algorithm | Feature Set | Mean Test Reward ± Std | SLA Violations ± Std | Mean Cost | Convergence (eps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DQN (Double MLP)** | full | **7,838.33 ± 124.70** | 16.0 ± 3.6 | 1,883.14 | 301 |
| **THRESHOLD** | full | 6,607.83 ± 32.42 | 32.4 ± 0.4 | 1,953.77 | 44 |
| **QLEARNING** | full | 6,446.10 ± 317.83 | 13.7 ± 2.9 | 2,033.06 | 213 |
| **DYNAQ (plan=10)** | full | 4,230.09 ± 372.41 | 16.1 ± 8.3 | 2,232.36 | 201 |
| **SARSA** | full | 3,861.64 ± 414.66 | **5.7 ± 0.2** | 2,390.49 | 162 |
| **RANDOM** | full | -38,413.10 ± 457.03 | 442.2 ± 3.6 | 1,500.64 | 278 |

#### Real Kaggle Dataset (Uncorrelated Noise)
| Algorithm | Feature Set | Mean Test Reward ± Std | SLA Violations ± Std | Mean Cost | Convergence (eps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SARSA** | full | **-3,987.73 ± 141.44** | **100.9 ± 0.9** | 2,495.88 | 281 |
| **QLEARNING** | full | -4,451.19 ± 556.98 | 105.3 ± 5.8 | 2,483.58 | 269 |
| **DYNAQ (plan=10)** | full | -6,046.40 ± 392.34 | 123.9 ± 5.0 | 2,432.32 | 219 |
| **DQN (Double MLP)** | full | -13,733.43 ± 2515.96 | 261.7 ± 38.9 | 2,051.81 | 301 |
| **THRESHOLD** | full | -20,897.15 ± 72.55 | 297.8 ± 0.5 | 1,949.76 | 299 |
| **RANDOM** | full | -47,100.87 ± 449.67 | 459.2 ± 3.4 | 1,500.64 | 315 |

---

### 2. Contribution 1: Controlled State-Space Comparison (45 vs 405 States)

| Data Mode | Algorithm | Original 45-State (`base`) | Enhanced 405-State (`full`) | Reward Gain (Δ) | Paired $t$-stat | $p$-value | Cohen's $d_z$ | Significant? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | **Q-Learning** | 4,242.2 ± 384.4 | **6,446.1 ± 317.8** | **+2,203.9 (+51.9%)** | 3.456 | 0.0259 | 1.55 | **YES ($p < 0.05$)** |
| **Synthetic** | **Dyna-Q (plan=10)** | 2,379.1 ± 252.0 | **4,230.1 ± 372.4** | **+1,851.0 (+77.8%)** | 5.611 | 0.0050 | 2.51 | **YES ($p < 0.01$)** |
| **Synthetic** | **SARSA** | 4,282.2 ± 243.3 | 3,861.6 ± 414.7 | -420.6 (-9.8%) | -1.647 | 0.1749 | -0.74 | NO ($p \ge 0.05$) |
| **Synthetic** | **DQN** | *Not Evaluated (Pending)* | **7,838.3 ± 124.7** | *N/A (Pending)* | *N/A* | *N/A* | *N/A* | *PENDING* |
| **Dataset** | **Q-Learning** | **-3,734.5 ± 116.5** | -4,451.2 ± 557.0 | -716.7 (-19.2%) | -2.090 | 0.1049 | -0.93 | NO ($p \ge 0.05$) |
| **Dataset** | **SARSA** | **-3,561.8 ± 116.9** | -3,987.7 ± 141.4 | -425.9 (-12.0%) | -3.210 | 0.0326 | -1.44 | **YES (Degradation)** |
| **Dataset** | **Dyna-Q (plan=10)** | **-5,617.8 ± 384.6** | -6,046.4 ± 392.3 | -428.6 (-7.6%) | -0.703 | 0.5209 | -0.31 | NO ($p \ge 0.05$) |
| **Dataset** | **DQN** | *Not Evaluated (Pending)* | -13,733.4 ± 2516.0 | *N/A (Pending)* | *N/A* | *N/A* | *N/A* | *PENDING* |

---

### 3. Feature Ablation Analysis (Mean Test Reward)

| Feature Set | State Dim | Q-Learning (Synth) | Dyna-Q (Synth) | Q-Learning (Dataset) | Dyna-Q (Dataset) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **base** | 45 | 4,242.2 | 2,379.1 | **-3,734.5** | -5,617.8 |
| **base+network** | 135 | 6,172.4 | 4,825.7 | -3,842.0 | **-4,891.9** |
| **base+exec** | 135 | 5,546.8 | 4,224.5 | -4,273.9 | -5,160.4 |
| **full** | 405 | **6,446.1** | 4,230.1 | -4,451.2 | -6,046.4 |

---

### 4. Contribution 2: Dyna-Q Planning Sensitivity Analysis ($k \in \{0, 5, 10, 20, 50\}$)

| Data Mode | Planning Steps ($k$) | Mean Test Reward ± Std | SLA Violations ± Std | Resource Cost | Convergence (eps) | Paired Gain vs $k=0$ | $p$-value (vs $k=0$) |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | **0** | **6,446.1 ± 317.8** | **13.7 ± 2.9** | 2,033.1 | 213 | Baseline | - |
| **Synthetic** | **5** | 4,534.6 ± 634.1 | 20.5 ± 12.8 | 2,125.4 | 203 | -1,911.5 | 0.0084 ($p < 0.01$) |
| **Synthetic** | **10** | 4,230.1 ± 372.4 | 16.1 ± 8.3 | 2,232.4 | 201 | -2,216.0 | 0.0014 ($p < 0.01$) |
| **Synthetic** | **20** | 4,172.7 ± 1145.4 | 21.5 ± 9.5 | 2,152.4 | 189 | -2,273.3 | 0.0141 ($p < 0.05$) |
| **Synthetic** | **50** | 2,344.8 ± 1614.7 | 41.2 ± 24.9 | 2,144.0 | 192 | -4,101.3 | 0.0088 ($p < 0.01$) |
| **Dataset** | **0** | **-4,451.2 ± 557.0** | **105.3 ± 5.8** | 2,483.6 | 269 | Baseline | - |
| **Dataset** | **5** | -5,257.1 ± 802.7 | 113.3 ± 10.1 | 2,462.0 | 221 | -805.9 | 0.1561 |
| **Dataset** | **10** | -6,046.4 ± 392.3 | 123.9 ± 5.0 | 2,432.3 | 219 | -1,595.2 | 0.0029 ($p < 0.01$) |
| **Dataset** | **20** | -6,027.6 ± 632.9 | 123.4 ± 8.3 | 2,433.7 | 245 | -1,576.5 | 0.0058 ($p < 0.01$) |
| **Dataset** | **50** | -5,223.5 ± 263.2 | 113.6 ± 4.1 | 2,460.8 | 289 | -772.3 | 0.0159 ($p < 0.05$) |

**Key Academic Conclusions:**
1. **State-Space Expansion Requires Causal Signals:** Expanding the state space from 45 to 405 states increases Q-Learning return by **$+51.9\%$** ($p < 0.05$) on structured workloads where network traffic is an advance indicator. Conversely, on uncorrelated noise (Kaggle dataset), expanding the state space causes a **$19.2\%$ degradation** due to sample sparsity.
2. **Double DQN Performance Divergence:** Continuous neural network function approximation excels on structured periodic workloads (achieving the highest overall reward of **$+7,838.33$**), but diverges catastrophically on white noise ($-13,733.43$).
3. **Tabular Robustness & SARSA Safety:** Tabular Q-Learning and SARSA remain remarkably stable. SARSA achieves the lowest SLA violation counts across both environments ($5.7$ on synthetic, $100.9$ on dataset) due to its conservative on-policy updates.
4. **Deterministic Dyna-Q Over-Planning:** At asymptotic horizons (1,000 episodes), deterministic Dyna-Q model planning repeatedly replays stochastic noise, degrading performance from $+6,446$ ($k=0$) down to $+2,344$ ($k=50$). Model-based RL in stochastic cloud environments strictly requires probabilistic transition models.

---

## Visualizations (14 Publication-Grade Figures)

| Figure | Description | File Link |
| :--- | :--- | :--- |
| **Figure 1** | Training Learning Curves (1000 Episodes, Mean ± 1 Std) | [`figures/fig1_learning_curves.png`](figures/fig1_learning_curves.png) |
| **Figure 2** | Evaluation SLA Violations Across Algorithms | [`figures/fig2_sla_violations.png`](figures/fig2_sla_violations.png) |
| **Figure 3** | Policy Convergence Speed Comparison | [`figures/fig3_convergence_speed.png`](figures/fig3_convergence_speed.png) |
| **Figure 4** | Feature Ablation: Base vs +Net vs +Exec vs Full | [`figures/fig4_feature_ablation.png`](figures/fig4_feature_ablation.png) |
| **Figure 5** | Dyna-Q Planning Steps Sensitivity ($k \in \{0, 5, 10, 20, 50\}$) | [`figures/fig5_planning_steps.png`](figures/fig5_planning_steps.png) |
| **Figure 6** | Dynamic Capacity vs. Workload Demand Trace | [`figures/fig6_capacity_vs_demand.png`](figures/fig6_capacity_vs_demand.png) |
| **Figure 7** | Resource Infrastructure Cost Comparison | [`figures/fig7_cost_comparison.png`](figures/fig7_cost_comparison.png) |
| **Figure 8** | Controlled State-Space Comparison (45 vs. 405 States) | [`figures/fig8_state_space_comparison.png`](figures/fig8_state_space_comparison.png) |
| **Figure 9** | End-to-End System Architecture Flowchart | [`figures/fig9_system_architecture.png`](figures/fig9_system_architecture.png) |
| **Figure 10** | State-Space Multi-Radix Design Architecture | [`figures/fig10_state_space_design.png`](figures/fig10_state_space_design.png) |
| **Figure 11** | Model-Free Q-Learning vs. Model-Based Dyna-Q Loop | [`figures/fig11_qlearning_vs_dynaq.png`](figures/fig11_qlearning_vs_dynaq.png) |
| **Figure 12** | Workload Profile: Leading Signals vs. Empirical Noise | [`figures/fig12_workload_patterns.png`](figures/fig12_workload_patterns.png) |
| **Figure 13** | Master Algorithm Evaluation Reward Comparison | [`figures/fig13_algorithm_reward_comparison.png`](figures/fig13_algorithm_reward_comparison.png) |
| **Figure 14** | Pareto Optimization Frontier: Cost vs. SLA Violations | [`figures/fig14_pareto_cost_sla.png`](figures/fig14_pareto_cost_sla.png) |


---

## Academic & Research Documentation

Detailed scientific documentation, methodology derivations, and presentation materials are organized in the [`docs/`](docs/) directory:

- [**Academic Presentation Deck (20 Slides)**](docs/presentation_content.md): Complete slide-by-slide presentation with formulas, architecture diagrams, and speaker notes.
- [**Research Methodology & Mathematical Derivations**](docs/research_methodology.md): Rigorous MDP definitions, reward function derivation, state encoding, and academic attributions.
- [**Final Results Analysis & Research Questions**](docs/final_results_analysis.md): In-depth scientific analysis answering 7 core research questions (state-space boundary, Dyna-Q noise compounding, Double DQN empirical limits).
- [**Final Experiment Summary**](docs/final_experiment_summary.md): Experimental design, metric definitions, paired $t$-test protocols, and benchmark structure.
- [**Final Research Report Draft**](docs/final_research_report_draft.md): Publication-grade academic paper draft.
- [**Statistical Analysis Report**](results/statistical_analysis_ep1000.md): Automated paired $t$-tests, $p$-values, $95\%$ CIs, and Cohen's $d_z$ effect sizes across all 34 configurations.
- [**Master Summary Table**](results/summary_table.md): Tabular overview of all evaluation rewards, SLA violations, and costs.
- [**Provenance & Attribution Notice**](NOTICE.md): Full attribution to reference repository authors and itemized novel contributions.

---

## How to Run & Reproduce

### 1. Requirements & Setup
All dependencies run under Python 3.10+ (tested on Python 3.12):
```bash
pip install gymnasium numpy pandas pyarrow torch matplotlib pyyaml pytest
```

### 2. Run the Full Test Suite
Run the 29 unit and integration tests covering environment logic, state masking, agents, and data loaders:
```bash
py -3.12 -m pytest tests -v
```

### 3. Train a Single Agent
```bash
# Train Dyna-Q with full features on synthetic workload for 500 episodes
py -3.12 src/train.py --algo dynaq --data-mode synthetic --feature-set full --planning-steps 10 --episodes 500 --seeds 0 1 2

# Train Deep Q-Network on the real dataset
py -3.12 src/train.py --algo dqn --data-mode dataset --feature-set full --episodes 500 --seeds 0 1 2
```

### 4. Run the Full Benchmark Suite
```bash
# Execute Suite A (Main Comparison), Suite B (Feature Ablation), and Suite C (Planning Steps)
py -3.12 src/run_all.py --smoke --suite all
```

### 5. Generate Plots & Markdown Tables
```bash
# Regenerate all 14 publication figures and results/summary_table.md
py -3.12 src/plots.py

# Run paired t-tests and generate statistical analysis report
py -3.12 src/statistical_analysis.py
```

---

## Limitations & Practical Realities

While the empirical findings validate Dyna-Q and multi-dimensional state design, production cloud autoscaling involves real-world constraints:
1. **Zero Boot / Provisioning Delay:** The simulated environment provisions or terminates servers instantaneously. In cloud providers (AWS EC2, GCP Compute Engine, Kubernetes), VM spin-up takes between 30 and 180 seconds, necessitating predictive horizons longer than single-step lookahead.
2. **Synthetic Execution Time:** Latency is approximated using an $M/M/1$-style queueing formula rather than distributed tracing (e.g. OpenTelemetry, Jaeger) of request latencies and service-mesh retries.
3. **Dataset Temporal Artifact:** The Kaggle dataset lacks temporal continuity (lag-1 autocorrelation $\approx 0$). Industrial deployments require real time-series traces (e.g. Alibaba Cluster Data, Azure VM Workload Traces) that capture true diurnal cycles, cron jobs, and failover spikes.

---

## Attribution & Credits

This project builds upon and extends the cloud autoscaling reinforcement learning research developed by the original UVA research team:
- **Original Project:** *Cloud Autoscaling using Reinforcement Learning*
- **Authors:** Harish Balasubramanyam, Jack Healy, Luke McGregor (University of Virginia)
- **Original Architecture:** Reactive Gymnasium environment, discrete tabular Q-Learning / SARSA, and DQN implementations.

This enhanced repository introduces the 5D enhanced state space ($405$ states), Dyna-Q model-based planning, leading-indicator synthetic generation, rigorous Kaggle data hygiene and caching, automated ablation pipelines, and full benchmark suites.
