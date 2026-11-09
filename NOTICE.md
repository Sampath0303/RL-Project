# Attribution and Project Provenance Notice

## Reference Repository
This project builds upon and extends the foundational concepts and workflow introduced in:
- **Reference Project**: *Cloud-Autoscaling-using-RL*
- **Reference Repository**: [https://github.com/rah-ds/Cloud-Autoscaling-using-RL](https://github.com/rah-ds/Cloud-Autoscaling-using-RL)
- **Original Author**: rah-ds (2025)
- **License**: MIT License

---

## Reused and Adapted Components
The following elements from the reference repository served as conceptual foundations and were adapted for this project:
1. **Discrete Autoscaling Action Space**: The discrete 3-action MDP structure ($\{-1, 0, +1\}$ corresponding to scale-down, maintain, and scale-up) adapted from the original environment concept.
2. **Baseline Cloud Heuristic**: The threshold-based reactive rule-scaling policy (triggering scaling when resource utilization crosses upper/lower bounds).
3. **Telemetry Data Source**: Utilization of the Kaggle Cloud Infrastructure Performance dataset schema (`vmCloud_data.csv`) for real-world workload evaluations.
4. **Foundational Tabular Formulation**: Baseline Q-Learning and SARSA state-action table indexing principles.

---

## Novel Architectural, Algorithmic, and Empirical Contributions
This standalone repository represents an independent implementation featuring the following substantial architectural and algorithmic advancements:

1. **Enhanced 5-Feature State Representation**:
   - Expanded the state space from 3 features (45 states) to 5 features (405 states).
   - Integrated **Network Ingress Traffic** (tercile-binned: Low, Normal, High) as a leading demand predictor.
   - Integrated **Execution Time / Latency** (tercile-binned: Fast, Normal, Slow) as an endogenous system feedback metric.

2. **Model-Based Planning (Dyna-Q)**:
   - Designed and implemented `DynaQAgent` combining real-world TD updates with background planning over a learned transition/reward model.
   - Evaluated planning sensitivity across $k \in \{0, 5, 10, 20, 50\}$ planning steps.

3. **Deep Reinforcement Learning (Double DQN)**:
   - Implemented a continuous-state Double Deep Q-Network with a 128x128 MLP architecture, experience replay buffer ($100,000$ transitions), and Polyak soft target updates ($\tau = 0.001$).

4. **Balanced Cloud Autoscaling Reward Function**:
   - Formulated a balanced multi-objective reward function addressing SLA violation penalties (utilization $\ge 90\%$), underutilization penalties (utilization $< 20\%$), server provisioning costs, normal operation bonuses, and capacity churn friction.

5. **Exhaustive 34-Configuration Benchmark**:
   - Conducted multi-seed ($5$ seeds: `[0, 1, 2, 3, 4]`), 1,000-episode training benchmarks ($1,000,000$ transitions per seed) and 100-episode held-out evaluations across Suite A (Main Algorithms), Suite B (Feature Ablation), and Suite C (Planning Sensitivity).

6. **Rigorous Statistical Analysis & Visualization Framework**:
   - Automated paired $t$-tests, $p$-values, $95\%$ confidence intervals, and Cohen's $d_z$ effect sizes (`statistical_analysis.py`).
   - Generated 14 presentation-grade publication figures (300 DPI) and comprehensive academic presentation documentation.
