# Notice and Attribution

## Reference Project
This project incorporates conceptual foundations and adapted components from:
- **Project**: *Cloud-Autoscaling-using-RL*
- **Repository**: [https://github.com/rah-ds/Cloud-Autoscaling-using-RL](https://github.com/rah-ds/Cloud-Autoscaling-using-RL)
- **Authors**: Harish Balasubramanyam, Jack Healy, Luke McGregor (University of Virginia, 2025)
- **License**: MIT License (Copyright (c) 2025 rah-ds)

---

## Adapted Components
The following elements were adapted from the reference repository:
1. **Discrete Autoscaling Action Space**: The discrete 3-action MDP structure ($\{-1, 0, +1\}$ for scale-down, hold, and scale-up).
2. **Threshold Heuristic Scaler**: The rule-based utilization threshold logic implemented in `src/agents/baselines.py`.
3. **Telemetry Feature Schema**: Workload utilization and capacity modeling based on the Kaggle cloud performance dataset (`vmCloud_data.csv`).
4. **Tabular TD Update Foundations**: Base Q-Learning and SARSA state-action table indexing principles.

All other components—including the 5-feature enhanced state space (405 states), model-based Dyna-Q agent, continuous Double DQN, synthetic leading-signal generator, and statistical evaluation framework—were independently developed. Full technical details are documented in `README.md` and `docs/`.
