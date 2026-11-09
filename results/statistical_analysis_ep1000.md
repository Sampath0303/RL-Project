# Rigorous Statistical Analysis Report (1000 Episodes, 5 Seeds [0, 1, 2, 3, 4])

## 1. Contribution 1: State-Space Representation Comparison

Comparing Original 3-Feature (45 states) vs Enhanced 5-Feature (405 states) across matched seeds.

| Data Mode | Algorithm | Original 45-State (Mean ± 95% CI) | Enhanced 405-State (Mean ± 95% CI) | Mean Diff | Paired t-stat | p-value | Cohen's d_z | Significant (p<0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | **QLEARNING** | 4,242.2 ± 1872.6 | 6,446.1 ± 441.2 | +2,203.9 | 3.456 | 0.0259 | 1.55 | **YES (Significant)** |
| **Synthetic** | **SARSA** | 4,282.2 ± 438.4 | 3,861.6 ± 575.6 | -420.6 | -1.647 | 0.1749 | -0.74 | NO |
| **Synthetic** | **DYNAQ** | 2,379.1 ± 1206.1 | 4,230.1 ± 517.0 | +1,851.0 | 5.611 | 0.0050 | 2.51 | **YES (Significant)** |
| **Synthetic** | **DQN** | 5,261.8 ± 250.9 | 7,838.3 ± 173.1 | +2,576.5 | 40.778 | 0.0000 | 18.24 | **YES (Significant)** |
| **Dataset** | **QLEARNING** | -3,734.5 ± 305.3 | -4,451.2 ± 773.2 | -716.7 | -2.090 | 0.1049 | -0.93 | NO |
| **Dataset** | **SARSA** | -3,561.8 ± 288.2 | -3,987.7 ± 196.3 | -425.9 | -3.210 | 0.0326 | -1.44 | **YES (Significant)** |
| **Dataset** | **DYNAQ** | -5,617.8 ± 1893.0 | -6,046.4 ± 544.7 | -428.6 | -0.703 | 0.5209 | -0.31 | NO |
| **Dataset** | **DQN** | -10,348.9 ± 7862.4 | -13,733.4 ± 3492.7 | -3,384.5 | -0.958 | 0.3921 | -0.43 | NO |


## 2. Contribution 2: Dyna-Q (Model Planning) vs Baselines (Enhanced 405-State Space)

Paired comparisons between Dyna-Q (plan=10) and alternative policies across matching seeds.

| Data Mode | Comparison Pair | Baseline (Mean ± 95% CI) | Dyna-Q (Mean ± 95% CI) | Mean Gain (Δ) | Paired t-stat | p-value | Cohen's d_z | Significant (p<0.05)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | Dyna-Q vs **QLEARNING** | 6,446.1 ± 441.2 | 4,230.1 ± 517.0 | -2,216.0 | -7.908 | 0.0014 | -3.54 | **YES (Significant)** |
| **Synthetic** | Dyna-Q vs **SARSA** | 3,861.6 ± 575.6 | 4,230.1 ± 517.0 | +368.5 | 1.459 | 0.2184 | 0.65 | NO |
| **Synthetic** | Dyna-Q vs **DQN** | 7,838.3 ± 173.1 | 4,230.1 ± 517.0 | -3,608.2 | -16.245 | 0.0001 | -7.27 | **YES (Significant)** |
| **Synthetic** | Dyna-Q vs **THRESHOLD** | 6,607.8 ± 45.0 | 4,230.1 ± 517.0 | -2,377.7 | -12.753 | 0.0002 | -5.70 | **YES (Significant)** |
| **Dataset** | Dyna-Q vs **QLEARNING** | -4,451.2 ± 773.2 | -6,046.4 ± 544.7 | -1,595.2 | -6.499 | 0.0029 | -2.91 | **YES (Significant)** |
| **Dataset** | Dyna-Q vs **SARSA** | -3,987.7 ± 196.3 | -6,046.4 ± 544.7 | -2,058.7 | -8.061 | 0.0013 | -3.61 | **YES (Significant)** |
| **Dataset** | Dyna-Q vs **DQN** | -13,733.4 ± 3492.7 | -6,046.4 ± 544.7 | +7,687.0 | 6.578 | 0.0028 | 2.94 | **YES (Significant)** |
| **Dataset** | Dyna-Q vs **THRESHOLD** | -20,897.1 ± 100.7 | -6,046.4 ± 544.7 | +14,850.7 | 88.841 | 0.0000 | 39.73 | **YES (Significant)** |


## 3. Dyna-Q Planning Step Sensitivity (k = 0, 5, 10, 20, 50)

| Data Mode | Planning Steps (k) | Mean Test Reward ± 95% CI | Test SLA Violations ± Std | Resource Cost | Convergence (eps) | Paired Gain vs k=0 | p-value (vs k=0) |
| :--- | :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | **0** | 6,446.1 ± 441.2 | 13.7 ± 3.2 | 2,033.1 | 213 | Baseline | - |
| **Synthetic** | **5** | 4,534.6 ± 880.3 | 20.5 ± 14.4 | 2,125.4 | 203 | -1,911.5 | 0.0084 |
| **Synthetic** | **10** | 4,230.1 ± 517.0 | 16.1 ± 9.3 | 2,232.4 | 201 | -2,216.0 | 0.0014 |
| **Synthetic** | **20** | 4,172.7 ± 1590.1 | 21.5 ± 10.6 | 2,152.4 | 189 | -2,273.3 | 0.0141 |
| **Synthetic** | **50** | 2,344.8 ± 2241.5 | 41.2 ± 27.8 | 2,144.0 | 192 | -4,101.3 | 0.0088 |
| **Dataset** | **0** | -4,451.2 ± 773.2 | 105.3 ± 6.5 | 2,483.6 | 269 | Baseline | - |
| **Dataset** | **5** | -5,257.1 ± 1114.3 | 113.3 ± 11.3 | 2,462.0 | 221 | -805.9 | 0.1561 |
| **Dataset** | **10** | -6,046.4 ± 544.7 | 123.9 ± 5.6 | 2,432.3 | 219 | -1,595.2 | 0.0029 |
| **Dataset** | **20** | -6,027.6 ± 878.6 | 123.4 ± 9.3 | 2,433.7 | 245 | -1,576.5 | 0.0058 |
| **Dataset** | **50** | -5,223.5 ± 365.4 | 113.6 ± 4.6 | 2,460.8 | 289 | -772.3 | 0.0159 |

