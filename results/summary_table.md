# Final Benchmark Results Summary

### Table 1: Main Algorithm Benchmark (Synthetic Workload (Leading Signal), 405-State Space)

| Algorithm | Feature Set | Mean Test Reward ± Std | SLA Violations ± Std | Mean Cost | Convergence (eps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | full | -38,413.10 ± 457.03 | 442.2 ± 3.6 | 1,500.64 | 278 |
| **THRESHOLD** | full | 6,607.83 ± 32.42 | 32.4 ± 0.4 | 1,953.77 | 44 |
| **QLEARNING** | full | 6,446.10 ± 317.83 | 13.7 ± 2.9 | 2,033.06 | 213 |
| **SARSA** | full | 3,861.64 ± 414.66 | 5.7 ± 0.2 | 2,390.49 | 162 |
| **DYNAQ (plan=10)** | full | 4,230.09 ± 372.41 | 16.1 ± 8.3 | 2,232.36 | 201 |
| **DQN** | full | 7,838.33 ± 124.70 | 16.0 ± 3.6 | 1,883.14 | 301 |


### Table 1: Main Algorithm Benchmark (Real Kaggle Dataset (Uncorrelated Noise), 405-State Space)

| Algorithm | Feature Set | Mean Test Reward ± Std | SLA Violations ± Std | Mean Cost | Convergence (eps) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RANDOM** | full | -47,100.87 ± 449.67 | 459.2 ± 3.4 | 1,500.64 | 315 |
| **THRESHOLD** | full | -20,897.15 ± 72.55 | 297.8 ± 0.5 | 1,949.76 | 299 |
| **QLEARNING** | full | -4,451.19 ± 556.98 | 105.3 ± 5.8 | 2,483.58 | 269 |
| **SARSA** | full | -3,987.73 ± 141.44 | 100.9 ± 0.9 | 2,495.88 | 281 |
| **DYNAQ (plan=10)** | full | -6,046.40 ± 392.34 | 123.9 ± 5.0 | 2,432.32 | 219 |
| **DQN** | full | -13,733.43 ± 2515.96 | 261.7 ± 38.9 | 2,051.81 | 301 |


### Table 2: Controlled State-Space Comparison (45 States vs 405 States)

| Data Mode | Algorithm | Original 45-State (`base`) | Enhanced 405-State (`full`) | Reward Gain (Δ) | SLA Violations (45 vs 405) | Cost (45 vs 405) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Synthetic** | **QLEARNING** | 4,242.2 | 6,446.1 | **+2,203.9** | 25.6 → 13.7 | 2,163.6 → 2,033.1 |
| **Synthetic** | **SARSA** | 4,282.2 | 3,861.6 | **-420.6** | 8.0 → 5.7 | 2,336.2 → 2,390.5 |
| **Synthetic** | **DYNAQ (plan=10)** | 2,379.1 | 4,230.1 | **+1,851.0** | 22.9 → 16.1 | 2,336.6 → 2,232.4 |
| **Synthetic** | **DQN** | 5,261.8 | 7,838.3 | **+2,576.5** | 54.6 → 16.0 | 1,823.9 → 1,883.1 |
| **Dataset** | **QLEARNING** | -3,734.5 | -4,451.2 | **-716.7** | 100.5 → 105.3 | 2,496.9 → 2,483.6 |
| **Dataset** | **SARSA** | -3,561.8 | -3,987.7 | **-425.9** | 100.5 → 100.9 | 2,497.0 → 2,495.9 |
| **Dataset** | **DYNAQ (plan=10)** | -5,617.8 | -6,046.4 | **-428.6** | 117.5 → 123.9 | 2,450.5 → 2,432.3 |
| **Dataset** | **DQN** | -10,348.9 | -13,733.4 | **-3,384.5** | 209.2 → 261.7 | 2,197.3 → 2,051.8 |


### Table 3: Dyna-Q Planning Steps Sensitivity (Enhanced 405-State Space)

| Data Mode | Planning Steps (k) | Mean Test Reward ± Std | SLA Violations ± Std | Resource Cost | Convergence (eps) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Synthetic** | **0** | 6,446.1 ± 317.8 | 13.7 ± 2.9 | 2,033.1 | 213 |
| **Synthetic** | **5** | 4,534.6 ± 634.1 | 20.5 ± 12.8 | 2,125.4 | 203 |
| **Synthetic** | **10** | 4,230.1 ± 372.4 | 16.1 ± 8.3 | 2,232.4 | 201 |
| **Synthetic** | **20** | 4,172.7 ± 1145.4 | 21.5 ± 9.5 | 2,152.4 | 189 |
| **Synthetic** | **50** | 2,344.8 ± 1614.7 | 41.2 ± 24.9 | 2,144.0 | 192 |
| **Dataset** | **0** | -4,451.2 ± 557.0 | 105.3 ± 5.8 | 2,483.6 | 269 |
| **Dataset** | **5** | -5,257.1 ± 802.7 | 113.3 ± 10.1 | 2,462.0 | 221 |
| **Dataset** | **10** | -6,046.4 ± 392.3 | 123.9 ± 5.0 | 2,432.3 | 219 |
| **Dataset** | **20** | -6,027.6 ± 632.9 | 123.4 ± 8.3 | 2,433.7 | 245 |
| **Dataset** | **50** | -5,223.5 ± 263.2 | 113.6 ± 4.1 | 2,460.8 | 289 |


### Table 4: Detailed Feature Ablation (Mean Test Reward)

| Feature Set | State Dim | Q-Learning (Synth) | Dyna-Q (Synth) | Q-Learning (Dataset) | Dyna-Q (Dataset) |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **base** | 45 | 4,242.2 | 2,379.1 | -3,734.5 | -5,617.8 |
| **base+network** | 135 | 6,172.4 | 4,825.7 | -3,842.0 | -4,891.9 |
| **base+exec** | 135 | 5,546.8 | 4,224.5 | -4,273.9 | -5,160.4 |
| **full** | 405 | 6,446.1 | 4,230.1 | -4,451.2 | -6,046.4 |