# Technical Research Methodology: Enhanced-State Reinforcement Learning for Dynamic Cloud Autoscaling

---

## 1. Executive Summary & Research Framework

This document outlines the theoretical, mathematical, architectural, and experimental methodology for our research project: **"Enhanced-State Reinforcement Learning for Dynamic Cloud Resource Auto-Scaling Using Q-Learning, SARSA, DQN, and Dyna-Q."**

Modern cloud data centers represent multi-billion-dollar infrastructures supporting mission-critical enterprise workloads. The fundamental operational challenge is **elastic dynamic resource provisioning**: how to dynamically adjust compute capacity to meet highly fluctuating, non-stationary user demand while simultaneously minimizing monetary resource expenditures and preventing Service Level Agreement (SLA) violations.

To investigate this problem, we develop a rigorous, first-principles reinforcement learning framework that explores two central contributions:
1. **Contribution 1 (State-Space Augmentation)**: Evaluating whether expanding the agent's observation space from a traditional 3-feature representation (45 discrete states) to an enhanced 5-feature representation incorporating leading network ingress and queueing execution latency (405 discrete states) improves autoscaling policy quality.
2. **Contribution 2 (Model-Based Planning via Dyna-Q)**: Investigating whether model-based background planning ($k \in \{0, 5, 10, 20, 50\}$ simulated updates per real interaction step) provides superior sample efficiency, stability, and asymptotic performance compared to model-free tabular methods (Q-Learning, SARSA), deep function approximation (Double DQN), and industrial heuristics (Threshold rules).

---

## 2. Problem Definition & Domain Background

### 2.1 What is Cloud Autoscaling?
Cloud autoscaling is the automated process of dynamically adjusting the quantity of computational resources (virtual machines, containers, or worker threads) allocated to an application or service in response to active incoming demand.

In cloud virtualization platforms (e.g., Amazon Web Services EC2, Google Cloud Compute Engine, Microsoft Azure Virtual Machines, Kubernetes clusters), compute instances are billed on an hourly or per-second basis. An autoscaler continuously monitors system telemetry (CPU usage, incoming request rate, memory consumption, latency) and issues scaling decisions to:
- **Scale Out / Scale Up**: Provision additional virtual instances when workload demand surges.
- **Scale In / Scale Down**: Deprovision surplus instances when workload demand subsides.

### 2.2 Why Dynamic Resource Allocation is Important
Workload demand on web applications, microservices, and databases fluctuates drastically across multiple temporal scales:
- **Diurnal Cycles**: High traffic during daytime business hours, low traffic during late-night hours.
- **Weekly Seasonality**: Distinct weekday vs. weekend patterns.
- **Unpredictable Spikes (Flash Crowds)**: Sudden, massive traffic surges caused by viral social media events, promotional flash sales, or breaking news.
- **Micro-bursts**: High-frequency stochastic fluctuations occurring on sub-second or multi-second scales.

A static allocation scheme cannot adapt to these dynamics.

### 2.3 Disadvantages of Static Provisioning
- **Over-Provisioning (Financial Waste)**: Sizing a cluster permanently for worst-case peak traffic guarantees high availability, but during average and off-peak hours, average CPU utilization hovers between 15% and 30%. Organizations end up paying for idle silicon, wasting vast monetary and electrical energy budgets.
- **Under-Provisioning (Service Degradation & SLA Penalties)**: Sizing a cluster for median load results in immediate request queue saturation during traffic surges. Response times degrade exponentially, requests time out, drop rates spike, and service providers incur severe financial and legal SLA penalties.

### 2.4 Why Reinforcement Learning?
Industry standard autoscaling solutions rely on **reactive threshold rules** (e.g., *"Add 1 VM if average CPU exceeds 80% for 5 minutes"*). These heuristics suffer from fundamental limitations:
1. **Inherent Actuation Latency**: Threshold mechanisms only trigger *after* resource saturation has already occurred, guaranteeing that the cluster suffers SLA breaches during the reactive lag.
2. **Inability to Balance Multi-Objective Trade-Offs**: Threshold rules do not mathematically optimize a global objective function. They cannot weigh the financial cost of an extra server against the probabilistic risk of an SLA breach.
3. **Threshold Thrashing & Hunting**: In the absence of calibrated hysteresis, systems oscillate between scaling up and scaling down, inducing unnecessary VM provisioning churn.
4. **Decoupled Forecasting**: While supervised deep learning models (e.g., LSTM, Transformer) can forecast future demand, forecasting alone does not prescribe optimal control actions under non-linear operational penalties.

**Reinforcement Learning (RL)** resolves these limitations by framing autoscaling as a closed-loop sequential decision-making problem. An RL agent directly learns a mapping from multidimensional telemetry states to optimal scaling actions, maximizing a long-term discounted cumulative reward that explicitly balances quality of service, infrastructure cost, and system stability.

---

## 3. Reinforcement Learning Mathematical Formulation (MDP)

The cloud autoscaling task is formally defined as an episodic, discrete-time Markov Decision Process (MDP) characterized by the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

### 3.1 Mathematical Definitions
- **Agent**: The autonomous scaling controller implementing policy $\pi(a \mid s)$.
- **Environment**: The cloud cluster simulation (`EnhancedAutoscalingEnv`), encapsulating VM instances, queue dynamics, workload stream, and telemetry generation.
- **Time Index $t$**: Discrete operational intervals ($t = 0, 1, 2, \dots, T$), where each step represents a fixed observation-and-action window.
- **Episode Horizon $T$**: Standardized episode length of $T = 200$ time steps.
- **Discount Factor $\gamma$**: Set to $\gamma = 0.99$. This near-unity discount factor forces the agent to optimize for long-term operational health and avoid myopic scaling decisions.
- **Cumulative Discounted Return $G_t$**:
  $$G_t = \sum_{k=0}^{T - t - 1} \gamma^k R_{t + k + 1}$$
- **Policy Objective**: Discover an optimal policy $\pi^*$ maximizing expected cumulative return:
  $$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[ G_0 \right] = \arg\max_\pi \mathbb{E}_\pi \left[ \sum_{t=0}^{T-1} \gamma^t R_{t+1} \right]$$

---

## 4. State-Space Representation: 45 vs. 405 States

### 4.1 Feature Definitions & Discretization Thresholds

The state representation directly dictates what information the agent observes before choosing an action. We implement and evaluate two state representations:

```
+-----------------------------------------------------------------------------------+
| BASELINE REPRESENTATION (3 Features -> 45 States)                                |
+-----------------------------------------------------------------------------------+
| Feature 1: CPU Utilization (U_t)    -> 3 Bins: Low (<50%), Normal [50-80%], High (>80%) |
| Feature 2: Current Capacity (C_t)   -> 5 Bins: {1, 2, 3, 4, 5} Active VM Servers  |
| Feature 3: Demand Trend (Delta D_t) -> 3 Bins: Decreasing (<-5), Stable, Increasing (>+5) |
| Total Combinations: 3 * 5 * 3 = 45 States                                        |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
| PROPOSED ENHANCED REPRESENTATION (5 Features -> 405 States)                       |
+-----------------------------------------------------------------------------------+
| Feature 1: CPU Utilization (U_t)    -> 3 Bins: Low, Normal, High                  |
| Feature 2: Current Capacity (C_t)   -> 5 Bins: 1 to 5 Servers                     |
| Feature 3: Demand Trend (Delta D_t) -> 3 Bins: Decreasing, Stable, Increasing     |
| Feature 4: Network Traffic (N_t)    -> 3 Bins: Low, Medium, High (Tertiles)       |
| Feature 5: Execution Time (E_t)     -> 3 Bins: Fast, Acceptable, Critical (Tertiles) |
| Total Combinations: 3 * 5 * 3 * 3 * 3 = 405 States                               |
+-----------------------------------------------------------------------------------+
```

### 4.2 State Encoding Index Formulation
For tabular algorithms, the multi-discrete tuple $\mathbf{s} = (u, c, d, n, e)$ is uniquely mapped to a scalar integer index $s_{\text{idx}} \in [0, 404]$ via mixed-radix dimensional strides:
$$s_{\text{idx}} = u \times (5 \times 3 \times 3 \times 3) + c \times (3 \times 3 \times 3) + d \times (3 \times 3) + n \times (3) + e$$
$$s_{\text{idx}} = 81 u + 27 c + 9 d + 3 n + e$$
where $u \in \{0, 1, 2\}$, $c \in \{0, 1, 2, 3, 4\}$, $d \in \{0, 1, 2\}$, $n \in \{0, 1, 2\}$, $e \in \{0, 1, 2\}$.

For the baseline 45-state representation, features $n$ and $e$ are masked to index 0:
$$s_{\text{idx}}^{\text{base}} = 9 u + 3 c + d, \quad s_{\text{idx}}^{\text{base}} \in [0, 44]$$

### 4.3 Theoretical Motivation & The Curse of Dimensionality
- **The Hypothesis**: In modern cloud systems, network packet arrival at load balancers precedes CPU core scheduling. Including network traffic ($N_t$) provides a **leading indicator**, enabling proactive scaling *before* utilization spikes. Execution time ($E_t$) captures non-linear queue backlog latency that CPU monitors fail to detect early.
- **The Risk (Curse of Dimensionality)**: Expanding from 45 to 405 states represents a $9\times$ expansion of the state space. For tabular RL, every state-action pair $(s, a)$ requires multiple exploratory visits to compute an accurate expectation. In workloads where the added features lack predictive correlation, the 405-state model dilutes training observations across 1,215 state-action pairs, increasing sample complexity without improving decision quality.

---

## 5. Action Space & Cluster Dynamics

### 5.1 Action Space Definition
The action space is discrete and compact:
$$\mathcal{A} = \{0: \text{Scale In } (-1), \quad 1: \text{Hold } (0), \quad 2: \text{Scale Out } (+1)\}$$

### 5.2 Capacity Clamping Dynamics
At step $t$, the agent selects action $a_t \in \{-1, 0, +1\}$. The cluster capacity for the next interval is updated and strictly clamped within physical provisioning limits $[C_{\min}, C_{\max}] = [1, 5]$:
$$C_{t+1} = \text{clip}\left( C_t + a_t, \; 1, \; 5 \right) = \max\left(1, \; \min\left(5, \; C_t + a_t\right)\right)$$

### 5.3 Cluster Serving Capacity & Utilization
- **Unit Server Capacity**: Each server handles $\mu = 20$ units of demand.
- **Total Cluster Serving Capacity**: $K_{t+1} = \mu \times C_{t+1} = 20 \times C_{t+1}$ (Range: 20 to 100 units).
- **Instantaneous CPU Utilization**:
  $$U_t = \min\left(100.0, \; \frac{D_t}{20 \times C_t} \times 100.0\right)\%$$

---

## 6. Exact Multi-Objective Reward Function

The reward function mathematically encodes the multi-objective trade-off between operational performance, SLA compliance, monetary expenditure, and cluster stability.

### 6.1 Mathematical Formulation
$$R_t = R_{\text{bonus}}(U_t) - P_{\text{SLA}}(U_t) - P_{\text{under}}(U_t) - P_{\text{cost}}(C_{t+1}) - P_{\text{churn}}(a_t)$$

where:
1. **Optimal Operation Bonus ($R_{\text{bonus}}$)**:
   $$R_{\text{bonus}}(U_t) = \begin{cases} +10.0, & \text{if } 50\% \le U_t \le 80\% \\ 0.0, & \text{otherwise} \end{cases}$$
2. **SLA Breach Penalty ($P_{\text{SLA}}$)**:
   $$P_{\text{SLA}}(U_t) = \begin{cases} 50.0, & \text{if } U_t > 80\% \\ 0.0, & \text{otherwise} \end{cases}$$
3. **Underutilization Penalty ($P_{\text{under}}$)**:
   $$P_{\text{under}}(U_t) = \begin{cases} 5.0, & \text{if } U_t < 20\% \\ 0.0, & \text{otherwise} \end{cases}$$
4. **Infrastructure Resource Cost ($P_{\text{cost}}$)**:
   $$P_{\text{cost}}(C_{t+1}) = 0.5 \times C_{t+1}$$
5. **Reconfiguration Churn Penalty ($P_{\text{churn}}$)**:
   $$P_{\text{churn}}(a_t) = \begin{cases} 0.5, & \text{if } a_t \ne 0 \text{ (i.e., scaling up or down)} \\ 0.0, & \text{if } a_t = 0 \text{ (maintain capacity)} \end{cases}$$

### 6.2 Worked Numerical Example
- **Case 1: Balanced Operation**
  - Current Capacity $C_t = 3$ (Serves 60 units), Incoming Demand $D_t = 42$ units.
  - Utilization $U_t = (42 / 60) \times 100 = 70.0\%$ (Within $[50\%, 80\%]$).
  - Action taken $a_t = 0$ (Hold capacity $\to C_{t+1} = 3$).
  - Calculation:
    $$R = +10.0 - 0.0 - 0.0 - (0.5 \times 3) - 0.0 = +10.0 - 1.5 = \mathbf{+8.5}$$
- **Case 2: Severe SLA Breach**
  - Current Capacity $C_t = 2$ (Serves 40 units), Incoming Demand $D_t = 38$ units.
  - Utilization $U_t = (38 / 40) \times 100 = 95.0\%$ ($> 80\%$, SLA Breach).
  - Action taken $a_t = +1$ (Scale up $\to C_{t+1} = 3$).
  - Calculation:
    $$R = 0.0 - 50.0 - 0.0 - (0.5 \times 3) - 0.5 = -50.0 - 1.5 - 0.5 = \mathbf{-52.0}$$

---

## 7. Detailed Algorithm Formulations

We implement four reinforcement learning algorithms and two industrial heuristic baselines:

### 7.1 Random Policy (Lower Bound Baseline)
At each step $t$, samples an action uniformly at random from $\mathcal{A}$:
$$a_t \sim \mathcal{U}(\{-1, 0, +1\})$$
Establishes the minimum baseline performance under uncoordinated scaling.

### 7.2 Threshold-Based Policy (Industrial Baseline)
Replicates the dominant rule-based autoscaling logic used in commercial platforms (e.g., AWS EC2 Auto Scaling Groups):
$$a_t = \begin{cases} +1, & \text{if } U_t > 80\% \text{ and } C_t < 5 \\ -1, & \text{if } U_t < 20\% \text{ and } C_t > 1 \\ 0, & \text{otherwise} \end{cases}$$

### 7.3 Q-Learning (Off-Policy TD Control)
- **Principle**: Learns the optimal action-value function $Q^*(s, a)$ independently of the agent's behavioral exploration policy.
- **Bellman Optimality Update Equation**:
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a \in \mathcal{A}} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
- **Exploration Strategy**: $\epsilon$-greedy action selection with exponential decay:
  $$\pi(a \mid s) = \begin{cases} \arg\max_{a'} Q(s, a'), & \text{with probability } 1 - \epsilon \\ \text{random action from } \mathcal{A}, & \text{with probability } \epsilon \end{cases}$$
  $$\epsilon_{k+1} = \max(\epsilon_{\min}, \; \epsilon_k \times \delta_\epsilon), \quad \epsilon_0 = 1.0, \; \epsilon_{\min} = 0.01, \; \delta_\epsilon = 0.995$$
- **Hyperparameters**: $\alpha = 0.1$, $\gamma = 0.99$, $\epsilon_0 = 1.0$, $\epsilon_{\min} = 0.01$, decay $= 0.995$.

### 7.4 SARSA (On-Policy TD Control)
- **Principle**: Updates action values based on the action $A_{t+1}$ actually executed under the behavioral exploration policy.
- **Update Equation**:
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
- **Significance in Autoscaling**: Because SARSA factors in exploratory actions, it learns a more conservative policy in hazardous regions, deliberately maintaining slightly higher capacity to buffer against the steep $-50$ penalty of SLA breaches.

### 7.5 Double Deep Q-Network (Double DQN)
- **Principle**: Approximates action values using deep neural networks while decoupling action selection from action evaluation to eliminate overestimation bias.
- **Architecture**:
  - Input: Normalized 5-dimensional continuous state vector $\mathbf{x} \in \mathbb{R}^5$.
  - Hidden Layers: Fully connected MLP with two hidden layers: Dense(128, ReLU) $\to$ Dense(128, ReLU).
  - Output: 3 continuous Q-values corresponding to each scaling action.
- **Target Computation**:
  $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \; \arg\max_{a} Q(S_{t+1}, a; \theta); \; \theta^-\right)$$
- **Loss Function**: Mean Squared Error (MSE) over sampled minibatch $\mathcal{B}$:
  $$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{(s, a, r, s', d) \in \mathcal{B}} \left( Y_t^{\text{DoubleQ}} - Q(s, a; \theta) \right)^2$$
- **Stability Hyperparameters**:
  - Experience Replay Buffer: 100,000 transitions.
  - Minibatch Size: 64.
  - Optimizer: Adam with learning rate $\eta = 5 \times 10^{-4}$.
  - Polyak Soft Target Update: $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ with $\tau = 0.001$.

### 7.6 Dyna-Q (Model-Based Planning)
- **Principle**: Unifies direct reinforcement learning, environment model learning, and simulated background planning.
- **Algorithm Flow**:
  1. **Direct RL Step**: Execute action $A_t$, observe $R_{t+1}, S_{t+1}$, and update:
     $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
  2. **Model Learning Step**: Update internal deterministic model:
     $$\text{Model}(S_t, A_t) \leftarrow (R_{t+1}, S_{t+1})$$
     Record $(S_t, A_t)$ in visited set $\mathcal{V}$.
  3. **Simulated Planning Phase**: Repeat $k$ times ($k \in \{0, 5, 10, 20, 50\}$):
     - Sample random previously observed state-action pair: $(\tilde{S}, \tilde{A}) \sim \mathcal{V}$.
     - Query learned transition model: $(\tilde{R}, \tilde{S}') = \text{Model}(\tilde{S}, \tilde{A})$.
     - Execute simulated Q-update:
       $$Q(\tilde{S}, \tilde{A}) \leftarrow Q(\tilde{S}, \tilde{A}) + \alpha \left[ \tilde{R} + \gamma \max_{a} Q(\tilde{S}', a) - Q(\tilde{S}, \tilde{A}) \right]$$

```
+-----------------------------------------------------------------------------+
|                          DYNA-Q PLANNING CYCLE                              |
+-----------------------------------------------------------------------------+
|                                                                             |
|      +------------+        Real Action A_t         +---------------+        |
|      |            | -----------------------------> |               |        |
|      |  RL Agent  |                                |  Environment  |        |
|      | (Q-Table)  | <----------------------------- |               |        |
|      +------------+   Next State S_{t+1}, R_{t+1}  +---------------+        |
|         ^      |                                           |                |
|         |      | (1) Direct Q-Update                       |                |
|         |      v                                           |                |
|         |   [Q-Table Update]                               |                |
|         |                                                  |                |
|         | (3) Simulated Planning                           v                |
|         |     Updates (k times)                  (2) Model Learning         |
|         |                                                  |                |
|    +---------+                                             v                |
|    | Search  | <----------------- Query -----------+---------------+        |
|    | Control |                                     | Learned Model |        |
|    +---------+      Sample (s, a) -> (r, s')       |  Model(s, a)  |        |
|                                                    +---------------+        |
+-----------------------------------------------------------------------------+
```

---

## 8. Dataset Characteristics vs. Synthetic Workload

A major contribution of this study is evaluating algorithm robustness across two fundamentally distinct workload domains:

### 8.1 Empirical Enterprise Telemetry: Kaggle `vmCloud_data.csv`
- **Source**: Enterprise Cloud Computing Telemetry dataset (Kaggle).
- **Size**: 338.30 MB raw text CSV; 687,148 rows.
- **Preprocessing Pipeline**:
  - Missing values: Rows containing null or NaN values were completely removed.
  - Telemetry features extracted: CPU utilization, Memory utilization, Network traffic, Execution time, and Request count.
  - Standardized temporal partition: First 80% (549,718 steps) reserved for training; final 20% (137,430 steps) strictly reserved for evaluation.
  - Cached into optimized compressed binary format (`vmCloud_data_cleaned.npz`, 23.3 MB).
- **Empirical Properties**:
  - Cross-correlation between Network Ingress and CPU Demand: $r = 0.00041$.
  - Lag-1 Autocorrelation of Demand: $\rho_1 = -0.00003$.
  - *Conclusion*: The real dataset behaves as **identically and independently distributed (i.i.d.) stochastic white noise**. There are no predictable leading signals.

### 8.2 Synthetic Controlled Workload (Predictive Leading Signal)
- **Mathematical Specification**:
  $$D_t = 50.0 + 25.0 \sin\left(\frac{2\pi t}{50}\right) + 10.0 \sin\left(\frac{2\pi t}{200}\right) + \epsilon_t + S_t$$
  where $\epsilon_t \sim \mathcal{N}(0, 2.0^2)$ and $S_t$ denotes Poisson burst spikes.
- **Engineered Predictive Signals**:
  1. *Leading Network Traffic*: $N_t = 0.8 \cdot D_{t+1} + \eta_t$. Traffic ingress precedes compute load by exactly 1 time step, simulating buffering at the reverse proxy.
  2. *Queueing Execution Latency*: Non-linear latency expansion $E_t = E_{\text{base}} / (1 - \min(0.95, U_t/100))$.
- **Scientific Rationale**: The synthetic workload tests the hypothesis: *Can the agent successfully discover and exploit causal leading signals when they exist?* The empirical Kaggle dataset tests the counter-hypothesis: *How do agents behave when exposed to unstructured white-noise fluctuations?*

---

## 9. Academic Attribution & Integrity Statement

In adherence to academic research integrity:
- **Reference Codebase**: This project utilizes architectural ideas from the open-source repository `Cloud-Autoscaling-using-RL` by Rah-ds (`https://github.com/rah-ds/Cloud-Autoscaling-using-RL`).
- **Original Code Retained**: The original repository code remains completely intact and unmodified in the parent directory.
- **Our Independent Implementations**:
  1. Development of the Gymnasium-compliant `EnhancedAutoscalingEnv` supporting dual workloads and dynamic feature masking.
  2. Mathematical formulation and code implementation of the 5-feature, 405-state discrete space.
  3. Ground-up implementations of Dyna-Q with configurable background planning steps ($k \in \{0, 5, 10, 20, 50\}$).
  4. Implementation of Double DQN with Polyak soft-target updating and replay buffer management.
  5. The multi-seed, 1,000-episode statistical evaluation framework incorporating paired $t$-tests and effect sizes.

