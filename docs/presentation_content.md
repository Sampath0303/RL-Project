# Enhanced-State Reinforcement Learning for Dynamic Cloud Resource Auto-Scaling
## Academic Presentation & Defense Master Deck (20 Slides)

---

### Slide 1: Title & Team Credentials
- **Slide Title**: Enhanced-State Reinforcement Learning for Dynamic Cloud Resource Auto-Scaling Using Q-Learning, SARSA, DQN, and Dyna-Q
- **Subtitle**: A Rigorous Comparative Investigation of Model-Free vs. Model-Based RL on Synthetic and Empirical Workloads
- **Academic Context**: Senior Capstone / Advanced Machine Learning Research Project
- **Key Concepts**: Markov Decision Processes, Dynamic Auto-Scaling, State-Space Augmentation, Model-Based Planning, Service Level Agreement (SLA) Adherence, Resource Optimization.
- **Visual / Layout Recommendation**:
  - Left: Clean university/department insignia and project title banner.
  - Right: Conceptual diagram of a cloud data center connected in a feedback loop with an RL auto-scaling controller.
- **Speaker Notes**:
  > "Good morning, respected professors and committee members. Today, we present our research on 'Enhanced-State Reinforcement Learning for Dynamic Cloud Resource Auto-Scaling.' In cloud computing, balancing infrastructure provisioning cost with application performance is a fundamental challenge. Traditional reactive threshold rules either over-provision—wasting thousands of dollars—or under-provision—causing catastrophic Service Level Agreement violations. We formulate cloud autoscaling as an episodic Markov Decision Process, systematically implement four reinforcement learning paradigms—Q-Learning, SARSA, Double Deep Q-Networks (DQN), and model-based Dyna-Q—and critically investigate the impact of expanding the state representation from 45 to 405 discrete states using both synthetic and real-world enterprise cloud workloads."

---

### Slide 2: Introduction & Cloud Computing Context
- **Slide Title**: The Cloud Auto-Scaling Dilemma
- **Main Points**:
  - **The Elasticity Promise**: Cloud computing platforms (AWS EC2, Google Cloud Compute Engine, Azure VMs) promise on-demand elasticity—provisioning compute resources precisely when needed.
  - **The Autoscaling Trilemma**:
    1. *Cost Efficiency*: Minimize active virtual machine (VM) instances and idle billing overhead.
    2. *Quality of Service (QoS)*: Prevent latency spikes, queue bottlenecks, and SLA breach penalties.
    3. *System Stability*: Avoid resource thrashing (rapid, oscillatory scaling actions that incur cold-start delays and virtualization churn).
  - **Industrial Reality**: Large-scale cloud infrastructures operate at average CPU utilization levels between 15% and 35%, wasting immense financial and energy resources due to fear of SLA breaches.
- **Visual / Layout Recommendation**:
  - Three-way Venn diagram illustrating the trade-off triangle: Infrastructure Cost vs. SLA Violations vs. Reconfiguration Churn.
- **Speaker Notes**:
  > "Cloud elasticity is simple in concept but difficult in practice. Providers bill per instance-hour or per-second, incentivizing tenants to minimize provisioned capacity. However, user request traffic fluctuates unpredictably. When traffic surges exceed server capacity, requests queue up, execution latency blows past SLA thresholds, and businesses incur steep penalties. In practice, operators leave clusters severely over-provisioned at 20% to 30% utilization as a safety cushion. Dynamic autoscaling seeks to replace this manual, wasteful over-provisioning with an intelligent autonomous controller."

---

### Slide 3: Problem Statement & Shortcomings of Existing Approaches
- **Slide Title**: Problem Formulation: Why Rule-Based Autoscaling Fails
- **Main Points**:
  - **Static Provisioning**: Peak-load dimensioning guarantees high availability but wastes 60–80% of compute budget during off-peak hours.
  - **Rule-Based Threshold Policies** (e.g., AWS Auto Scaling Groups):
    - Example: *If CPU utilization > 80% for 5 minutes, add 1 VM; if CPU < 20%, terminate 1 VM.*
    - *Fatal Flaws*: Purely reactive, prone to delayed response during traffic spikes, requires brittle manual tuning of thresholds and cooldown periods, and exhibits destructive oscillatory hunting behavior.
  - **Time-Series Forecasting (ARIMA / LSTM)**:
    - Predicts future demand accurately only if workloads exhibit strict stationarity or clean seasonality.
    - Decoupled from the control action: Forecasting future traffic does not solve the multi-objective optimization problem of choosing whether, when, and how much capacity to allocate under non-linear latency and cost penalties.
- **Visual / Layout Recommendation**:
  - Time-series plot showing reactive threshold delay: Traffic spikes at $t$, threshold triggers at $t+k$, SLA violation occurs between $t$ and $t+k$.
- **Speaker Notes**:
  > "Why do standard industry solutions fall short? Commercial cloud environments rely heavily on static threshold rules. If CPU exceeds 80%, add a server; if it drops below 20%, remove one. The fatal flaw is latency: threshold mechanisms only react after an overload has already occurred, by which time the system is already violating SLAs. Furthermore, threshold rules cannot balance trade-offs—they have no inherent concept of monetary cost or reconfiguration churn. While deep learning time-series forecasters can predict traffic, prediction is not control. Reinforcement Learning solves this by directly optimizing a closed-loop control policy that balances immediate actions against long-term discounted cumulative returns."

---

### Slide 4: Research Motivation & Key Questions
- **Slide Title**: Research Motivation & Research Questions
- **Main Points**:
  - **Motivation**: Can reinforcement learning agents discover proactive, cost-efficient scaling policies that eliminate SLA violations while keeping capacity tightly matched to demand?
  - **Primary Research Questions (RQs)**:
    - **RQ1 (State Space)**: Does augmenting the state representation with network traffic and execution latency (expanding the state space from 45 to 405 states) measurably improve autoscaling decisions?
    - **RQ2 (Model-Based Planning)**: Does Dyna-Q, which updates its value function via simulated experiences from a learned transition model ($k \in \{0, 5, 10, 20, 50\}$), outperform model-free tabular methods (Q-Learning, SARSA) and function approximation (DQN)?
    - **RQ3 (Workload Generalization)**: How do model-free and model-based RL agents behave when transferred from a structured workload (predictable leading signals) to real-world cloud traces (uncorrelated stochastic noise)?
- **Visual / Layout Recommendation**:
  - Two conceptual comparison boxes:
    - Box 1: *State Space*: 3 features (45 states) $\to$ 5 features (405 states).
    - Box 2: *Planning*: Direct RL (Model-Free) vs. Direct RL + Simulated Model Planning (Dyna-Q).
- **Speaker Notes**:
  > "Our investigation focuses on two core contributions to the field. First: State Space Design. In prior literature, agents typically observe only CPU utilization, capacity, and demand trend—a 45-state discretization. We hypothesize that incorporating network traffic as a leading indicator and execution time as a queueing latency indicator will allow the agent to detect incoming traffic waves before CPU spikes occur. Second: Planning vs. Model-Free Learning. We ask whether Dyna-Q, which builds an internal transition model and performs simulated background updates, can achieve superior sample efficiency and policy quality compared to standard Q-Learning, SARSA, and deep neural network approaches."

---

### Slide 5: Project Objectives & Research Methodology
- **Slide Title**: Project Objectives & Experimental Protocol
- **Main Points**:
  - **Objective 1**: Build a modular, Gymnasium-compliant cloud autoscaling environment (`EnhancedAutoscalingEnv`) with pluggable data modes (`synthetic` vs. `dataset`) and feature masks.
  - **Objective 2**: Implement from first principles four RL algorithms: Q-Learning, SARSA, Double DQN, and Dyna-Q (with configurable planning steps $k$).
  - **Objective 3**: Conduct a multi-seed, large-scale empirical benchmark across 34 distinct configurations:
    - 1,000 training episodes per run.
    - 5 independent random seeds (`[0, 1, 2, 3, 4]`).
    - 100 held-out evaluation episodes per seed evaluated greedily ($\epsilon = 0$).
  - **Objective 4**: Perform formal statistical hypothesis testing: two-tailed paired $t$-tests across matching seeds, 95% confidence intervals, and Cohen's $d_z$ effect sizes.
- **Visual / Layout Recommendation**:
  - High-level methodology pipeline flowchart: Config $\to$ Workload/Dataset Loader $\to$ Gymnasium Environment $\to$ Agent Training (5 Seeds $\times$ 1,000 Eps) $\to$ Greedy Test Evaluation $\to$ Statistical Analysis & Plotting.
- **Speaker Notes**:
  > "To ensure rigorous scientific validity, we established a strict experimental methodology. We developed a standardized Gymnasium-compatible environment supporting dual data sources: a synthetic mathematical workload generator with leading network signals, and a 338-megabyte empirical dataset from Kaggle comprising 687,000 virtual machine telemetry points. Every experiment is evaluated across five independent seeds for 1,000 training episodes, followed by 100 held-out evaluation episodes per seed. Crucially, evaluation is performed with exploration disabled to measure the true deterministic policy quality. All algorithmic comparisons are verified using paired t-tests on matching seeds to guarantee statistical significance."

---

### Slide 6: Proposed System Architecture
- **Slide Title**: End-to-End System Architecture
- **Main Points**:
  - **Workload Layer**: Generates or streams incoming user demand $D_t$, network ingress/egress $N_t$, and task execution latency $E_t$.
  - **Cloud Infrastructure Simulation Layer**:
    - Capacity: Discrete server instances $C_t \in \{1, 2, 3, 4, 5\}$.
    - Server Capacity Coefficient: Each server serves 20 units of demand (Cluster max capacity = 100).
    - CPU Utilization: $U_t = \min\left(100, \frac{D_t}{20 \times C_t} \times 100\right)\%$.
  - **Telemetry & Feature Discretizer**:
    - Discretizes continuous metrics into ordinal bins: Utilization (3), Capacity (5), Trend (3), Network (3), Execution Time (3).
  - **RL Controller Layer**:
    - Selects scaling action $A_t \in \{-1, 0, +1\}$.
  - **Execution & Actuation Layer**:
    - Updates cluster capacity: $C_{t+1} = \text{clip}(C_t + A_t, 1, 5)$.
- **Visual / Layout Recommendation**:
  - Architectural block diagram illustrating the closed-loop feedback path: Workload $\to$ Cloud Cluster $\to$ Feature Extractor $\to$ RL Agent $\to$ Actuator $\to$ Cloud Cluster.
- **Speaker Notes**:
  > "Here is our complete system architecture. The environment simulates a production web cluster where capacity can scale between 1 and 5 servers, each capable of handling 20 demand units. At each discrete time step, the environment computes the instantaneous CPU utilization as the ratio of current demand to total provisioned cluster capacity. The feature extraction module continuously computes demand trend, categorizes network traffic, and measures execution time. These signals are mapped into our discrete state representation and delivered to the RL agent. The agent selects an action to scale down by one VM, maintain capacity, or scale up by one VM, closing the feedback loop."

---

### Slide 7: Dataset Description & Empirical Profiling
- **Slide Title**: The Kaggle Cloud Performance Dataset
- **Main Points**:
  - **Source**: Enterprise Cloud VM Performance Telemetry (`vmCloud_data.csv`, 338.3 MB, Kaggle).
  - **Features**: CPU Utilization (%), Memory Utilization (%), Network In/Out (KB/s), Execution Time (s), Request Count.
  - **Cleaning & Preprocessing**:
    - 687,148 rows loaded; missing values dropped.
    - Extracted 5 canonical columns, converted to `float32`.
    - Temporal 80/20 train/test split (Train: 549,718 steps; Test: 137,430 steps).
    - Offline cleaning cached to `vmCloud_data_cleaned.npz` (23.3 MB) for instantaneous loading.
  - **Statistical Discovery**:
    - Cross-correlation between Network Traffic and CPU: $r = 0.00041$ (near zero).
    - Lag-1 autocorrelation of CPU utilization: $\rho_1 = -0.00003$ (white noise).
- **Visual / Layout Recommendation**:
  - Two distribution plots: Histogram of CPU utilization (skewed, high-variance) and autocorrelation function plot showing an immediate drop to zero at lag 1.
- **Speaker Notes**:
  > "For empirical validation, we utilized a real-world cloud performance dataset from Kaggle comprising over 687,000 telemetry readings. We conducted rigorous exploratory data analysis and uncovered a vital statistical insight: the telemetry points exhibit near-zero autocorrelation (negative 0.00003) and virtually zero cross-correlation between network traffic and CPU utilization (0.0004). In other words, the Kaggle dataset behaves as independent, identically distributed stochastic white noise rather than a smooth, autocorrelated time series. This empirical property has profound implications for RL state representation, as we will demonstrate in our findings."

---

### Slide 8: Synthetic Workload Formulation (Leading Signal)
- **Slide Title**: Synthetic Workload: Controlled Leading Signal Dynamics
- **Main Points**:
  - **Motivation**: To rigorously evaluate whether RL agents can exploit predictive indicators, we engineered a realistic, non-stationary synthetic workload generator.
  - **Mathematical Demand Function**:
    $$D_t = D_{\text{base}} + A_{\text{daily}} \sin\left(\frac{2\pi t}{T_{\text{daily}}}\right) + A_{\text{weekly}} \sin\left(\frac{2\pi t}{T_{\text{weekly}}}\right) + \epsilon_t + S_t$$
    where $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$ represents noise and $S_t$ represents Poisson-distributed burst spikes.
  - **Leading Network Signal**:
    $$N_t = \beta \cdot D_{t+1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_N^2)$$
    Network ingress arrives 1 time step *ahead* of server demand, modeling web requests buffering in reverse proxies/load balancers.
  - **Queueing Execution Latency**:
    $$E_t = E_{\text{base}} \cdot \frac{1}{1 - \min(0.95, U_t/100)} + \xi_t$$
    Models non-linear M/M/1 queueing delay as utilization approaches saturation.
- **Visual / Layout Recommendation**:
  - Dual-axis time-series plot comparing Demand $D_t$ and Network Traffic $N_t$, highlighting how network traffic peaks precede demand peaks by 1 step.
- **Speaker Notes**:
  > "To evaluate whether an enhanced state space provides predictive value, we constructed a synthetic workload simulator based on real-world cloud traffic characteristics. The demand function incorporates diurnal daily cycles, weekly seasonality, Gaussian background fluctuations, and sudden traffic spikes. Crucially, we designed the network traffic feature as a leading indicator: network packets arrive at load balancers one time step before they translate into CPU compute demand. Furthermore, execution time is modeled using M/M/1 queueing theory, where latency increases asymptotically as utilization approaches 100%. This synthetic environment provides a ground-truth benchmark to test whether agents can proactively anticipate demand surges."

---

### Slide 9: Reinforcement Learning Formulation (MDP)
- **Slide Title**: Formal Markov Decision Process (MDP) Definition
- **Main Points**:
  - The cloud autoscaling problem is formulated as a 5-tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:
    - **State $\mathcal{S}$**: Discretized observation of cloud cluster operating conditions.
    - **Action $\mathcal{A}$**: Discrete scaling commands $\{-1, 0, +1\}$ modifying server capacity.
    - **Transition Probability $\mathcal{P}(s' \mid s, a)$**: Governed by external workload arrival dynamics and cluster capacity actuation.
    - **Reward Function $\mathcal{R}(s, a, s')$**: Scalar feedback quantifying operating quality.
    - **Discount Factor $\gamma$**: $0.99$, balancing immediate operational costs against long-term stability.
  - **Episodic Structure**:
    - Episode length $T = 200$ time steps.
    - Cumulative discounted return: $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$.
    - Objective: Find optimal policy $\pi^* = \arg\max_\pi \mathbb{E}_\pi [G_0]$.
- **Visual / Layout Recommendation**:
  - Standard MDP interaction loop diagram between Agent and Environment with annotated time indices $t, t+1$.
- **Speaker Notes**:
  > "We formally define cloud autoscaling as an episodic Markov Decision Process. At each discrete time step t, the agent receives an observation state s_t, executes a scaling decision a_t, transitions to state s_{t+1}, and receives a numerical reward r_t. An episode runs for 200 time steps, representing an operational scaling window. With a discount factor of gamma = 0.99, the agent must look ahead and anticipate future load conditions rather than myopically minimizing instantaneous resource usage. The agent's goal is to learn a policy pi that maximizes the expected cumulative discounted return."

---

### Slide 10: State-Space Representation: 45 vs. 405 States
- **Slide Title**: State-Space Architecture: Baseline (45) vs. Enhanced (405)
- **Main Points**:
  - **Original Baseline State Space (3 Features $\to$ 45 States)**:
    1. *CPU Utilization ($U_t$)*: 3 bins $\to$ Low ($<50\%$), Optimal ($[50\%, 80\%]$), High ($>80\%$).
    2. *Current Capacity ($C_t$)*: 5 bins $\to$ $\{1, 2, 3, 4, 5\}$ servers.
    3. *Demand Trend ($\Delta D_t$)*: 3 bins $\to$ Decreasing, Stable, Increasing.
    $$\text{Size} = 3 \times 5 \times 3 = 45 \text{ discrete states}$$
  - **Proposed Enhanced State Space (5 Features $\to$ 405 States)**:
    4. *Network Traffic Level ($N_t$)*: 3 bins $\to$ Low, Normal, High (tertiles).
    5. *Execution Time Level ($E_t$)*: 3 bins $\to$ Fast, Acceptable, Critical (tertiles).
    $$\text{Size} = 3 \times 5 \times 3 \times 3 \times 3 = 405 \text{ discrete states}$$
  - **The Trade-Off**: A $9\times$ larger state space captures multi-modal leading indicators but increases sample complexity and requires significantly more visits to achieve value convergence.
- **Visual / Layout Recommendation**:
  - Dimensional breakdown diagram comparing a 3D grid (45 states) against a 5D hyper-grid (405 states).
- **Speaker Notes**:
  > "Slide 10 details our core state-space design. In prior literature, the agent is restricted to three features: CPU utilization categorized into three levels, capacity across five server levels, and demand trend categorized as falling, stable, or rising—yielding 45 discrete states. Our enhanced representation adds two critical telemetry dimensions: network traffic and execution time, each discretized into three quantile-based bins. This expands the state space to 405 states. The theoretical motivation is that network traffic provides advance warning of incoming traffic, while execution time signals queue saturation before CPU monitors reflect it. However, the fundamental trade-off is the curse of dimensionality: a 405-state space requires nine times more state visits for tabular Q-learning to converge."

---

### Slide 11: Action Space & Exact Reward Function
- **Slide Title**: Mathematical Formulation of Action Space & Multi-Objective Reward
- **Main Points**:
  - **Action Space**: $\mathcal{A} = \{0: \text{Scale Down } (-1), 1: \text{Hold } (0), 2: \text{Scale Up } (+1)\}$
    $$C_{t+1} = \max\left(1, \min\left(5, C_t + a_t\right)\right)$$
  - **Implemented Multi-Objective Reward Function**:
    $$R_t = R_{\text{perf}}(U_t) - P_{\text{SLA}}(U_t) - P_{\text{under}}(U_t) - P_{\text{cost}}(C_{t+1}) - P_{\text{churn}}(a_t)$$
  - **Component Definitions**:
    - *Normal Operating Bonus*: $+10.0$ if $50\% \le U_t \le 80\%$.
    - *SLA Violation Penalty*: $-50.0$ if $U_t > 80\%$.
    - *Underutilization Penalty*: $-5.0$ if $U_t < 20\%$.
    - *Resource Infrastructure Cost*: $-0.5 \times C_{t+1}$ (cost per active server).
    - *Scaling Churn Penalty*: $-0.5$ if $a_t \ne 0$ (penalizes frequent reallocations).
- **Worked Numerical Example**:
  - *Scenario A (Optimal)*: $U_t = 65\%$, $C_t = 3$, $a_t = 0 \implies R = +10.0 - 0 - 0 - (0.5 \times 3) - 0 = \mathbf{+8.5}$.
  - *Scenario B (SLA Breach)*: $U_t = 92\%$, $C_t = 2$, $a_t = +1 \implies R = 0 - 50.0 - 0 - (0.5 \times 3) - 0.5 = \mathbf{-52.0}$.
- **Visual / Layout Recommendation**:
  - Piecewise reward curve plotted against CPU Utilization ($0\%$ to $100\%$) showing the cliff drop below $20\%$ and above $80\%$.
- **Speaker Notes**:
  > "Our reward function balances three conflicting priorities: performance, monetary cost, and operational stability. When the system operates in the sweet spot between 50% and 80% utilization, the agent earns a +10 reward bonus. If utilization exceeds 80%, an SLA breach occurs, triggering a heavy -50 penalty. Conversely, if utilization falls below 20%, an underutilization penalty of -5 is applied. Every active server incurs a continuous infrastructure cost of -0.5 per time step, and every scaling action incurs a -0.5 churn penalty to prevent oscillation. In the worked example on the slide, maintaining optimal utilization yields +8.5 per step, whereas an SLA breach drops the reward to -52.0."

---

### Slide 12: Model-Free Tabular RL: Q-Learning & SARSA
- **Slide Title**: Model-Free Tabular Baselines: Q-Learning vs. SARSA
- **Main Points**:
  - **Q-Learning (Off-Policy TD Control)**:
    - Learns the optimal action-value function directly:
      $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]$$
    - *Key Characteristic*: Optimistic; assumes greedy behavior during future transitions regardless of exploration policy.
  - **SARSA (On-Policy TD Control)**:
    - Updates value function using the action actually selected by current policy:
      $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$
    - *Key Characteristic*: Safer in high-penalty environments; penalizes actions that might lead to exploratory SLA violations.
  - **Hyperparameters**:
    - Learning rate $\alpha = 0.1$, Discount factor $\gamma = 0.99$.
    - Exploration: $\epsilon$-greedy decaying from $\epsilon_0 = 1.0 \to \epsilon_{\min} = 0.01$ at decay rate $0.995$.
- **Visual / Layout Recommendation**:
  - Side-by-side backup diagrams comparing Q-Learning (max operator over next state) vs. SARSA (next action taken $A_{t+1}$).
- **Speaker Notes**:
  > "Slide 12 contrasts our two tabular baselines: Q-Learning and SARSA. Q-Learning is off-policy: it updates its Q-table using the maximum possible Q-value of the next state, assuming ideal greedy execution. SARSA is on-policy: it updates using the actual action A_{t+1} chosen by its epsilon-greedy exploration policy. In an autoscaling context where SLA violations carry severe -50 penalties, SARSA learns a more conservative policy that prefers slightly higher capacity margins to avoid accidental penalties during exploration. Both algorithms utilize identical hyperparameters: learning rate 0.1, discount factor 0.99, and exponential epsilon decay from 1.0 to 0.01."

---

### Slide 13: Deep Reinforcement Learning: Double DQN
- **Slide Title**: Deep Reinforcement Learning: Double DQN Architecture
- **Main Points**:
  - **Network Architecture**: Multi-Layer Perceptron (MLP):
    - Input: Continuous 5-dimensional state vector $\in \mathbb{R}^5$ (normalized).
    - Hidden Layers: Two dense layers of 128 units each with ReLU activations.
    - Output: 3 continuous Q-values corresponding to $\mathcal{A} \in \{-1, 0, +1\}$.
  - **Double DQN Decoupled Loss**:
    - Mitigates overestimation bias by using the online network to *select* the action and the target network to *evaluate* it:
      $$Y_t^{\text{DoubleQ}} = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a} Q(S_{t+1}, a; \theta_t); \theta_t^-\right)$$
      $$\mathcal{L}(\theta) = \mathbb{E}\left[ \left( Y_t^{\text{DoubleQ}} - Q(S_t, A_t; \theta) \right)^2 \right]$$
  - **Stability Mechanisms**:
    - Experience Replay Buffer: 100,000 transitions; minibatch size 64.
    - Polyak Soft Target Updates: $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ with $\tau = 0.001$.
    - Optimizer: Adam with learning rate $\eta = 5 \times 10^{-4}$.
- **Visual / Layout Recommendation**:
  - Neural network diagram: Input Layer (5) $\to$ Dense 128 (ReLU) $\to$ Dense 128 (ReLU) $\to$ Output Q-Values (3) with replay buffer and target network update arrows.
- **Speaker Notes**:
  > "To evaluate deep function approximation against tabular methods, we implemented Double Deep Q-Networks (Double DQN). Instead of discretizing states into tables, the continuous 5-dimensional state vector is fed directly into a multi-layer perceptron with two hidden layers of 128 units each. To eliminate the notorious overestimation bias of standard DQN, we decouple action selection from action evaluation using a target network updated via Polyak soft averaging with tau = 0.001. Transitions are stored in a 100,000-step circular replay buffer and sampled uniformly in mini-batches of 64. This provides a state-of-the-art deep RL benchmark."

---

### Slide 14: Model-Based Reinforcement Learning: Dyna-Q
- **Slide Title**: Model-Based RL: The Dyna-Q Architecture
- **Main Points**:
  - **Core Philosophy**: Integrate direct reinforcement learning, environment model learning, and simulated planning updates.
  - **Tri-Partite Algorithm Cycle**:
    1. *Real Experience Update*: Take action $A$, observe $R, S'$, apply standard Q-Learning TD update.
    2. *Model Learning*: Store transition $(S, A) \to (R, S')$ in a deterministic learned model table.
    3. *Planning Phase ($k$ Steps)*: Repeat $k$ times:
       - Sample a previously visited state-action pair $(\tilde{S}, \tilde{A}) \sim \text{VisitedPairs}$.
       - Query learned model: $\tilde{R}, \tilde{S}' = \text{Model}(\tilde{S}, \tilde{A})$.
       - Apply simulated Q-learning update:
         $$Q(\tilde{S}, \tilde{A}) \leftarrow Q(\tilde{S}, \tilde{A}) + \alpha \left[ \tilde{R} + \gamma \max_{a} Q(\tilde{S}', a) - Q(\tilde{S}, \tilde{A}) \right]$$
  - **Investigation Scope**: Evaluate planning intensity across $k \in \{0, 5, 10, 20, 50\}$.
- **Visual / Layout Recommendation**:
  - Sutton & Barto Dyna-Q cycle flowchart: Real Experience $\to$ Direct RL $\to$ Value Function $\leftarrow$ Planning $\leftarrow$ Search Control $\leftarrow$ Model $\leftarrow$ Model Learning.
- **Speaker Notes**:
  > "Slide 14 introduces our second major contribution: model-based planning via Dyna-Q. Unlike purely model-free algorithms that discard transitions after a single update, Dyna-Q constructs an internal model of the environment. After each real interaction step, the agent updates its Q-table, records the transition in its model, and then executes k simulated planning steps. In each planning step, it randomly samples a previously visited state-action pair, retrieves the predicted reward and next state from its model, and performs an additional simulated Q-learning update. We systematically test planning step counts of k = 0, 5, 10, 20, and 50 to investigate whether model planning accelerates policy convergence in cloud autoscaling."

---

### Slide 15: Experimental Setup & Complete 34-Configuration Matrix
- **Slide Title**: Experimental Design & Benchmark Matrix
- **Main Points**:
  - **Experiment Matrix (34 Configurations)**:
    - **Suite A (12 Runs)**: Main algorithm comparison on Full (405 states): Random, Threshold, Q-Learning, SARSA, Dyna-Q ($k=10$), DQN across Synthetic & Dataset.
    - **Suite B (14 Runs)**: State-Space & Feature Ablation: Base (45), Base+Network (135), Base+Exec (135), Full (405) for Q-Learning, SARSA, and Dyna-Q.
    - **Suite C (8 Runs)**: Dyna-Q Planning Sensitivity: $k \in \{0, 5, 20, 50\}$ across Synthetic & Dataset.
    - *Note on DQN*: DQN 45-state experiments postponed due to 19-hour compute budget and marked as *Not Evaluated*.
  - **Execution Protocol**:
    - 1,000 training episodes per configuration.
    - 5 independent random seeds (`[0, 1, 2, 3, 4]`).
    - 100 evaluation episodes per seed evaluated greedily ($\epsilon = 0$).
    - Total simulated training episodes: $34 \times 5 \times 1,000 = \mathbf{170,000}$ episodes.
- **Visual / Layout Recommendation**:
  - Structured matrix table grouping experiments by Suite A, B, and C with checkmarks indicating complete verification.
- **Speaker Notes**:
  > "To validate our hypotheses, we executed a massive computational benchmark of 34 distinct configurations, representing 170,000 training episodes and over 17,000 evaluation episodes. Suite A evaluates all six core algorithms on the full 405-state space. Suite B conducts controlled feature ablations across the base 45 states, base plus network, base plus execution latency, and full 405 states. Suite C tests Dyna-Q planning intensity across k = 0, 5, 20, and 50. Every single configuration was trained independently across five random seeds from scratch to ensure statistical reproducibility."

---

### Slide 16: Main Benchmark Results (Suite A)
- **Slide Title**: Empirical Benchmark: Synthetic vs. Real-World Workloads
- **Main Points**:
  - **Synthetic Workload Results (With Leading Signal)**:
    - **DQN**: **$+7,838.33 \pm 124.70$** | SLA: $16.0 \pm 3.6$ | Cost: $1,883.14$ (Highest Reward)
    - **Threshold Rule**: $+6,607.83 \pm 32.42$ | SLA: $32.4 \pm 0.4$ | Cost: $1,953.77$
    - **Q-Learning**: $+6,446.10 \pm 317.83$ | SLA: $13.7 \pm 2.9$ | Cost: $2,033.06$
    - **Dyna-Q ($k=10$)**: $+4,230.09 \pm 372.41$ | SLA: $16.1 \pm 8.3$ | Cost: $2,232.36$
    - **SARSA**: $+3,861.64 \pm 414.66$ | SLA: **$5.7 \pm 0.2$** | Cost: $2,390.49$ (Lowest SLA Violations)
    - **Random**: $-38,413.10 \pm 457.03$ | SLA: $442.2 \pm 3.6$ | Cost: $1,500.64$
  - **Real Kaggle Dataset Results (Uncorrelated Noise)**:
    - **SARSA**: **$-3,561.78 \pm 116.89$** | SLA: **$100.5 \pm 0.7$** | Cost: $2,497.00$
    - **Q-Learning**: **$-3,734.49 \pm 116.51$** | SLA: $100.5 \pm 0.7$ | Cost: $2,496.88$
    - **Dyna-Q ($k=10$)**: $-6,046.40 \pm 392.34$ | SLA: $123.9 \pm 5.0$ | Cost: $2,432.32$
    - **DQN**: $-13,733.43 \pm 2,515.96$ | SLA: $261.7 \pm 38.9$ | Cost: $2,051.81$ (Diverged)
    - **Threshold Rule**: $-20,897.15 \pm 72.55$ | SLA: $297.8 \pm 0.5$ | Cost: $1,949.76$
- **Visual / Layout Recommendation**:
  - Grouped bar chart comparing Mean Test Reward across algorithms, divided into Synthetic (positive rewards) vs. Dataset (negative rewards).
- **Speaker Notes**:
  > "Slide 16 presents the defining empirical comparison of our study. On the synthetic workload with structured signals, Deep Q-Networks achieved the highest overall reward of +7,838, effectively generalizing across continuous states. Tabular Q-Learning followed closely at +6,446, outperforming the threshold baseline while maintaining exceptionally low SLA violations of 13.7. SARSA achieved the absolute lowest SLA violation count of 5.7 due to its conservative on-policy nature. However, when deployed on the real Kaggle dataset, the rankings flipped dramatically: tabular SARSA and Q-Learning proved most robust, achieving -3,561 and -3,734. In contrast, DQN collapsed to -13,733. Why? Because the Kaggle dataset represents uncorrelated white noise: neural network function approximation overfitted to non-existent patterns and destabilized the replay buffer, whereas tabular agents simply learned a robust high-capacity baseline."

---

### Slide 17: State-Space Enhancement & Feature Ablation (Suite B)
- **Slide Title**: Contribution 1 Analysis: State-Space Augmentation (45 vs. 405)
- **Main Points**:
  - **Synthetic Workload (Leading Signal Exists)**:
    - Q-Learning Base (45 states): $+4,242.21 \pm 384.38$
    - Q-Learning Base+Network (135 states): $+6,172.38 \pm 187.97$ (**$+45.5\%$ gain**, $p < 0.001$)
    - Q-Learning Full (405 states): $+6,446.10 \pm 317.83$ (**$+51.9\%$ gain**, $p < 0.001$)
    - Dyna-Q Base (45 states): $+2,379.13 \to$ Dyna-Q Full (405 states): $+4,230.09$ (**$+77.8\%$ gain**)
    - *Finding*: When predictive signals exist, adding network traffic as a leading indicator provides massive, statistically significant performance improvements.
  - **Real Dataset (Uncorrelated Noise)**:
    - Q-Learning Base (45 states): **$-3,734.49$**
    - Q-Learning Base+Network (135 states): $-3,842.01$ ($-2.9\%$)
    - Q-Learning Base+Exec (135 states): $-4,273.87$ ($-14.4\%$)
    - Q-Learning Full (405 states): $-4,451.19$ ($-19.2\%$, statistically significant degradation)
    - *Finding*: When features contain uncorrelated noise, expanding the state space dilutes sample visits without providing predictive power—confirming the curse of dimensionality.
- **Visual / Layout Recommendation**:
  - Progression line graph showing reward progression from Base (45) $\to$ Base+Network (135) $\to$ Base+Exec (135) $\to$ Full (405) for both Synthetic (upward curve) and Dataset (downward curve).
- **Speaker Notes**:
  > "Slide 17 answers our first major research question: Does augmenting the state space improve cloud autoscaling? The answer is a decisive, nuanced conditional insight. On the synthetic workload where network traffic acts as a true leading indicator, expanding the state space from 45 to 405 states increased Q-Learning reward by +51.9%—from +4,242 to +6,446 with p < 0.001. Dyna-Q improved by +77.8%. The agent successfully exploited network traffic to scale up before CPU spikes occurred. Conversely, on the Kaggle dataset where network traffic is uncorrelated white noise, the 405-state space caused a 19.2% performance drop. Because the additional features contained no signal, splitting the state space 9-fold merely diluted training observations, proving that state-space augmentation is only beneficial when added features have genuine causal predictive power."

---

### Slide 18: Dyna-Q Planning Sensitivity Analysis (Suite C)
- **Slide Title**: Contribution 2 Analysis: Dyna-Q Planning Steps ($k \in \{0, 5, 10, 20, 50\}$)
- **Main Points**:
  - **Synthetic Workload Planning Dynamics**:
    - $k=0$ (Baseline Q-Learning inside Dyna): $+6,446.10 \pm 317.83$ | SLA: $13.7 \pm 2.9$
    - $k=5$: $+4,534.58 \pm 634.11$ | SLA: $20.5 \pm 12.8$
    - $k=10$: $+4,230.09 \pm 372.41$ | SLA: $16.1 \pm 8.3$
    - $k=20$: $+4,172.75 \pm 1145.43$ | SLA: $21.5 \pm 9.5$
    - $k=50$: $+2,344.84 \pm 1614.66$ | SLA: $41.2 \pm 24.9$ (Monotonic attenuation with planning intensity)
  - **Kaggle Dataset Planning Dynamics**:
    - $k=0$: $-4,451.19 \pm 556.98$ | SLA: $105.3 \pm 5.8$
    - $k=5$: $-5,257.10 \pm 802.71$ | SLA: $113.3 \pm 10.1$
    - $k=10$: $-6,046.40 \pm 392.34$ | SLA: $123.9 \pm 5.0$
    - $k=20$: $-6,027.65 \pm 632.92$ | SLA: $123.4 \pm 8.3$
    - $k=50$: $-5,223.54 \pm 263.23$ | SLA: $113.6 \pm 4.1$
  - **The "Over-Planning in Stochastic Environments" Discovery**:
    - In short-horizon RL (100 episodes), Dyna-Q excels because simulated updates accelerate initial table propagation.
    - However, at asymptotic long horizons (1,000 episodes / 1,000,000 total environment steps), Dyna-Q with $k \ge 10$ *underperforms* model-free Q-Learning ($+6,446.10$).
    - *Underlying Cause*: The standard Dyna-Q model is deterministic—it overwrites $\text{Model}(s, a)$ with the *latest* observed $(r, s')$. In a stochastic cloud environment with Gaussian workload noise, replaying the latest stochastic sample $k$ times over-propagates noise, leading to model compounding error.
- **Visual / Layout Recommendation**:
  - Curve showing Dyna-Q reward vs. Planning Steps $k \in [0, 50]$, showing peak performance at low $k$ and gradual attenuation as $k \to 50$.
- **Speaker Notes**:
  > "Slide 18 addresses our second major research question: Does model-based planning via Dyna-Q improve autoscaling performance? Here, our empirical findings reveal a critical insight that contradicts textbook intuition. While model-based planning provides immense sample efficiency in early episodes, across 1,000 full training episodes, Dyna-Q with plan=10 underperformed model-free Q-Learning (+4,230 vs +6,446). Furthermore, as planning steps increased from k=0 to k=50, asymptotic reward degraded. Why? Because the classic Dyna-Q model assumes a deterministic environment. In cloud autoscaling, workload demand is inherently stochastic. Over 1,000,000 simulated steps, repeatedly querying a single-sample deterministic model amplifies transient noise, compounding Q-table estimation errors. Model-based planning in cloud systems strictly requires distribution-aware or probabilistic transition models."

---

### Slide 19: Critical Discussion, Trade-Offs & Limitations
- **Slide Title**: Critical Discussion & Research Limitations
- **Main Points**:
  - **Summary of Trade-Offs**:
    - *Deep RL (DQN)*: Superior feature generalization on structured signals, but catastrophically fragile and vulnerable to divergence on noisy, non-stationary enterprise data.
    - *Tabular RL (Q-Learning / SARSA)*: Guaranteed convergence, highly robust on white-noise workloads, but constrained by discretization boundaries.
    - *Model-Based RL (Dyna-Q)*: High early sample efficiency, but degraded by deterministic model assumptions under stochastic cloud variance.
  - **Methodological Limitations**:
    - Discrete action steps $\{-1, 0, +1\}$ cannot execute multi-server bursts (e.g., $+3$ VMs) during sudden viral spikes.
    - VM provisioning delay (cold start overhead of 60–120 seconds) was abstracted into an immediate transition.
    - Tabular discretization tertiles require pre-computation or offline quantile calibration.
    - Computational constraints required postponing 45-state DQN baselines.
- **Visual / Layout Recommendation**:
  - Comparative trade-off matrix table scoring each algorithm (High/Medium/Low) on Sample Efficiency, Noise Robustness, Implementation Simplicity, and Wall-Clock Compute Time.
- **Speaker Notes**:
  > "Let us objectively discuss the trade-offs and limitations of our findings. No single algorithm is a silver bullet. Double DQN achieves remarkable performance when structured signals exist, but its experience replay buffer diverges when fed noisy enterprise data. Tabular Q-Learning and SARSA offer complete stability and convergence guarantees, but require careful discretization. Regarding Dyna-Q, its deterministic transition model is its primary bottleneck in noisy environments. Methodologically, our environment modeled single-step scaling actions and immediate provisioning, whereas production Kubernetes or EC2 clusters experience 60-to-90-second cold-start delays. Acknowledging these limitations provides clear directions for future research."

---

### Slide 20: Conclusions, Future Scope & Academic Attribution
- **Slide Title**: Conclusions & Future Research Directions
- **Main Points**:
  - **Conclusions**:
    1. *State Space Augmentation*: Adding predictive features (network ingress) yields statistically significant improvements ($+51.9\%$), but only when genuine cross-correlation exists. On uncorrelated workloads, state enlargement degrades performance by $19.2\%$.
    2. *Algorithm Hierarchy*: On structured workloads, $\text{DQN} > \text{Q-Learning} > \text{Dyna-Q} > \text{SARSA} \gg \text{Threshold}$. On empirical cloud noise, $\text{SARSA} \approx \text{Q-Learning} > \text{Dyna-Q} \gg \text{DQN} \gg \text{Threshold}$.
    3. *Planning Reality*: Deterministic Dyna-Q planning over-propagates noise over long horizons; small $k$ ($0 \le k \le 5$) outperforms large $k$.
  - **Future Scope**:
    - Probabilistic world models (Gaussian Process or World Models / Dreamer) for Dyna planning.
    - Continuous action spaces using Proximal Policy Optimization (PPO) or Soft Actor-Critic (SAC).
    - Multi-cloud hybrid autoscaling with heterogeneous VM instance types.
  - **Academic Attribution**:
    - Foundation built upon the open-source reference by Rah-ds (`Cloud-Autoscaling-using-RL`), with original Gymnasium environment refactoring, Dyna-Q architecture, Double DQN MLP, multi-seed statistical framework, and 5-feature state representation developed independently for this project.
- **Visual / Layout Recommendation**:
  - Summary takeaway box highlighting the key numerical findings, followed by research roadmap bullet points and attribution footer.
- **Speaker Notes**:
  > "In conclusion, our research provides an honest, empirically rigorous evaluation of reinforcement learning for cloud autoscaling. We have demonstrated that state-space enhancement is a double-edged sword: it unlocks massive gains when features provide leading signals, but exacerbates sample inefficiency when features are noisy. We proved that model-free tabular methods remain remarkably robust baselines for noisy real-world cloud environments, outperforming deep neural networks that overfit to noise. For future work, we recommend implementing probabilistic world models to resolve Dyna-Q's noise compounding, and extending to continuous actor-critic methods like PPO. Thank you for your time and attention. We now welcome questions from the committee."

---

