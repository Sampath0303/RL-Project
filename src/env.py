"""Enhanced Gymnasium environment for cloud autoscaling."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.data import (
    CloudDataset,
    load_and_clean_data,
    compute_terciles,
    discretize_feature,
)
from src.workload import (
    generate_demand,
    generate_network_traffic,
    exec_time,
)


class EnhancedAutoscalingEnv(gym.Env):
    """Enhanced Cloud Autoscaling Gymnasium Environment.

    Supports two workload modes:
    - 'dataset': Exogenous demand, network_level, and execution_level sampled from cleaned Kaggle dataset.
    - 'synthetic': Demand generated with diurnal cycles/spikes, network traffic as a leading indicator,
                   and execution time dynamically coupled to utilization (endogenous).

    Observation Space:
        MultiDiscrete([3, 5, 3, 3, 3]) = 405 states
        - util_level (0: <40%, 1: 40-80%, 2: >=80%)
        - capacity_level (0: 1 server ... 4: 5 servers)
        - demand_trend (0: falling (<-2), 1: stable [-2, 2], 2: rising (>+2))
        - network_level (0: low, 1: medium, 2: high)
        - exec_level (0: low, 1: medium, 2: high)

    Action Space:
        Discrete(3): 0 = scale down (-1), 1 = hold (0), 2 = scale up (+1)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_mode: Optional[str] = None,
        dataset: Optional[CloudDataset] = None,
        feature_set: Optional[str] = None,
        split: str = "train",
    ):
        super().__init__()
        self.config = config or {}

        # Environment parameters
        self.data_mode = data_mode or self.config.get("data_mode", "synthetic")
        self.feature_set = feature_set or self.config.get("feature_set", "full")
        self.split = split

        self.min_capacity = 1
        self.max_capacity = int(self.config.get("max_capacity", 5))
        self.start_capacity = int(self.config.get("start_capacity", 3))
        self.capacity_unit = float(self.config.get("capacity_unit", 20.0))
        self.episode_length = int(self.config.get("episode_length", 1000))

        # Reward weights & SLA criteria
        self.sla_latency = float(self.config.get("sla_latency", 100.0))

        # Discretization thresholds
        self.network_thresholds = tuple(
            self.config.get("network_thresholds", [333.33, 666.67])
        )
        self.execution_time_thresholds = tuple(
            self.config.get("execution_time_thresholds", [33.33, 66.67])
        )

        # Dataset mode support
        self.dataset = dataset
        if self.data_mode == "dataset" and self.dataset is None:
            raw_path = self.config.get(
                "dataset_path", "data/raw/vmCloud_data.csv"
            )
            df = load_and_clean_data(raw_path)
            self.dataset = CloudDataset(
                df,
                train_ratio=0.8,
                network_thresholds=self.network_thresholds,
                exec_thresholds=self.execution_time_thresholds,
            )

        # Action space: 0 down, 1 hold, 2 up
        self.action_space = spaces.Discrete(3)

        # Observation space: 5 discrete dimensions
        # util_level: 3, capacity_level: 5, trend: 3, network_level: 3, exec_level: 3
        self.observation_space = spaces.MultiDiscrete([3, 5, 3, 3, 3])

        # Internal state tracking
        self.current_step = 0
        self.capacity = self.start_capacity
        self.prev_demand = 0.0
        self.demand_series = np.array([], dtype=np.float32)
        self.network_raw_series = np.array([], dtype=np.float32)
        self.exec_raw_series = np.array([], dtype=np.float32)
        self.network_level_series = np.array([], dtype=np.int32)
        self.exec_level_series = np.array([], dtype=np.int32)

        self.rng = np.random.RandomState(None)

    def _get_trend_level(self, diff: float) -> int:
        """Categorize demand change into falling (0), stable (1), rising (2)."""
        if diff < -2.0:
            return 0
        elif diff > 2.0:
            return 2
        return 1

    def _get_util_level(self, util: float) -> int:
        """Categorize utilization into low (0: <40%), medium (1: 40-80%), high (2: >=80%)."""
        if util < 0.40:
            return 0
        elif util < 0.80:
            return 1
        return 2

    def _get_exec_level(self, latency: float) -> int:
        """Discretize execution time for synthetic mode based on execution_time_thresholds."""
        q1, q2 = self.execution_time_thresholds
        if latency < q1:
            return 0
        elif latency < q2:
            return 1
        return 2

    def _apply_feature_mask(self, raw_state: Tuple[int, int, int, int, int]) -> Tuple[int, int, int, int, int]:
        """Apply feature_set ablations while keeping the state tuple shape (5,) intact.

        - 'full': all features active
        - 'base': network_level = 0, exec_level = 0
        - 'base+network': network_level active, exec_level = 0
        - 'base+exec': network_level = 0, exec_level active
        """
        u, c, tr, net, ex = raw_state
        if self.feature_set == "base":
            return (u, c, tr, 0, 0)
        elif self.feature_set == "base+network":
            return (u, c, tr, net, 0)
        elif self.feature_set == "base+exec":
            return (u, c, tr, 0, ex)
        return (u, c, tr, net, ex)

    def state_to_index(self, state: Tuple[int, int, int, int, int] | np.ndarray) -> int:
        """Convert 5-dimensional state into unique integer index in [0, 404].

        Multipliers: [135, 27, 9, 3, 1] for dimensions [3, 5, 3, 3, 3].
        """
        u, c, tr, net, ex = state
        return int(u * 135 + c * 27 + tr * 9 + net * 3 + ex)

    def index_to_state(self, index: int) -> Tuple[int, int, int, int, int]:
        """Convert single integer index back into 5-dimensional state tuple."""
        u = index // 135
        rem = index % 135
        c = rem // 27
        rem = rem % 27
        tr = rem // 9
        rem = rem % 9
        net = rem // 3
        ex = rem % 3
        return (int(u), int(c), int(tr), int(net), int(ex))

    def get_state_vector(self, state: Tuple[int, int, int, int, int] | np.ndarray) -> np.ndarray:
        """Convert discrete state tuple into a normalized float feature vector for DQN.

        Normalized to range [0.0, 1.0].
        """
        u, c, tr, net, ex = state
        return np.array(
            [
                u / 2.0,
                c / 4.0,
                tr / 2.0,
                net / 2.0,
                ex / 2.0,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.rng = np.random.RandomState(seed)

        self.current_step = 0
        self.capacity = self.start_capacity

        # Initialize workload series for the episode
        if self.data_mode == "dataset":
            if self.dataset is None:
                raise RuntimeError("Dataset not loaded in dataset mode.")
            window = self.dataset.get_episode_window(
                split=self.split,
                episode_length=self.episode_length,
                seed=seed,
            )
            self.demand_series = window["demand"]
            self.network_raw_series = window["network_raw"]
            self.exec_raw_series = window["execution_raw"]
            self.network_level_series = window["network_level"]
            self.exec_level_series = window["execution_level"]
        else:  # synthetic
            # Generate structured demand
            demand_seed = self.rng.randint(0, 1_000_000)
            net_seed = self.rng.randint(0, 1_000_000)
            self.demand_series = generate_demand(self.episode_length, seed=demand_seed)
            self.network_raw_series = generate_network_traffic(
                self.demand_series, seed=net_seed, lead=1
            )
            self.network_level_series = discretize_feature(
                self.network_raw_series, self.network_thresholds
            )
            self.exec_raw_series = np.zeros(self.episode_length, dtype=np.float32)
            self.exec_level_series = np.zeros(self.episode_length, dtype=np.int32)

        # Initial demand and metrics
        cur_demand = float(self.demand_series[0])
        self.prev_demand = cur_demand

        utilization = cur_demand / (self.capacity * self.capacity_unit)
        util_level = self._get_util_level(utilization)
        capacity_level = self.capacity - self.min_capacity  # 0 to 4
        trend = 1  # stable at initial step

        if self.data_mode == "dataset":
            net_level = int(self.network_level_series[0])
            ex_level = int(self.exec_level_series[0])
            exec_duration = float(self.exec_raw_series[0])
            net_traffic = float(self.network_raw_series[0])
        else:
            net_level = int(self.network_level_series[0])
            exec_seed = self.rng.randint(0, 1_000_000)
            exec_duration = float(exec_time(utilization, seed=exec_seed))
            self.exec_raw_series[0] = exec_duration
            ex_level = self._get_exec_level(exec_duration)
            self.exec_level_series[0] = ex_level
            net_traffic = float(self.network_raw_series[0])

        raw_state = (util_level, capacity_level, trend, net_level, ex_level)
        masked_state = np.array(self._apply_feature_mask(raw_state), dtype=np.int64)

        info = {
            "utilization": utilization,
            "demand": cur_demand,
            "capacity": self.capacity,
            "execution_time": exec_duration,
            "network_traffic": net_traffic,
            "sla_violation": 0,
            "total_cost": 0.5 * self.capacity,
            "capacity_changes": 0,
            "state_index": self.state_to_index(masked_state),
        }

        return masked_state, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        # 1. Update capacity based on action: 0 = down (-1), 1 = hold (0), 2 = up (+1)
        prev_capacity = self.capacity
        if action == 0:
            self.capacity = max(self.min_capacity, self.capacity - 1)
        elif action == 2:
            self.capacity = min(self.max_capacity, self.capacity + 1)
        # action == 1 -> hold

        capacity_changed = 1 if self.capacity != prev_capacity else 0

        # Advance step
        self.current_step += 1
        terminated = self.current_step >= (self.episode_length - 1)
        truncated = False

        # 2. Get current demand and step metrics
        cur_demand = float(self.demand_series[self.current_step])
        demand_diff = cur_demand - self.prev_demand
        trend = self._get_trend_level(demand_diff)
        self.prev_demand = cur_demand

        # 3. Compute utilization
        total_capacity_units = self.capacity * self.capacity_unit
        utilization = cur_demand / total_capacity_units
        util_level = self._get_util_level(utilization)
        capacity_level = self.capacity - self.min_capacity

        # 4. Exogenous/Endogenous metrics
        if self.data_mode == "dataset":
            net_level = int(self.network_level_series[self.current_step])
            ex_level = int(self.exec_level_series[self.current_step])
            exec_duration = float(self.exec_raw_series[self.current_step])
            net_traffic = float(self.network_raw_series[self.current_step])
        else:  # synthetic: execution time is dependent on current utilization
            net_level = int(self.network_level_series[self.current_step])
            exec_seed = self.rng.randint(0, 1_000_000)
            exec_duration = float(exec_time(utilization, seed=exec_seed))
            self.exec_raw_series[self.current_step] = exec_duration
            ex_level = self._get_exec_level(exec_duration)
            self.exec_level_series[self.current_step] = ex_level
            net_traffic = float(self.network_raw_series[self.current_step])

        # 5. Compute Reward
        reward = 0.0

        # Normal operating reward (+10 if 40% <= util <= 80%; +5 more if 60% <= util <= 70%)
        if 0.40 <= utilization <= 0.80:
            reward += 10.0
            if 0.60 <= utilization <= 0.70:
                reward += 5.0

        # SLA violation: -50 * (1 + overshoot) when util >= 0.90
        # In synthetic mode: also when execution time > sla_latency
        sla_violation = 0
        if utilization >= 0.90:
            overshoot = utilization - 0.90
            reward -= 50.0 * (1.0 + overshoot)
            sla_violation = 1

        if self.data_mode == "synthetic":
            if exec_duration > self.sla_latency and sla_violation == 0:
                reward -= 50.0
                sla_violation = 1

        # Waste penalty: -5 * (0.2 - util) if util < 0.2
        if utilization < 0.20:
            reward -= 5.0 * (0.20 - utilization)

        # Churn penalty: -2 if action != hold (action != 1)
        if action != 1:
            reward -= 2.0

        # Resource cost: -0.5 * capacity
        step_cost = 0.5 * self.capacity
        reward -= step_cost

        # 6. Form observation
        raw_state = (util_level, capacity_level, trend, net_level, ex_level)
        masked_state = np.array(self._apply_feature_mask(raw_state), dtype=np.int64)

        info = {
            "utilization": utilization,
            "demand": cur_demand,
            "capacity": self.capacity,
            "execution_time": exec_duration,
            "network_traffic": net_traffic,
            "sla_violation": sla_violation,
            "total_cost": step_cost,
            "capacity_changes": capacity_changed,
            "state_index": self.state_to_index(masked_state),
        }

        return masked_state, float(reward), terminated, truncated, info
