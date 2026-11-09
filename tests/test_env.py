"""Unit tests for synthetic workload and enhanced autoscaling environment."""
import numpy as np
import pytest

from src.workload import (
    generate_demand,
    generate_network_traffic,
    exec_time,
)
from src.env import EnhancedAutoscalingEnv
from src.data import CloudDataset
import pandas as pd


def test_workload_generate_demand_properties():
    """Verify synthetic demand values are bounded [10, 100] and reproducible with seeds."""
    length = 500
    d1 = generate_demand(length, seed=42)
    d2 = generate_demand(length, seed=42)
    d3 = generate_demand(length, seed=99)

    assert len(d1) == length
    assert np.all(d1 >= 10.0)
    assert np.all(d1 <= 100.0)
    np.testing.assert_array_equal(d1, d2)
    assert not np.array_equal(d1, d3)


def test_workload_network_leads_demand():
    """Verify that network traffic acts as a leading indicator for demand."""
    length = 200
    # Clean step function demand to clearly observe lead
    demand = np.full(length, 20.0, dtype=np.float32)
    demand[50:] = 80.0

    net = generate_network_traffic(demand, seed=42, lead=1, scale=10.0, noise_std=0.0)
    # At step 49, demand is 20, but next demand (step 50) is 80 -> network should reflect 800
    assert net[49] == 800.0
    assert net[0] == 200.0
    # Last step should safely hold the last demand value
    assert net[-1] == 800.0


def test_workload_exec_time_increases_with_utilization():
    """Verify queueing delay formula: execution time monotonically increases with utilization."""
    low_lat = exec_time(0.20, seed=42, noise_std=0.0)
    med_lat = exec_time(0.60, seed=42, noise_std=0.0)
    high_lat = exec_time(0.90, seed=42, noise_std=0.0)
    assert low_lat < med_lat < high_lat


def test_env_observation_space_and_indexing():
    """Verify state indexing bijection and vector conversion."""
    env = EnhancedAutoscalingEnv({"data_mode": "synthetic", "episode_length": 50})
    state, info = env.reset(seed=123)

    assert state.shape == (5,)
    assert env.observation_space.contains(state)
    assert 0 <= info["state_index"] < 405

    # Check bijective index conversions
    idx = env.state_to_index(state)
    reconstructed_state = env.index_to_state(idx)
    assert tuple(state) == reconstructed_state

    # Check normalized vector
    vec = env.get_state_vector(state)
    assert vec.shape == (5,)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_env_capacity_bounds():
    """Verify capacity strictly clips at [1, 5]."""
    env = EnhancedAutoscalingEnv({"data_mode": "synthetic", "start_capacity": 3})
    env.reset(seed=42)

    # Scale down repeatedly
    for _ in range(5):
        _, _, _, _, info = env.step(0)  # down
    assert info["capacity"] == 1
    assert env.capacity == 1

    # Scale up repeatedly
    for _ in range(10):
        _, _, _, _, info = env.step(2)  # up
    assert info["capacity"] == 5
    assert env.capacity == 5


def test_env_reward_rules():
    """Verify exact reward computation cases."""
    env = EnhancedAutoscalingEnv({"data_mode": "synthetic", "episode_length": 100})
    env.reset(seed=42)

    # 1. Normal optimal operating condition: util in [0.60, 0.70] with hold (action 1)
    # capacity = 3 (units = 60). If demand = 39 -> util = 0.65
    env.capacity = 3
    env.demand_series = np.array([39.0, 39.0, 39.0], dtype=np.float32)
    env.current_step = 0
    _, reward, _, _, info = env.step(1)  # hold
    # Expected: +10 (normal) +5 (optimal 60-70%) - 0 (no SLA) - 0 (no churn) - 0.5*3 (cost 1.5) = +13.5
    assert pytest.approx(reward, 0.01) == 13.5
    assert info["sla_violation"] == 0

    # 2. SLA violation condition: util >= 0.90
    # capacity = 1 (units = 20), demand = 20 -> util = 1.0 (overshoot = 0.10)
    env.capacity = 1
    env.demand_series = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    env.current_step = 0
    _, reward, _, _, info = env.step(1)  # hold
    # Expected: SLA penalty: -50 * (1 + 0.10) = -55.0; cost = -0.5*1 = -0.5 -> total -55.5
    assert pytest.approx(reward, 0.01) == -55.5
    assert info["sla_violation"] == 1


def test_env_feature_set_ablations():
    """Verify feature masking sets disabled features to 0."""
    for feat_set, expected_mask in [
        ("base", (True, True, True, False, False)),
        ("base+network", (True, True, True, True, False)),
        ("base+exec", (True, True, True, False, True)),
        ("full", (True, True, True, True, True)),
    ]:
        env = EnhancedAutoscalingEnv({
            "data_mode": "synthetic",
            "feature_set": feat_set,
            "episode_length": 10,
        })
        state, _ = env.reset(seed=42)
        assert len(state) == 5
        if not expected_mask[3]:
            assert state[3] == 0
        if not expected_mask[4]:
            assert state[4] == 0


def test_dataset_mode_env():
    """Verify dataset mode works seamlessly with the real cached dataset or mock dataset."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=200, freq="min"),
            "cpu_usage": np.linspace(15, 95, 200, dtype=np.float32),
            "network_traffic": np.linspace(100, 900, 200, dtype=np.float32),
            "execution_time": np.linspace(10, 80, 200, dtype=np.float32),
            "memory_usage": np.full(200, 50.0, dtype=np.float32),
        }
    )
    dataset = CloudDataset(df, train_ratio=0.8)
    env = EnhancedAutoscalingEnv(
        config={"episode_length": 20},
        data_mode="dataset",
        dataset=dataset,
    )
    state, info = env.reset(seed=42)
    assert state.shape == (5,)
    assert "utilization" in info
    next_state, reward, terminated, _, info = env.step(1)
    assert next_state.shape == (5,)
    assert isinstance(reward, float)

