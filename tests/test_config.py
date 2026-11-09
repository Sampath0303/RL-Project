"""Tests for enhanced_autoscaling configuration and setup."""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import load_config

def test_load_default_config():
    """Verify that default.yaml loads correctly and has all required keys."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    
    # Check top-level keys
    assert cfg["data_mode"] in ["dataset", "synthetic"]
    assert "dataset_path" in cfg
    assert cfg["episodes"] == 1000
    assert cfg["episode_length"] == 1000
    assert cfg["seeds"] == [0, 1, 2, 3, 4]
    assert cfg["max_capacity"] == 5
    assert cfg["start_capacity"] == 3
    assert cfg["capacity_unit"] == 20
    assert cfg["feature_set"] in ["base", "base+network", "base+exec", "full"]

    # Check reward weights
    assert "reward" in cfg
    assert "sla_violation_penalty" in cfg["reward"]
    assert "underutilization_penalty" in cfg["reward"]
    assert "scaling_cost" in cfg["reward"]
    assert "normal_operating_bonus" in cfg["reward"]

    # Check thresholds
    assert len(cfg["network_thresholds"]) == 2
    assert len(cfg["execution_time_thresholds"]) == 2

    # Check hyperparameters
    assert "hyperparameters" in cfg
    hyper = cfg["hyperparameters"]

    # Tabular
    assert hyper["tabular"]["alpha"] == 0.1
    assert hyper["tabular"]["gamma"] == 0.99
    assert hyper["tabular"]["epsilon_start"] == 1.0
    assert hyper["tabular"]["epsilon_min"] == 0.01
    assert hyper["tabular"]["epsilon_decay"] == 0.995

    # Dyna-Q
    assert hyper["dyna_q"]["planning_steps"] == 10

    # DQN
    assert hyper["dqn"]["learning_rate"] == 0.0005
    assert hyper["dqn"]["batch_size"] == 64
    assert hyper["dqn"]["buffer_size"] == 100000
    assert hyper["dqn"]["hidden_dims"] == [128, 128]
    assert hyper["dqn"]["tau"] == 0.001

