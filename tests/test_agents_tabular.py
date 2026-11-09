"""Unit tests for baseline policies and tabular RL agents."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.agents.baselines import RandomPolicy, ThresholdPolicy
from src.agents.tabular import QLearningAgent, SARSAAgent
from src.env import EnhancedAutoscalingEnv


def test_baseline_policies():
    """Verify RandomPolicy and ThresholdPolicy action selections."""
    # RandomPolicy
    rand_policy = RandomPolicy(seed=42)
    actions = [rand_policy.select_action(np.array([1, 2, 1, 0, 0])) for _ in range(50)]
    assert set(actions).issubset({0, 1, 2})
    assert len(set(actions)) == 3  # All actions sampled

    # ThresholdPolicy
    thresh_policy = ThresholdPolicy()
    # High util (state[0] == 2) -> scale up (2)
    assert thresh_policy.select_action((2, 2, 1, 0, 0)) == 2
    # Low util (state[0] == 0) -> scale down (0)
    assert thresh_policy.select_action((0, 2, 1, 0, 0)) == 0
    # Medium util (state[0] == 1) -> hold (1)
    assert thresh_policy.select_action((1, 2, 1, 0, 0)) == 1


def test_tabular_agent_shape_and_initialization():
    """Verify Q-table shape (405, 3) and initial values."""
    agent = QLearningAgent(num_states=405, num_actions=3)
    assert agent.q_table.shape == (405, 3)
    assert np.all(agent.q_table == 0.0)


def test_tabular_only_visited_cell_changes():
    """Verify that only Q(s, a) updates while all other Q-table cells remain untouched."""
    agent = QLearningAgent(num_states=405, num_actions=3, alpha=0.1, gamma=0.9)
    state = (1, 2, 1, 0, 0)
    s_idx = agent._to_index(state)
    action = 2
    reward = 10.0
    next_state = (1, 3, 1, 0, 0)

    agent.update(state, action, reward, next_state, done=False)

    # Q(s, a) should be updated from 0 to alpha * r = 0.1 * 10 = 1.0
    assert pytest.approx(agent.q_table[s_idx, action], 0.01) == 1.0

    # Ensure all other entries in Q-table are still exactly 0
    copy_table = agent.q_table.copy()
    copy_table[s_idx, action] = 0.0
    assert np.all(copy_table == 0.0)


def test_tabular_epsilon_decay_and_greedy_evaluation():
    """Verify epsilon decay rate, minimum floor, and purely greedy behavior when training=False."""
    agent = QLearningAgent(
        num_states=405,
        num_actions=3,
        epsilon=1.0,
        epsilon_decay=0.9,
        epsilon_min=0.05,
        seed=42,
    )
    # Decay several times
    agent.decay_epsilon()
    assert pytest.approx(agent.epsilon, 0.001) == 0.9
    for _ in range(50):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05

    # Set specific Q-values for a state
    state = (0, 0, 0, 0, 0)
    s_idx = agent._to_index(state)
    agent.q_table[s_idx] = [1.5, 9.0, 3.2]

    # When training=False, action must be strictly greedy (action 1)
    for _ in range(20):
        act = agent.select_action(state, training=False)
        assert act == 1


def test_tabular_terminal_state_handling():
    """Verify that when done=True, TD target does not include discounted bootstrap value."""
    agent = QLearningAgent(num_states=405, num_actions=3, alpha=0.5, gamma=0.99)
    state = (1, 1, 1, 1, 1)
    s_idx = agent._to_index(state)
    next_state = (2, 2, 2, 2, 2)
    ns_idx = agent._to_index(next_state)

    # Put a large Q value in next_state
    agent.q_table[ns_idx] = [100.0, 100.0, 100.0]

    # Update with done=True
    reward = -10.0
    agent.update(state, action=0, reward=reward, next_state=next_state, done=True)

    # Target should be just reward (-10.0), not reward + gamma * 100
    # Q = 0 + 0.5 * (-10.0 - 0) = -5.0
    assert pytest.approx(agent.q_table[s_idx, 0], 0.01) == -5.0


def test_sarsa_update_rule():
    """Verify SARSA on-policy update using next_action."""
    agent = SARSAAgent(num_states=405, num_actions=3, alpha=0.5, gamma=0.9)
    state = (1, 1, 1, 0, 0)
    s_idx = agent._to_index(state)
    next_state = (1, 2, 1, 0, 0)
    ns_idx = agent._to_index(next_state)

    # Put distinct values in next state
    agent.q_table[ns_idx] = [2.0, 10.0, 4.0]

    # Action 0 chosen in state, next_action 1 chosen in next_state
    agent.update(
        state=state,
        action=0,
        reward=5.0,
        next_state=next_state,
        done=False,
        next_action=1,  # value is 10.0
    )
    # Target = 5.0 + 0.9 * 10.0 = 14.0
    # Q(s, 0) = 0 + 0.5 * (14.0 - 0) = 7.0
    assert pytest.approx(agent.q_table[s_idx, 0], 0.01) == 7.0


def test_tabular_save_and_load(tmp_path):
    """Verify Q-table persistence to .npy file."""
    agent = QLearningAgent(num_states=405, num_actions=3)
    agent.q_table[10, 1] = 42.5
    save_file = tmp_path / "q_table.npy"

    agent.save(save_file)
    assert save_file.exists()

    agent2 = QLearningAgent(num_states=405, num_actions=3)
    agent2.load(save_file)
    assert agent2.q_table[10, 1] == 42.5
    np.testing.assert_array_equal(agent.q_table, agent2.q_table)

