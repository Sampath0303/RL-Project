"""Unit tests for Dyna-Q agent (model learning, planning, and value propagation)."""
import numpy as np
import pytest

from src.agents.dyna_q import DynaQAgent
from src.agents.tabular import QLearningAgent
from src.env import EnhancedAutoscalingEnv


def test_dyna_q_zero_planning_equals_q_learning():
    """Verify planning_steps=0 gives the exact same Q-table as QLearningAgent."""
    seed = 42
    dyna_zero = DynaQAgent(
        num_states=405,
        num_actions=3,
        alpha=0.1,
        gamma=0.99,
        planning_steps=0,
        seed=seed,
    )
    q_agent = QLearningAgent(
        num_states=405,
        num_actions=3,
        alpha=0.1,
        gamma=0.99,
        seed=seed,
    )

    transitions = [
        ((1, 2, 1, 0, 0), 2, 10.0, (1, 3, 1, 0, 0), False),
        ((1, 3, 1, 0, 0), 1, 5.0, (1, 3, 1, 0, 0), False),
        ((1, 3, 1, 0, 0), 0, -2.0, (1, 2, 0, 0, 0), False),
        ((1, 2, 0, 0, 0), 1, 15.0, (1, 2, 1, 0, 0), True),
    ]

    for s, a, r, ns, done in transitions:
        dyna_zero.update(s, a, r, ns, done)
        q_agent.update(s, a, r, ns, done)

    np.testing.assert_array_almost_equal(dyna_zero.q_table, q_agent.q_table)


def test_dyna_q_model_learning_and_visited_pairs():
    """Verify model accurately stores observed transitions and visited pairs list."""
    agent = DynaQAgent(num_states=10, num_actions=2, planning_steps=5, seed=42)
    s, a, r, ns, done = 3, 1, 12.5, 4, False

    agent.update(s, a, r, ns, done)

    # Check model contents
    assert agent.model_size == 1
    assert (3, 1) in agent.model
    assert agent.model[(3, 1)] == (12.5, 4, False)
    assert agent.visited_pairs == [(3, 1)]
    assert agent.total_planning_updates == 5


def test_dyna_q_planning_only_uses_visited_pairs():
    """Ensure planning steps sample exclusively from visited pairs."""
    agent = DynaQAgent(num_states=20, num_actions=3, planning_steps=10, seed=123)

    # Visit only (2, 0) and (5, 1)
    agent.update(2, 0, 1.0, 3, False)
    agent.update(5, 1, 2.0, 6, False)

    # Visited pairs must contain exactly those two
    assert set(agent.visited_pairs) == {(2, 0), (5, 1)}

    # Cells that were never visited and never transitioned into must stay 0
    # Check unrelated pair, e.g. state 15, action 2
    assert agent.q_table[15, 2] == 0.0


def test_dyna_q_value_propagation_on_tiny_mdp():
    """Verify that more planning steps propagate values faster across a chain MDP.

    MDP: State 0 -> State 1 -> State 2 (Terminal, Reward = +100).
    Actions deterministically step right: action 0 moves s -> s+1.
    """
    # Compare 0 planning steps vs 20 planning steps
    agent_no_plan = DynaQAgent(
        num_states=3,
        num_actions=1,
        alpha=0.5,
        gamma=0.9,
        planning_steps=0,
        seed=42,
    )
    agent_with_plan = DynaQAgent(
        num_states=3,
        num_actions=1,
        alpha=0.5,
        gamma=0.9,
        planning_steps=20,
        seed=42,
    )

    # Step 1: 0 -> 1 (reward 0)
    agent_no_plan.update(0, 0, 0.0, 1, False)
    agent_with_plan.update(0, 0, 0.0, 1, False)

    # Step 2: 1 -> 2 (reward 100, terminal)
    agent_no_plan.update(1, 0, 100.0, 2, True)
    agent_with_plan.update(1, 0, 100.0, 2, True)

    # In 0-planning agent, State 0 has NOT yet received any reward signal (Q(0,0) == 0)
    assert agent_no_plan.q_table[0, 0] == 0.0

    # In 20-planning agent, background simulated updates will have propagated reward to State 0
    assert agent_with_plan.q_table[0, 0] > 0.0
    print(f"\nValue propagated to State 0: Q(0,0) = {agent_with_plan.q_table[0, 0]:.4f}")


def test_dyna_q_stochastic_model():
    """Verify stochastic model option retains multiple transitions for noisy environments."""
    agent = DynaQAgent(
        num_states=5,
        num_actions=2,
        planning_steps=2,
        stochastic_model=True,
        seed=42,
    )
    agent.update(0, 0, 1.0, 1, False)
    agent.update(0, 0, 2.0, 2, False)

    assert len(agent.model[(0, 0)]) == 2
    assert agent.model[(0, 0)][0] == (1.0, 1, False)
    assert agent.model[(0, 0)][1] == (2.0, 2, False)

