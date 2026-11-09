"""Unit tests for DQNAgent (network architecture, replay buffer, target soft update, loss minimization)."""
import torch
import numpy as np
import pytest

from src.agents.dqn import DQNAgent, ReplayBuffer, QNetwork


def test_qnetwork_architecture_and_output_shape():
    """Verify QNetwork output shape (batch_size, 3) and forward pass."""
    net = QNetwork(state_dim=5, action_dim=3, hidden_dims=(128, 128))
    dummy_input = torch.randn(8, 5)
    output = net(dummy_input)

    assert output.shape == (8, 3)
    assert not torch.isnan(output).any()


def test_replay_buffer_push_and_sample():
    """Verify experience replay buffer sampling types and shapes."""
    buf = ReplayBuffer(capacity=100, seed=42)
    for i in range(10):
        s = np.zeros(5, dtype=np.float32) + i
        ns = np.zeros(5, dtype=np.float32) + i + 1
        buf.push(s, action=i % 3, reward=float(i), next_state=ns, done=(i == 9))

    assert len(buf) == 10
    states, actions, rewards, next_states, dones = buf.sample(4)

    assert states.shape == (4, 5)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 5)
    assert dones.shape == (4, 1)


def test_target_network_soft_update():
    """Verify that target network parameters move towards online network parameters via tau."""
    agent = DQNAgent(
        state_dim=5,
        action_dim=3,
        tau=0.1,  # 10% update
        seed=42,
        device="cpu",
    )

    # Initial target and local network weights are identical
    for target_p, local_p in zip(agent.target_network.parameters(), agent.q_network.parameters()):
        assert torch.all(target_p == local_p)

    # Modify online network weights
    with torch.no_grad():
        for p in agent.q_network.parameters():
            p.add_(1.0)

    # Trigger soft update
    agent._soft_update()

    # Target parameters should have increased by ~tau * 1.0 = 0.1
    for target_p, local_p in zip(agent.target_network.parameters(), agent.q_network.parameters()):
        diff = local_p - target_p
        # diff should be around (1 - tau) * 1.0 = 0.9
        assert torch.allclose(diff, torch.full_like(diff, 0.9), atol=1e-4)


def test_dqn_loss_finite_and_decreases_on_fixed_batch():
    """Verify that training on a fixed batch produces finite loss that decreases over steps."""
    agent = DQNAgent(
        state_dim=5,
        action_dim=3,
        learning_rate=1e-3,
        batch_size=16,
        seed=42,
        device="cpu",
    )

    # Populate buffer with 16 identical or fixed transitions
    for _ in range(16):
        agent.replay_buffer.push(
            state=np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            action=1,
            reward=10.0,
            next_state=np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            done=False,
        )

    # Run multiple updates on this batch
    losses = []
    for _ in range(30):
        # We manually call update with dummy transition to trigger batch optimization
        loss = agent.update(
            state=np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            action=1,
            reward=10.0,
            next_state=np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            done=False,
        )
        if loss is not None:
            losses.append(loss)

    assert len(losses) > 0
    assert all(np.isfinite(l) for l in losses)
    # Final loss should be lower than initial loss on this consistent task
    assert losses[-1] < losses[0]


def test_double_dqn_flag():
    """Verify Double DQN runs without error."""
    agent = DQNAgent(
        state_dim=5,
        action_dim=3,
        batch_size=4,
        double_dqn=True,
        seed=42,
        device="cpu",
    )
    for _ in range(10):
        agent.update(
            state=(1, 2, 1, 0, 0),
            action=0,
            reward=1.0,
            next_state=(1, 1, 1, 0, 0),
            done=False,
        )
    assert len(agent.replay_buffer) == 10

