"""Deep Q-Network (DQN) agent with replay buffer and target network.

Supports:
- Normalized 5-feature vector input -> 3 Q-values
- Architecture: 5 -> 128 -> 128 -> 3 (ReLU)
- ReplayBuffer (up to 100k transitions)
- Soft target updates with Polyak averaging (tau = 1e-3)
- Standard DQN or Double DQN (DDQN)
- Adam optimizer (lr = 5e-4) and Huber (smooth L1) loss
- Epsilon-greedy exploration schedule
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    """Feedforward Multi-Layer Perceptron Q-Network."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""

    def __init__(self, capacity: int = 100000, seed: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent with optional Double-DQN."""

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 128),
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        double_dqn: bool = False,
        device: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.double_dqn = bool(double_dqn)

        # Device selection: CUDA if available else CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Seeding
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(None)

        # Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

        self.total_steps: int = 0

    def _normalize_state(self, state: Union[np.ndarray, Tuple[int, ...]]) -> np.ndarray:
        """Ensure state is a normalized float32 vector of shape (5,).

        If raw discrete state (u, c, tr, net, ex) is given:
        u: 0-2 -> /2.0
        c: 0-4 -> /4.0
        tr: 0-2 -> /2.0
        net: 0-2 -> /2.0
        ex: 0-2 -> /2.0
        """
        arr = np.array(state, dtype=np.float32)
        if arr.max() > 1.0 or arr.dtype in [np.int32, np.int64]:
            u, c, tr, net, ex = arr
            return np.array(
                [u / 2.0, c / 4.0, tr / 2.0, net / 2.0, ex / 2.0],
                dtype=np.float32,
            )
        return arr

    def select_action(
        self,
        state: Union[np.ndarray, Tuple[int, ...]],
        training: bool = True,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select action using epsilon-greedy exploration if training, else greedy."""
        if training and (self.rng.rand() < self.epsilon):
            return int(self.rng.randint(0, self.action_dim))

        norm_state = self._normalize_state(state)
        state_tensor = torch.tensor(norm_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = int(q_values.argmax(dim=1).item())

        return action

    def update(
        self,
        state: Union[np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[np.ndarray, Tuple[int, ...]],
        done: bool,
        next_action: Optional[int] = None,
    ) -> Optional[float]:
        """Push transition to buffer and perform a training step if buffer is ready."""
        norm_s = self._normalize_state(state)
        norm_ns = self._normalize_state(next_state)

        self.replay_buffer.push(norm_s, action, reward, norm_ns, done)
        self.total_steps += 1

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values: Q(s, a)
        current_q = self.q_network(states).gather(1, actions)

        # Target Q computation
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: action selected by online network, evaluated by target network
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN: max Q from target network
                next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]

            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        # Huber loss (Smooth L1)
        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Soft target network update
        self._soft_update()

        return float(loss.item())

    def _soft_update(self) -> None:
        """Polyak soft update: theta_target = tau * theta_local + (1 - tau) * theta_target."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def decay_epsilon(self) -> None:
        """Decay exploration factor epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str | Path) -> None:
        """Save Q-network weights and training state."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
            },
            filepath,
        )

    def load(self, filepath: str | Path) -> None:
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "epsilon" in checkpoint:
            self.epsilon = checkpoint["epsilon"]
        if "total_steps" in checkpoint:
            self.total_steps = checkpoint["total_steps"]
