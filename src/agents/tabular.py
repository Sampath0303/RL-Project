"""Tabular reinforcement learning agents: QLearningAgent and SARSAAgent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class TabularAgent:
    """Base tabular agent with Q-table of shape (num_states, num_actions)."""

    def __init__(
        self,
        num_states: int = 405,
        num_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.RandomState(seed)

        # Q-table: (405, 3) initialized to zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def _to_index(self, state: Union[int, np.ndarray, Tuple[int, ...]]) -> int:
        """Resolve state representation to integer index in [0, num_states - 1]."""
        if isinstance(state, (int, np.integer)):
            return int(state)
        # MultiDiscrete state: (u, c, tr, net, ex)
        u, c, tr, net, ex = state
        return int(u * 135 + c * 27 + tr * 9 + net * 3 + ex)

    def select_action(
        self,
        state: Union[int, np.ndarray, Tuple[int, ...]],
        training: bool = True,
        info: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Select action using epsilon-greedy exploration if training, else greedy."""
        s_idx = self._to_index(state)

        if training and (self.rng.rand() < self.epsilon):
            return int(self.rng.randint(0, self.num_actions))

        # Greedy choice with random tie-breaking
        q_vals = self.q_table[s_idx]
        max_val = np.max(q_vals)
        best_actions = np.flatnonzero(q_vals == max_val)
        if len(best_actions) == 1:
            return int(best_actions[0])
        return int(self.rng.choice(best_actions))

    def decay_epsilon(self) -> None:
        """Decay exploration rate epsilon by factor epsilon_decay down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str | Path) -> None:
        """Save Q-table array to .npy file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, self.q_table)

    def load(self, filepath: str | Path) -> None:
        """Load Q-table array from .npy file."""
        self.q_table = np.load(filepath).astype(np.float32)


class QLearningAgent(TabularAgent):
    """Q-Learning agent (off-policy TD control).

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
        If s' is terminal: target is simply r.
    """

    def update(
        self,
        state: Union[int, np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[int, np.ndarray, Tuple[int, ...]],
        done: bool,
        next_action: Optional[int] = None,
    ) -> float:
        s_idx = self._to_index(state)
        current_q = self.q_table[s_idx, action]

        if done:
            target = float(reward)
        else:
            ns_idx = self._to_index(next_state)
            max_next_q = np.max(self.q_table[ns_idx])
            target = float(reward + self.gamma * max_next_q)

        td_error = target - current_q
        self.q_table[s_idx, action] += self.alpha * td_error
        return float(td_error)


class SARSAAgent(TabularAgent):
    """SARSA agent (on-policy TD control).

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]
        where a' is the action actually selected in next_state s'.
        If s' is terminal: target is simply r.
    """

    def update(
        self,
        state: Union[int, np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[int, np.ndarray, Tuple[int, ...]],
        done: bool,
        next_action: Optional[int] = None,
    ) -> float:
        s_idx = self._to_index(state)
        current_q = self.q_table[s_idx, action]

        if done:
            target = float(reward)
        else:
            ns_idx = self._to_index(next_state)
            if next_action is None:
                # If not explicitly provided, select action under current policy
                next_action = self.select_action(next_state, training=True)
            target = float(reward + self.gamma * self.q_table[ns_idx, next_action])

        td_error = target - current_q
        self.q_table[s_idx, action] += self.alpha * td_error
        return float(td_error)
