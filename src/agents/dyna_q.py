"""Dyna-Q reinforcement learning agent (model-based planning).

Main contribution for enhanced-state cloud autoscaling.
Combines real experience Q-learning with a learned transition/reward model
and background planning replay steps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.agents.tabular import TabularAgent


class DynaQAgent(TabularAgent):
    """Dyna-Q Agent (Sutton & Barto Chapter 8).

    Components:
    1. Direct RL: standard Q-learning TD update on real transition (s, a, r, s', done).
    2. Model Learning: model[(s, a)] = (r, s', done).
       Supports either deterministic latest observation or sample-based history.
    3. Planning: repeats `planning_steps` times:
       - Randomly sample previously visited state-action pair (s, a)
       - Query learned model: (r, s', done) <- model[(s, a)]
       - Perform simulated Q-learning TD update on (s, a, r, s', done).
    """

    def __init__(
        self,
        num_states: int = 405,
        num_actions: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        planning_steps: int = 10,
        stochastic_model: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__(
            num_states=num_states,
            num_actions=num_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
        self.planning_steps = int(planning_steps)
        self.stochastic_model = bool(stochastic_model)

        # Learned model: (s, a) -> list of (r, s', done) or single (r, s', done)
        # Stores indices for fast table indexing
        self.model: Dict[Tuple[int, int], Any] = {}
        # Keep track of observed (s, a) pairs for fast random sampling
        self.visited_pairs: List[Tuple[int, int]] = []

        # Diagnostics & Tracking
        self.total_planning_updates: int = 0

    @property
    def model_size(self) -> int:
        """Return number of unique state-action transitions learned in the model."""
        return len(self.model)

    def _apply_q_update(
        self,
        s_idx: int,
        action: int,
        reward: float,
        ns_idx: int,
        done: bool,
    ) -> float:
        """Perform a single standard Q-learning TD update."""
        current_q = self.q_table[s_idx, action]
        if done:
            target = float(reward)
        else:
            max_next_q = np.max(self.q_table[ns_idx])
            target = float(reward + self.gamma * max_next_q)

        td_error = target - current_q
        self.q_table[s_idx, action] += self.alpha * td_error
        return float(td_error)

    def update(
        self,
        state: Union[int, np.ndarray, Tuple[int, ...]],
        action: int,
        reward: float,
        next_state: Union[int, np.ndarray, Tuple[int, ...]],
        done: bool,
        next_action: Optional[int] = None,
    ) -> float:
        """1. Real experience update, 2. Model learning, 3. Planning steps."""
        s_idx = self._to_index(state)
        ns_idx = self._to_index(next_state)
        action = int(action)

        # 1. Real Q-learning update
        real_td_error = self._apply_q_update(s_idx, action, reward, ns_idx, done)

        # 2. Model learning
        pair = (s_idx, action)
        if pair not in self.model:
            self.visited_pairs.append(pair)

        if self.stochastic_model:
            if pair not in self.model:
                self.model[pair] = []
            # Keep small ring buffer of recent transitions (e.g. up to 10)
            self.model[pair].append((float(reward), ns_idx, bool(done)))
            if len(self.model[pair]) > 10:
                self.model[pair].pop(0)
        else:
            # Latest observation model
            self.model[pair] = (float(reward), ns_idx, bool(done))

        # 3. Planning (simulated experience updates)
        if self.planning_steps > 0 and len(self.visited_pairs) > 0:
            num_pairs = len(self.visited_pairs)
            for _ in range(self.planning_steps):
                # Uniformly sample previously visited (s, a)
                sample_idx = self.rng.randint(0, num_pairs)
                sim_s, sim_a = self.visited_pairs[sample_idx]

                if self.stochastic_model:
                    transitions = self.model[(sim_s, sim_a)]
                    t_idx = self.rng.randint(0, len(transitions))
                    sim_r, sim_ns, sim_done = transitions[t_idx]
                else:
                    sim_r, sim_ns, sim_done = self.model[(sim_s, sim_a)]

                # Q-learning update on simulated step
                self._apply_q_update(sim_s, sim_a, sim_r, sim_ns, sim_done)
                self.total_planning_updates += 1

        return real_td_error

    def reset_model(self) -> None:
        """Clear learned model and visited pairs history."""
        self.model.clear()
        self.visited_pairs.clear()
        self.total_planning_updates = 0
