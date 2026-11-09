"""Baseline policies for cloud autoscaling."""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


class BaselinePolicy:
    """Base class for heuristic and baseline policies."""

    def select_action(
        self,
        state: np.ndarray | tuple,
        info: Optional[Dict[str, Any]] = None,
        training: bool = False,
    ) -> int:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        """No-op for non-learning policies."""
        pass

    def decay_epsilon(self) -> None:
        """No-op for baseline policies."""
        pass

    def __str__(self) -> str:
        return self.__class__.__name__


class RandomPolicy(BaselinePolicy):
    """Uniformly random action selection (0 = down, 1 = hold, 2 = up)."""

    def __init__(self, n_actions: int = 3, seed: Optional[int] = None):
        self.n_actions = n_actions
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        state: np.ndarray | tuple,
        info: Optional[Dict[str, Any]] = None,
        training: bool = False,
    ) -> int:
        return int(self.rng.randint(0, self.n_actions))


class ThresholdPolicy(BaselinePolicy):
    """Simple threshold-based autoscaling heuristic policy.

    Rules:
    - If utilization level is high (util_level == 2, >=80%): scale up (action 2)
    - If utilization level is low (util_level == 0, <40%): scale down (action 0)
    - Otherwise (medium, 40-80%): hold (action 1)
    """

    def __init__(self, n_actions: int = 3, seed: Optional[int] = None):
        self.n_actions = n_actions

    def select_action(
        self,
        state: np.ndarray | tuple,
        info: Optional[Dict[str, Any]] = None,
        training: bool = False,
    ) -> int:
        # state[0] is utilization level (0: low, 1: medium, 2: high)
        util_level = int(state[0])
        if util_level == 2:
            return 2  # scale up
        elif util_level == 0:
            return 0  # scale down
        return 1  # hold
