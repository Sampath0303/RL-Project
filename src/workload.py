"""Synthetic workload generator with leading indicators and load-dependent metrics."""
from __future__ import annotations

from typing import Tuple
import numpy as np


def generate_demand(
    length: int,
    seed: int | None = None,
    base_load: float = 45.0,
    daily_amplitude: float = 20.0,
    daily_period: int = 144,  # e.g., 144 steps per day (10 min per step)
    weekly_period: int = 1008,
    weekly_drift: float = 8.0,
    noise_std: float = 3.0,
    spike_prob: float = 0.05,
    spike_magnitude: float = 20.0,
) -> np.ndarray:
    """Generate structured synthetic demand in range [10, 100].

    Components:
    - Daily sine cycle
    - Weekly drift
    - Gaussian noise
    - Occasional spikes (spike_prob, e.g. 5% of steps, +20)
    - Clipped to [10.0, 100.0]
    Seeded with independent np.random.RandomState (no global state modification).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(length, dtype=np.float64)

    # 1. Daily cycle
    daily_cycle = daily_amplitude * np.sin(2 * np.pi * t / daily_period)

    # 2. Weekly drift
    weekly = weekly_drift * np.sin(2 * np.pi * t / weekly_period)

    # 3. Base + periodic variations
    signal = base_load + daily_cycle + weekly

    # 4. Gaussian noise
    noise = rng.normal(loc=0.0, scale=noise_std, size=length)

    # 5. Occasional spikes
    spike_mask = rng.rand(length) < spike_prob
    spikes = spike_mask * spike_magnitude

    demand = signal + noise + spikes
    return np.clip(demand, 10.0, 100.0).astype(np.float32)


def generate_network_traffic(
    demand: np.ndarray,
    seed: int | None = None,
    lead: int = 1,
    scale: float = 10.0,  # e.g. demand [10, 100] -> network roughly [100, 1000]
    noise_std: float = 25.0,
) -> np.ndarray:
    """Generate network traffic that acts as a LEADING indicator for demand.

    network[t] reflects demand[t + lead] scaled and noisy.
    The last `lead` steps are handled safely (using the final demand value).
    """
    rng = np.random.RandomState(seed)
    length = len(demand)

    # Shift demand forward in time so network[t] anticipates future demand
    shifted_demand = np.empty_like(demand)
    if length > lead:
        shifted_demand[:-lead] = demand[lead:]
        shifted_demand[-lead:] = demand[-1]
    else:
        shifted_demand[:] = demand[-1]

    noise = rng.normal(loc=0.0, scale=noise_std, size=length).astype(np.float32)
    network = shifted_demand * scale + noise
    return np.clip(network, 0.0, 1000.0).astype(np.float32)


def exec_time(
    utilization: float | np.ndarray,
    seed: int | None = None,
    base_latency: float = 15.0,
    noise_std: float = 2.0,
    max_util_cap: float = 0.95,
) -> float | np.ndarray:
    """Compute queueing-style load-dependent execution time.

    Formula:
        latency = base / (1 - min(utilization, max_util_cap)) + noise
    Higher utilization directly increases latency and queueing delay.
    """
    rng = np.random.RandomState(seed)
    util = np.minimum(utilization, max_util_cap)
    denom = 1.0 - util
    # Prevent divide by zero (safe threshold)
    denom = np.maximum(denom, 1e-4)

    latency = base_latency / denom

    if isinstance(latency, np.ndarray):
        noise = rng.normal(0.0, noise_std, size=latency.shape)
        return np.maximum(latency + noise, 1.0).astype(np.float32)
    else:
        noise = rng.normal(0.0, noise_std)
        return float(max(latency + noise, 1.0))
