"""Dataset loading, cleaning, rolling windows, and train/test splitting."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS_TO_LOAD = [
    "timestamp",
    "cpu_usage",
    "network_traffic",
    "execution_time",
    "memory_usage",
]

DTYPE_MAPPING = {
    "cpu_usage": "float32",
    "network_traffic": "float32",
    "execution_time": "float32",
    "memory_usage": "float32",
}


def load_and_clean_data(
    raw_path: str | Path,
    processed_dir: str | Path | None = None,
    force_reload: bool = False,
) -> pd.DataFrame:
    """Load specified columns, drop rows with critical missing values, sort by timestamp,

    and cache the cleaned dataframe to processed_dir.
    """
    raw_path = Path(raw_path)
    if processed_dir is None:
        processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_npz_path = processed_dir / f"{raw_path.stem}_cleaned.npz"

    if not force_reload and cache_npz_path.exists():
        print(f"Loading cleaned data from cache: {cache_npz_path}")
        with np.load(cache_npz_path, allow_pickle=True) as data:
            df = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(data["timestamp"]),
                    "cpu_usage": data["cpu_usage"],
                    "network_traffic": data["network_traffic"],
                    "execution_time": data["execution_time"],
                    "memory_usage": data["memory_usage"],
                }
            )
        return df

    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(
        raw_path,
        usecols=COLUMNS_TO_LOAD,
        dtype=DTYPE_MAPPING,
        parse_dates=["timestamp"],
    )
    initial_rows = len(df)

    # Cleaning: drop rows where timestamp, cpu_usage, network_traffic or execution_time is missing
    clean_subset = ["timestamp", "cpu_usage", "network_traffic", "execution_time"]
    df_cleaned = df.dropna(subset=clean_subset).copy()
    dropped_rows = initial_rows - len(df_cleaned)
    print(
        f"Cleaned dataset: dropped {dropped_rows:,} rows out of {initial_rows:,} "
        f"({dropped_rows / initial_rows * 100:.2f}%) due to missing values."
    )

    # Sort strictly by timestamp
    df_cleaned = df_cleaned.sort_values(by="timestamp").reset_index(drop=True)

    # Save to NPZ cache
    print(f"Caching cleaned data to: {cache_npz_path}")
    np.savez_compressed(
        cache_npz_path,
        timestamp=df_cleaned["timestamp"].values.astype("datetime64[ns]"),
        cpu_usage=df_cleaned["cpu_usage"].values.astype(np.float32),
        network_traffic=df_cleaned["network_traffic"].values.astype(np.float32),
        execution_time=df_cleaned["execution_time"].values.astype(np.float32),
        memory_usage=df_cleaned["memory_usage"].values.astype(np.float32),
    )

    return df_cleaned


def compute_terciles(train_series: pd.Series | np.ndarray) -> Tuple[float, float]:
    """Compute 33.33% and 66.67% quantile thresholds."""
    q1 = float(np.percentile(train_series, 100.0 / 3.0))
    q2 = float(np.percentile(train_series, 200.0 / 3.0))
    return q1, q2


def discretize_feature(
    values: np.ndarray, thresholds: Tuple[float, float]
) -> np.ndarray:
    """Discretize continuous feature into levels 0, 1, or 2 based on thresholds (q1, q2)."""
    q1, q2 = thresholds
    levels = np.zeros_like(values, dtype=np.int32)
    levels[values >= q1] = 1
    levels[values >= q2] = 2
    return levels


class CloudDataset:
    """Manages cloud dataset splitting, feature discretizing, and episode sampling."""

    def __init__(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        network_thresholds: Optional[Tuple[float, float]] = None,
        exec_thresholds: Optional[Tuple[float, float]] = None,
    ):
        self.df = df
        n_total = len(df)
        split_idx = int(n_total * train_ratio)

        self.train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        self.test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

        # Compute terciles from train partition ONLY if not provided
        if network_thresholds is None:
            self.network_thresholds = compute_terciles(
                self.train_df["network_traffic"].values
            )
        else:
            self.network_thresholds = network_thresholds

        if exec_thresholds is None:
            self.exec_thresholds = compute_terciles(
                self.train_df["execution_time"].values
            )
        else:
            self.exec_thresholds = exec_thresholds

        # Process train arrays
        self.train_arrays = self._process_partition(self.train_df)
        # Process test arrays
        self.test_arrays = self._process_partition(self.test_df)

    def _process_partition(self, part_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        # demand = cpu_usage clipped to [10, 100]
        demand = np.clip(part_df["cpu_usage"].values.astype(np.float32), 10.0, 100.0)
        net_raw = part_df["network_traffic"].values.astype(np.float32)
        exec_raw = part_df["execution_time"].values.astype(np.float32)

        network_level = discretize_feature(net_raw, self.network_thresholds)
        exec_level = discretize_feature(exec_raw, self.exec_thresholds)

        return {
            "demand": demand,
            "network_raw": net_raw,
            "execution_raw": exec_raw,
            "network_level": network_level,
            "execution_level": exec_level,
            "timestamp": part_df["timestamp"].values,
        }

    def get_episode_window(
        self,
        split: str = "train",
        episode_length: int = 1000,
        seed: Optional[int] = None,
        start_idx: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Extract a contiguous window of episode_length steps from either train or test split."""
        arrays = self.train_arrays if split == "train" else self.test_arrays
        total_len = len(arrays["demand"])

        if episode_length > total_len:
            raise ValueError(
                f"episode_length ({episode_length}) cannot exceed partition length ({total_len})"
            )

        max_start = total_len - episode_length
        if start_idx is None:
            rng = np.random.RandomState(seed)
            start_idx = rng.randint(0, max_start + 1)
        else:
            if start_idx < 0 or start_idx > max_start:
                raise ValueError(
                    f"start_idx {start_idx} out of range [0, {max_start}]"
                )

        end_idx = start_idx + episode_length

        return {
            "demand": arrays["demand"][start_idx:end_idx],
            "network_level": arrays["network_level"][start_idx:end_idx],
            "execution_level": arrays["execution_level"][start_idx:end_idx],
            "network_raw": arrays["network_raw"][start_idx:end_idx],
            "execution_raw": arrays["execution_raw"][start_idx:end_idx],
            "timestamp": arrays["timestamp"][start_idx:end_idx],
        }


def profile_dataset(
    raw_path: str | Path,
    figures_dir: str | Path | None = None,
    force_reload: bool = False,
) -> Dict[str, object]:
    """Profile the dataset, print summary report, and save plots to figures_dir."""
    raw_path = Path(raw_path)
    if figures_dir is None:
        figures_dir = raw_path.parent.parent.parent / "figures"
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Row counts before cleaning
    # Fast row count without loading all data
    with open(raw_path, "rb") as f:
        total_lines = sum(1 for _ in f)
    raw_total_rows = total_lines - 1  # excluding header

    df = load_and_clean_data(raw_path, force_reload=force_reload)
    cleaned_rows = len(df)
    dropped_rows = raw_total_rows - cleaned_rows

    date_min = df["timestamp"].min()
    date_max = df["timestamp"].max()

    # 2. Correlation matrix of key columns
    feature_cols = ["cpu_usage", "network_traffic", "execution_time"]
    corr_matrix = df[feature_cols].corr()

    # 3. Autocorrelation of cpu_usage at lag 1
    cpu_series = df["cpu_usage"]
    autocorr_lag1 = float(cpu_series.autocorr(lag=1))

    # 4. Terciles of train split (80%)
    split_idx = int(cleaned_rows * 0.8)
    train_net = df["network_traffic"].iloc[:split_idx]
    train_exec = df["execution_time"].iloc[:split_idx]
    net_q1, net_q2 = compute_terciles(train_net.values)
    exec_q1, exec_q2 = compute_terciles(train_exec.values)

    # Text report
    summary = (
        f"================ DATASET PROFILE REPORT ================\n"
        f"File: {raw_path}\n"
        f"Raw Rows:               {raw_total_rows:,}\n"
        f"Cleaned Rows:           {cleaned_rows:,}\n"
        f"Dropped Rows:           {dropped_rows:,} ({dropped_rows / raw_total_rows * 100:.2f}%)\n"
        f"Date Range:             {date_min} to {date_max}\n"
        f"Autocorrelation (lag 1): {autocorr_lag1:.5f}\n"
        f"\nTrain Terciles (first 80%):\n"
        f"  network_traffic:      [{net_q1:.2f}, {net_q2:.2f}]\n"
        f"  execution_time:       [{exec_q1:.2f}, {exec_q2:.2f}]\n"
        f"\nCorrelation Matrix:\n"
        f"{corr_matrix.to_string()}\n"
        f"========================================================\n"
    )
    print(summary)

    # Save summary report text
    report_file = figures_dir / "dataset_profile_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(summary)

    # 5. Save Plots
    # Histogram of the 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(df["cpu_usage"].dropna(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].set_title("cpu_usage")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(df["network_traffic"].dropna(), bins=50, color="forestgreen", edgecolor="black", alpha=0.7)
    axes[1].set_title("network_traffic")
    axes[1].set_xlabel("Value")

    axes[2].hist(df["execution_time"].dropna(), bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[2].set_title("execution_time")
    axes[2].set_xlabel("Value")

    plt.tight_layout()
    hist_path = figures_dir / "dataset_histograms.png"
    plt.savefig(hist_path, dpi=150)
    plt.close(fig)

    # First 500 demand values
    demand_500 = np.clip(df["cpu_usage"].iloc[:500].values, 10.0, 100.0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(demand_500, label="Demand (Clipped CPU %)", color="royalblue", lw=1.2)
    ax.set_title("First 500 Demand Values (Clipped [10, 100])")
    ax.set_xlabel("Step")
    ax.set_ylabel("Demand (%)")
    ax.set_ylim([0, 105])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    demand_plot_path = figures_dir / "first_500_demand.png"
    plt.savefig(demand_plot_path, dpi=150)
    plt.close(fig)

    return {
        "raw_total_rows": raw_total_rows,
        "cleaned_rows": cleaned_rows,
        "dropped_rows": dropped_rows,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "autocorr_lag1": autocorr_lag1,
        "corr_matrix": corr_matrix.to_dict(),
        "network_terciles": (net_q1, net_q2),
        "exec_terciles": (exec_q1, exec_q2),
        "hist_path": str(hist_path),
        "demand_plot_path": str(demand_plot_path),
    }


if __name__ == "__main__":
    profile_dataset("data/raw/vmCloud_data.csv")
