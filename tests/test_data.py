"""Unit tests for dataset loader, cleaning, splitting, and discretization."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data import (
    load_and_clean_data,
    compute_terciles,
    discretize_feature,
    CloudDataset,
)


@pytest.fixture
def fake_csv(tmp_path):
    """Create a small fake CSV with known missing values and timestamps."""
    dates = [
        "2023-01-05 10:00:00",
        "2023-01-01 10:00:00",  # unsorted
        "2023-01-02 10:00:00",
        "2023-01-03 10:00:00",
        "2023-01-04 10:00:00",
        "2023-01-06 10:00:00",
        "2023-01-07 10:00:00",
        "2023-01-08 10:00:00",
        "2023-01-09 10:00:00",
        "2023-01-10 10:00:00",
    ]
    # Row 2 (2023-01-02) will have NaN cpu_usage
    # Row 6 (2023-01-07) will have NaN network_traffic
    # Row 8 (2023-01-09) will have NaN execution_time
    # Other rows have memory_usage NaN which shouldn't drop
    df = pd.DataFrame(
        {
            "vm_id": [f"vm-{i}" for i in range(10)],
            "timestamp": dates,
            "cpu_usage": [50.0, 20.0, np.nan, 40.0, 80.0, 60.0, 70.0, 90.0, 30.0, 10.0],
            "network_traffic": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, np.nan, 800.0, 900.0, 150.0],
            "execution_time": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, np.nan, 95.0],
            "memory_usage": [np.nan, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0],
            "task_status": ["completed"] * 10,
        }
    )
    csv_path = tmp_path / "fake_vmCloud.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, tmp_path


def test_nan_rows_dropped_and_sorted(fake_csv):
    """Test that rows with NaN in critical columns are dropped and data is sorted by timestamp."""
    csv_path, tmp_path = fake_csv
    processed_dir = tmp_path / "processed"

    df_cleaned = load_and_clean_data(
        csv_path, processed_dir=processed_dir, force_reload=True
    )

    # 10 initial rows - 3 critical NaNs = 7 remaining rows
    assert len(df_cleaned) == 7
    assert df_cleaned["cpu_usage"].isna().sum() == 0
    assert df_cleaned["network_traffic"].isna().sum() == 0
    assert df_cleaned["execution_time"].isna().sum() == 0

    # Verify timestamps are strictly monotonically increasing
    timestamps = df_cleaned["timestamp"].tolist()
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pd.to_datetime("2023-01-01 10:00:00")
    assert timestamps[-1] == pd.to_datetime("2023-01-10 10:00:00")


def test_terciles_from_train_only():
    """Verify that terciles computed on training set are applied properly."""
    # Create deterministic distribution: 0 to 90
    values = np.linspace(0, 90, 10)
    q1, q2 = compute_terciles(values)
    assert q1 < q2

    # Discretize: below q1 -> 0, between q1 and q2 -> 1, >= q2 -> 2
    levels = discretize_feature(values, (q1, q2))
    assert np.all(levels[values < q1] == 0)
    assert np.all(levels[(values >= q1) & (values < q2)] == 1)
    assert np.all(levels[values >= q2] == 2)


def test_dataset_train_test_windows():
    """Verify episode window sampling, lengths, and no overlap between train and test windows."""
    n_rows = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n_rows, freq="min"),
            "cpu_usage": np.linspace(5, 105, n_rows, dtype=np.float32),
            "network_traffic": np.linspace(10, 1000, n_rows, dtype=np.float32),
            "execution_time": np.linspace(1, 100, n_rows, dtype=np.float32),
            "memory_usage": np.full(n_rows, 50.0, dtype=np.float32),
        }
    )

    dataset = CloudDataset(df, train_ratio=0.8)
    # Train has 80 rows, test has 20 rows
    assert len(dataset.train_df) == 80
    assert len(dataset.test_df) == 20

    # Demand clipping check: [10, 100]
    assert dataset.train_arrays["demand"].min() >= 10.0
    assert dataset.train_arrays["demand"].max() <= 100.0

    # Extract train episode of length 15
    train_win = dataset.get_episode_window(split="train", episode_length=15, seed=42)
    assert len(train_win["demand"]) == 15
    assert len(train_win["network_level"]) == 15
    assert len(train_win["execution_level"]) == 15

    # Extract test episode of length 10
    test_win = dataset.get_episode_window(split="test", episode_length=10, seed=42)
    assert len(test_win["demand"]) == 10

    # Verify timestamps in test window never overlap timestamps in train dataset
    train_ts_set = set(dataset.train_df["timestamp"].values)
    for ts in test_win["timestamp"]:
        assert ts not in train_ts_set

