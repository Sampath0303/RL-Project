"""Enhanced autoscaling source package."""
from pathlib import Path
import yaml

def load_config(config_path: str | Path | None = None) -> dict:
    """Load configuration from a YAML file.
    
    If config_path is None, loads the default config from configs/default.yaml.
    """
    if config_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        config_path = base_dir / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config

