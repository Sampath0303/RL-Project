"""Backward-compatibility shim for enhanced_autoscaling package."""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import src

# Ensure enhanced_autoscaling.src resolves to src
sys.modules["enhanced_autoscaling.src"] = src

