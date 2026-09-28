"""
Root execution script for Business Entity Resolution Pipeline.

Usage:
    python run.py --config configs/pipeline_config.yaml --mode full
    python run.py --config configs/pipeline_config.yaml --mode train
    python run.py --config configs/pipeline_config.yaml --mode test
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add code/business_entity_resolution to sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "code" / "business_entity_resolution"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from src.main import main

if __name__ == "__main__":
    main()
