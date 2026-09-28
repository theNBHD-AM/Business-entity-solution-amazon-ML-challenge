"""
CLI Entry Point
================
    python run.py --config configs/pipeline_config.yaml --mode train
    python run.py --config configs/pipeline_config.yaml --mode test
    python run.py --config configs/pipeline_config.yaml --mode full
"""
from __future__ import annotations

import sys
from pathlib import Path
import click

try:
    from .pipeline import EntityResolutionPipeline, load_config, setup_logging
except (ImportError, ValueError):
    _src_parent = str(Path(__file__).resolve().parent.parent)
    if _src_parent not in sys.path:
        sys.path.insert(0, _src_parent)
    from src.pipeline import EntityResolutionPipeline, load_config, setup_logging


@click.command()
@click.option(
    "--config", "-c",
    default="configs/pipeline_config.yaml",
    help="Path to pipeline_config.yaml",
    show_default=True,
)
@click.option(
    "--mode", "-m",
    type=click.Choice(["train", "test", "full"]),
    default="full",
    help="Pipeline mode: train, test, or full (train+test)",
    show_default=True,
)
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level",
    show_default=True,
)
def main(config: str, mode: str, log_level: str) -> None:
    """Business Entity Resolution — production pipeline CLI."""
    cfg = load_config(config)
    setup_logging(
        log_dir=cfg.get("logging", {}).get("log_dir", "logs"),
        level=log_level,
    )
    pipeline = EntityResolutionPipeline(cfg)

    if mode in ("train", "full"):
        pipeline.train()
    if mode in ("test", "full"):
        pipeline.test()


if __name__ == "__main__":
    main()
