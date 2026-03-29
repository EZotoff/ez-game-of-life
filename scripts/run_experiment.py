#!/usr/bin/env python3
"""Thin CLI wrapper for Petri Dish experiment runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from petri_dish.main import run_experiment_with_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Petri Dish experiment")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--null", action="store_true", help="Use null model baseline instead of Ollama"
    )
    parser.add_argument("--run-id", default=None, help="Optional custom run identifier")
    parser.add_argument(
        "--multi-agent",
        action="store_true",
        help="Enable multi-agent mode (uses multi_agent_count from config)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Number of agents (overrides multi_agent_count in config)",
    )
    args = parser.parse_args()

    if args.agents is not None or args.multi_agent:
        import yaml as _yaml

        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path) as f:
                cfg = _yaml.safe_load(f) or {}
        else:
            cfg = {}
        if args.multi_agent:
            cfg["multi_agent_enabled"] = True
        if args.agents is not None:
            cfg["multi_agent_enabled"] = True
            cfg["multi_agent_count"] = args.agents
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(config_path.parent)
        )
        _yaml.dump(cfg, tmp)
        tmp.close()
        config_to_use = tmp.name
    else:
        config_to_use = args.config

    result = run_experiment_with_id(
        config_path=config_to_use,
        null_model=args.null,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "total_turns": result.total_turns,
                "final_balance": result.final_balance,
                "tiers_reached": result.tiers_reached,
                "termination_reason": result.termination_reason,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
