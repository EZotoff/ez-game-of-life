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
    args = parser.parse_args()

    result = run_experiment_with_id(
        config_path=args.config,
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
