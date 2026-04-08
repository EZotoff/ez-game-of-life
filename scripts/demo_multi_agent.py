#!/usr/bin/env python3
"""Multi-agent demo: 3 agents with traits + overseer scout.

Runs alice, bob, and carol through a configurable number of rounds
with the full stack: zod economy, file ecology, agent traits,
overseer scout (Tavily-backed), and the promotion pipeline.

Usage:
    # With Ollama (local):
    uv run python scripts/demo_multi_agent.py

    # With Z.AI API:
    ZAI_API_KEY=your-key TAVILY_API_KEY=your-key uv run python scripts/demo_multi_agent.py --api

    # Quick smoke (null model, no LLM):
    uv run python scripts/demo_multi_agent.py --null

    # Custom rounds:
    uv run python scripts/demo_multi_agent.py --null --rounds 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml
from petri_dish.config import Settings
from petri_dish.main import _run_multi_agent_async, _create_llm_client, _resolve_db_path
from petri_dish.economy import SharedReserve
from petri_dish.logging_db import LoggingDB
from petri_dish.orchestrator import MultiAgentOrchestrator
from petri_dish.sandbox import SandboxManager
from petri_dish.ecology import ResourceEcology
from petri_dish.validators import FileValidator
from petri_dish.traits import TraitEngine

import asyncio


def _print_header(settings: Settings) -> None:
    print("\n" + "=" * 70)
    print("  PETRI DISH — Multi-Agent Demo")
    print("=" * 70)
    print(f"  Agents:     {settings.multi_agent_names}")
    print(f"  Rounds:     {settings.max_turns}")
    print(f"  Backend:    {settings.llm_backend}")
    print(
        f"  Web Search: {settings.web_search_provider}  Overseer: {settings.overseer_enabled}"
    )
    print(f"  Traits:     {settings.traits_enabled}")
    print(f"  Promotion:  {settings.promotion_enabled}")
    print(f"  Starting $: {settings.initial_zod}")
    print("=" * 70 + "\n")


def _print_trait_snapshot(traits: TraitEngine, agent_ids: list[str]) -> None:
    print("\n--- Trait Snapshot ---")
    for aid in agent_ids:
        snap = traits.get_trait_snapshot(aid)
        traits_str = "  ".join(
            f"{k}={v:.2f}" for k, v in snap.items() if k != "agent_id"
        )
        print(f"  {aid}: {traits_str}")
    print()


def _print_results(
    result, settings: Settings, logging_db: LoggingDB, run_id: str
) -> None:
    print("\n" + "=" * 70)
    print("  RUN COMPLETE")
    print("=" * 70)
    print(f"  Total rounds:    {result.total_rounds}")
    print(f"  Common pool:     {result.common_pool:.2f}")
    print(f"  Termination:     {result.termination_reason}")

    for aid, ar in result.agent_results.items():
        print(f"\n  [{aid}]")
        print(f"    Turns:         {ar.total_turns}")
        print(f"    Final balance: {ar.final_balance:.2f}")
        print(f"    Termination:   {ar.termination_reason}")

    try:
        evals = (
            logging_db.get_overseer_evaluations(run_id)
            if hasattr(logging_db, "get_overseer_evaluations")
            else []
        )
    except Exception:
        evals = []
    if evals:
        print(f"\n  Overseer evals:  {len(evals)}")
        for ev in evals[:5]:
            print(
                f"    {ev.get('agent_id', '?')}: bonus={ev.get('bonus_granted', 0):.3f} dims={ev.get('dimensions', {})}"
            )

    print("=" * 70 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-agent Petri Dish demo")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config_demo_3agent_scout.yaml"),
        help="Config YAML path",
    )
    parser.add_argument("--null", action="store_true", help="Use null model")
    parser.add_argument("--api", action="store_true", help="Use Z.AI/OpenAI API")
    parser.add_argument("--rounds", type=int, default=None, help="Override max_turns")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    if args.api:
        cfg["llm_backend"] = "openai_compatible"
    if args.rounds is not None:
        cfg["max_turns"] = args.rounds

    tmp_path: str | None = None
    if args.api or args.rounds is not None:
        import tempfile, os

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=str(config_path.parent)
        )
        yaml.dump(cfg, tmp)
        tmp.close()
        tmp_path = tmp.name
        effective_config = tmp_path
    else:
        effective_config = str(config_path)

    settings = Settings.from_yaml(effective_config)
    _print_header(settings)

    from uuid import uuid4

    run_id = args.run_id or f"demo-3agent-{uuid4().hex[:8]}"

    result = asyncio.run(
        _run_multi_agent_async(
            settings=settings,
            config_path=effective_config,
            null_model=args.null,
            run_id=run_id,
        )
    )

    db_path = _resolve_db_path(run_id)
    logging_db = LoggingDB(db_path=db_path)
    logging_db.connect()
    _print_results(result, settings, logging_db, run_id)

    if tmp_path:
        try:
            import os

            os.unlink(tmp_path)
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
