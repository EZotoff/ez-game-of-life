from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast


ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_overseer_config_files_load_via_settings() -> None:
    code = """
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from petri_dish.config import Settings

data = {}
for name in ('config_overseer_smoke.yaml', 'config_overseer_pilot.yaml', 'config_overseer_ga.yaml', 'config_overseer_forced_smoke_tavily.yaml'):
    s = Settings.from_yaml(name)
    data[name] = {
        'max_turns': s.max_turns,
            'budget': s.web_search_daily_budget,
            'queries': s.web_search_max_queries_per_call,
            'cost': s.tool_costs.get('web_search'),
            'provider': s.web_search_provider,
            'overseer_enabled': s.overseer_enabled,
        }

print(json.dumps(data))
"""
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    payload = cast(dict[str, object], json.loads(proc.stdout.strip()))
    assert payload["config_overseer_smoke.yaml"] == {
        "max_turns": 12,
        "budget": 6,
        "queries": 2,
        "cost": 0.15,
        "provider": "duckduckgo_instant_answer",
        "overseer_enabled": False,
    }
    assert payload["config_overseer_pilot.yaml"] == {
        "max_turns": 30,
        "budget": 20,
        "queries": 3,
        "cost": 0.15,
        "provider": "duckduckgo_instant_answer",
        "overseer_enabled": False,
    }
    assert payload["config_overseer_ga.yaml"] == {
        "max_turns": 60,
        "budget": 50,
        "queries": 3,
        "cost": 0.15,
        "provider": "duckduckgo_instant_answer",
        "overseer_enabled": False,
    }
    assert payload["config_overseer_forced_smoke_tavily.yaml"] == {
        "max_turns": 8,
        "budget": 5,
        "queries": 1,
        "cost": 0.15,
        "provider": "tavily",
        "overseer_enabled": False,
    }


def test_run_experiment_cli_accepts_overseer_smoke_config() -> None:
    code = """
import importlib.util
import json
import types
import sys
from pathlib import Path

root = Path('.').resolve()
sys.path.insert(0, str((root / 'src').resolve()))

spec = importlib.util.spec_from_file_location('run_experiment_script', root / 'scripts' / 'run_experiment.py')
module = importlib.util.module_from_spec(spec)

observed = {}

class FakeResult:
    total_turns = 1
    final_balance = 1.0
    tiers_reached = []
    termination_reason = 'ok'

def fake_run_experiment_with_id(*, config_path, null_model, run_id):
    observed['config_path'] = config_path
    observed['null_model'] = null_model
    observed['run_id'] = run_id
    from petri_dish.config import Settings
    loaded = Settings.from_yaml(config_path)
    observed['llm_backend'] = loaded.llm_backend
    observed['budget'] = loaded.web_search_daily_budget
    return FakeResult()

main_mod = types.ModuleType('petri_dish.main')
main_mod.run_experiment_with_id = fake_run_experiment_with_id
sys.modules['petri_dish.main'] = main_mod
spec.loader.exec_module(module)
sys.argv = ['run_experiment.py', '--config', 'config_overseer_smoke.yaml', '--run-id', 'scout-smoke-001', '--api']
exit_code = module.main()

print(json.dumps({
    'exit_code': exit_code,
    'run_id': observed['run_id'],
    'null_model': observed['null_model'],
    'llm_backend': observed['llm_backend'],
    'budget': observed['budget'],
}))
"""
    proc = _run_python(code)
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = cast(dict[str, object], json.loads(lines[-1]))
    assert payload["exit_code"] == 0
    assert payload["run_id"] == "scout-smoke-001"
    assert payload["null_model"] is False
    assert payload["llm_backend"] == "openai_compatible"
    assert payload["budget"] == 6
