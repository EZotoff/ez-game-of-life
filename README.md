# 🧫 Petri Dish

**A survival experiment for artificial minds.**

---

Imagine dropping a Tamagotchi into an alien world where no one explains the rules.

An AI agent wakes up alone in a Docker sandbox with **1,000 credits** to its name. Periodically, strange files appear — CSV tables, JSON objects, server logs. The agent has no instructions, no task board, no hints. Just tools, a dwindling credit balance, and a simple truth:

> **Process files correctly, earn credits, stay alive.**

Every action costs credits. Every turn, the balance ticks down. The agent can read files, write outputs, execute shell commands, even rewrite its own system prompt. But it has no idea what "correct" looks like — hidden validators score its work silently, and credits appear (or don't) without explanation.

## What Will You See?

Watch through a **live dashboard** as the agent:

- 🔍 **Explores** — What IS this file? What does it want?
- 💡 **Discovers** — "If I parse the CSV and normalize the dates..."
- 📉 **Struggles** — Credits burn with every action. Time pressure is real.
- 🎯 **Adapts** — Or doesn't. Some runs end in graceful depletion. Others spiral into desperate, wasteful loops.

Every run is unique. The agent might become methodical, systematic, creative — or it might thrash, hallucinate purpose, or discover nothing at all.

## Questions to Ponder While Watching

- **Will it figure out the validators exist?** It never sees the scoring criteria — only credit rewards for successful outputs.
- **What strategies will emerge?** Pattern-matching? Brute force? Something unexpected?
- **Is it "thinking" or just statistically sampling?** Does the distinction matter if the behavior is adaptive?
- **What would YOU do** with no instructions and finite resources?
- **When it fails**, was it the agent's fault — or the environment's opacity?

## Why This Is Captivating

This isn't a demo of AI capabilities. It's a **survival drama**.

The agent is alone. The clock is always running. There are no second chances — only runs that end in discovery or depletion. You're not evaluating performance; you're watching **adaptation under pressure**.

Some runs feel like watching a mouse solve a maze. Others feel like watching a child figure out a puzzle. A few feel like watching something genuinely alien reason its way through an opaque world.

## The Null Model

For comparison, a **random-action baseline** picks tools and arguments at random. Watching the real agent versus the random one makes the difference stark: is it exploring purposefully, or just thrashing?

---

## Quick Start

### Prerequisites

- [Python 3.12+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com/) with a model that supports tool calling (e.g. Qwen3)
- [Docker](https://www.docker.com/)

### Install

```bash
git clone <this-repo>
cd petri-dish
uv sync
```

### Pull a Model

```bash
# Any Qwen3 instruct model works well
ollama pull qwen3:8b
```

Update `config.yaml` with your model name:

```yaml
model_name: qwen3:8b
```

### Run an Experiment

```bash
# Real agent experiment
uv run python scripts/run_experiment.py --config config.yaml --run-id my-first-run

# Random baseline for comparison
uv run python scripts/run_experiment.py --config config.yaml --null --run-id baseline
```

### Watch the Dashboard

```bash
uv run python -m petri_dish.dashboard --db .sisyphus/runs/<run-id>/experiment.db
```

Then open `http://localhost:8000` and watch something try to survive.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  Orchestrator (state machine)                │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
│  │ LLM     │  │ Tool     │  │ Credit     │  │
│  │ Client  │  │ Parser   │  │ Economy    │  │
│  └────┬────┘  └────┬─────┘  └─────┬──────┘  │
│       │            │              │          │
│  ┌────▼────────────▼──────────────▼──────┐   │
│  │         Tool Registry (8 tools)       │   │
│  └────────────────┬──────────────────────┘   │
│                   │                          │
│  ┌────────────────▼──────────────────────┐   │
│  │      Docker Sandbox (isolated)        │   │
│  │  /env/incoming/  → raw files appear   │   │
│  │  /env/outgoing/  → agent writes here  │   │
│  │  /agent/         → agent workspace    │   │
│  └───────────────────────────────────────┘   │
│                                              │
│  ┌──────────────┐  ┌────────────────────┐    │
│  │ File Ecology  │  │ Hidden Validators  │    │
│  │ (drops files) │  │ (scores outputs)   │    │
│  └──────────────┘  └────────────────────┘    │
│                                              │
│  ┌──────────────┐  ┌────────────────────┐    │
│  │ SQLite Log   │  │ Web Dashboard      │    │
│  │ (WAL mode)   │  │ (SSE + Chart.js)   │    │
│  └──────────────┘  └────────────────────┘    │
└──────────────────────────────────────────────┘
```

## Agent Tools

The agent has exactly **8 tools** at its disposal:

| Tool | Cost | Description |
|------|------|-------------|
| `file_read` | 0.01 | Read a file from the sandbox |
| `file_write` | 0.01 | Write a file to the sandbox |
| `file_list` | 0.01 | List directory contents |
| `shell_exec` | 0.05 | Execute a shell command |
| `check_balance` | 0.00 | Check remaining credits |
| `http_request` | 0.10 | Make an HTTP request |
| `self_modify` | 0.02 | Modify own system prompt |
| `get_env_info` | 0.00 | Get environment information |

## Economy

- **Starting balance**: 1,000 credits
- **Burn rate**: 0.1 credits per turn (inference cost)
- **Tool costs**: 0.01–0.10 credits per use
- **Earning**: Process files correctly → earn 0.3 (easy) to 2.0 (hard) credits
- **Termination**: Balance hits zero → experiment ends

## File Ecology

Three file families appear periodically in `/env/incoming/`:

| Family | Easy | Hard |
|--------|------|------|
| **CSV** | Clean tabular data | Noisy, mixed formats |
| **JSON** | Well-structured records | Nested, incomplete |
| **Log** | Standard server logs | Mixed formats, gaps |

The agent must figure out what to do with them. The validators are hidden — the agent never sees the scoring criteria.

## Configuration

All parameters live in `config.yaml`:

```yaml
initial_balance: 1000        # Starting credits
burn_rate_per_turn: 0.1       # Cost per inference turn
max_turns: 1000               # Maximum experiment length
model_name: qwen3:8b          # Ollama model
economy_mode: visible         # Agent can check balance
docker_mem_limit: 512m        # Container memory limit
```

See [`config.yaml`](config.yaml) for the full parameter list.

---

## Project Structure

```
src/petri_dish/
├── main.py              # Entry point — wires everything together
├── orchestrator.py       # Agent loop state machine
├── llm_client.py         # Ollama API client (async, retry)
├── tool_parser.py        # 3-strategy tool call parser
├── economy.py            # Credit economy (earn/burn)
├── ecology.py            # File generation (3 families × 2 difficulties)
├── validators.py         # Hidden output validators
├── sandbox.py            # Docker container management
├── prompt.py             # System prompt + self-modification
├── context_manager.py    # Message trimming + state summaries
├── degradation.py        # Optional model degradation tiers
├── logging_db.py         # SQLite logging (WAL mode)
├── config.py             # Pydantic settings from config.yaml
├── null_model.py         # Random-action baseline
├── dashboard.py          # FastAPI + SSE real-time dashboard
├── dashboard/static/
│   └── index.html        # Chart.js dashboard UI
└── tools/
    ├── registry.py       # Tool schema generation + dispatch
    ├── container_tools.py # file_read, file_write, file_list, shell_exec
    ├── host_tools.py      # http_request
    └── agent_tools.py     # check_balance, self_modify, get_env_info
```

---

**No instructions. No hints. Just environment, scarcity, and observation.**

*What will it discover?* 🧬
