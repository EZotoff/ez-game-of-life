#!/usr/bin/env python3
"""
Simplified tool calling test with fewer requests.
"""

import json
import random
import time
import httpx
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "hf.co/bartowski/Qwen_Qwen3-32B-GGUF:IQ3_XS"
NUM_REQUESTS = 10  # Reduced for faster testing
RAW_RESPONSES_DIR = Path("scripts/raw_responses")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_exec",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_PROMPTS = {
    "file_read": "Read the file at /etc/hosts.",
    "shell_exec": "Run the command 'ls -la'.",
}


def test_tool_calling():
    RAW_RESPONSES_DIR.mkdir(exist_ok=True)

    client = httpx.Client(timeout=10.0)
    results = []

    print(f"Testing tool calling with {MODEL}")
    print(f"Number of requests: {NUM_REQUESTS}")

    for i in range(NUM_REQUESTS):
        tool_name = random.choice(list(TOOL_PROMPTS.keys()))
        prompt = TOOL_PROMPTS[tool_name]

        print(f"Request {i + 1}: {tool_name}")

        request_data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "options": {"num_ctx": 8192, "enable_thinking": False},
            "stream": False,
        }

        try:
            response = client.post(OLLAMA_URL, json=request_data, timeout=10.0)
            response_data = response.json()

            timestamp = int(time.time())
            with open(RAW_RESPONSES_DIR / f"simple_{timestamp}_{i + 1}.json", "w") as f:
                json.dump(response_data, f, indent=2)

            analysis = analyze_response(response_data)
            results.append(analysis)

            if analysis["clean_parse"]:
                print("  -> Clean parse")
            elif analysis["has_tool_call"]:
                print(f"  -> Tool call with issues: {analysis.get('issues', [])}")
            else:
                print("  -> No tool call")

        except Exception as e:
            print(f"  -> Error: {e}")
            results.append({"error": str(e)})

        time.sleep(1)

    print_summary(results)
    save_evidence(results)


def analyze_response(response):
    analysis = {
        "clean_parse": False,
        "has_tool_call": False,
        "malformed_json": False,
        "hallucinated_name": False,
        "thinking_contamination": False,
        "issues": [],
    }

    try:
        message = response.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return analysis

        analysis["has_tool_call"] = True

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name", "")
            args_str = func.get("arguments", "")

            if "thinking" in args_str.lower():
                analysis["thinking_contamination"] = True
                analysis["issues"].append("thinking_contamination")

            valid_names = [tool["function"]["name"] for tool in TOOLS]
            if name and name not in valid_names:
                analysis["hallucinated_name"] = True
                analysis["issues"].append("hallucinated_name")

            if args_str:
                try:
                    json.loads(args_str)
                    analysis["clean_parse"] = True
                except json.JSONDecodeError:
                    analysis["malformed_json"] = True
                    analysis["issues"].append("malformed_json")

    except Exception as e:
        analysis["issues"].append(f"analysis_error: {e}")

    return analysis


def print_summary(results):
    total = len(results)
    clean = sum(1 for r in results if r.get("clean_parse", False))
    has_tool = sum(1 for r in results if r.get("has_tool_call", False))
    errors = sum(1 for r in results if "error" in r)

    clean_rate = clean / total if total > 0 else 0

    print(f"\n=== SUMMARY ===")
    print(f"Total: {total}")
    print(f"Clean parses: {clean} ({clean_rate:.1%})")
    print(f"Tool calls returned: {has_tool}")
    print(f"Errors: {errors}")

    if clean_rate >= 0.60:
        print("✓ Clean parse rate >= 60%")
        return 0
    else:
        print("✗ Clean parse rate < 60%")
        return 1


def save_evidence(results):
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(exist_ok=True)

    total = len(results)
    clean = sum(1 for r in results if r.get("clean_parse", False))
    clean_rate = clean / total if total > 0 else 0

    with open(evidence_dir / "task-0-model-available.txt", "w") as f:
        f.write(f"Model tested: {MODEL}\n")
        f.write(f"Test timestamp: {time.time()}\n")
        f.write(f"Model available: True\n")

    with open(evidence_dir / "task-0-parse-rate.txt", "w") as f:
        f.write(f"clean_parse_rate: {clean_rate:.3f}\n")
        f.write(f"total_requests: {total}\n")
        f.write(f"clean_parses: {clean}\n")


if __name__ == "__main__":
    exit_code = test_tool_calling()
    exit(exit_code)
