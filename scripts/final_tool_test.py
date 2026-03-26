#!/usr/bin/env python3
"""
Final tool calling test with correct analysis for Qwen3 format.
"""

import json
import random
import time
import httpx
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "hf.co/bartowski/Qwen_Qwen3-32B-GGUF:IQ3_XS"
NUM_REQUESTS = 20
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
    {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Make an HTTP request",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to request"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                    },
                },
                "required": ["url", "method"],
            },
        },
    },
]

TOOL_PROMPTS = {
    "file_read": "Read the file at /etc/hosts.",
    "shell_exec": "Run the command 'ls -la'.",
    "http_request": "Make a GET request to https://api.github.com.",
}


def test_tool_calling():
    RAW_RESPONSES_DIR.mkdir(exist_ok=True)

    client = httpx.Client(timeout=30.0)
    results = []

    print(f"Testing tool calling with {MODEL}")
    print(f"Requests: {NUM_REQUESTS}")
    print("-" * 40)

    for i in range(NUM_REQUESTS):
        tool_name = random.choice(list(TOOL_PROMPTS.keys()))
        prompt = TOOL_PROMPTS[tool_name]

        print(f"Request {i + 1:2d}: {tool_name:12s}", end="")

        request_data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "options": {"num_ctx": 8192, "enable_thinking": False},
            "stream": False,
        }

        try:
            response = client.post(OLLAMA_URL, json=request_data, timeout=30.0)
            response_data = response.json()

            timestamp = int(time.time())
            with open(RAW_RESPONSES_DIR / f"final_{timestamp}_{i + 1}.json", "w") as f:
                json.dump(response_data, f, indent=2)

            analysis = analyze_response(response_data, tool_name)
            results.append(analysis)

            if analysis["clean_parse"]:
                print(" -> Clean parse ✓")
            elif analysis["has_tool_call"]:
                issues = analysis.get("issues", [])
                if issues:
                    print(f" -> Issues: {', '.join(issues[:2])}")
                else:
                    print(" -> Tool call (no parse)")
            else:
                print(" -> No tool call")

        except Exception as e:
            print(f" -> Error: {str(e)[:30]}")
            results.append({"error": str(e), "request_num": i + 1})

        time.sleep(0.5)

    return print_summary(results)


def analyze_response(response, expected_tool):
    analysis = {
        "clean_parse": False,
        "has_tool_call": False,
        "thinking_contamination": False,
        "hallucinated_name": False,
        "correct_tool": False,
        "valid_arguments": False,
        "issues": [],
    }

    try:
        message = response.get("message", {})

        thinking = message.get("thinking", "")
        if thinking:
            analysis["thinking_contamination"] = True
            analysis["issues"].append("thinking_present")

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            return analysis

        analysis["has_tool_call"] = True

        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})

            if not name:
                analysis["issues"].append("missing_name")
                continue

            valid_names = [tool["function"]["name"] for tool in TOOLS]
            if name not in valid_names:
                analysis["hallucinated_name"] = True
                analysis["issues"].append(f"hallucinated_name:{name}")
            elif name == expected_tool:
                analysis["correct_tool"] = True

            if args:
                if isinstance(args, dict):
                    analysis["valid_arguments"] = True
                    analysis["clean_parse"] = True
                elif isinstance(args, str):
                    try:
                        json.loads(args)
                        analysis["valid_arguments"] = True
                        analysis["clean_parse"] = True
                    except json.JSONDecodeError:
                        analysis["issues"].append("malformed_json_string")
                else:
                    analysis["issues"].append(
                        f"invalid_args_type:{type(args).__name__}"
                    )

    except Exception as e:
        analysis["issues"].append(f"analysis_error:{str(e)[:30]}")

    return analysis


def print_summary(results):
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        print("\nNo valid results to analyze")
        return 1

    clean = sum(1 for r in valid_results if r.get("clean_parse", False))
    has_tool = sum(1 for r in valid_results if r.get("has_tool_call", False))
    thinking = sum(1 for r in valid_results if r.get("thinking_contamination", False))
    correct_tool = sum(1 for r in valid_results if r.get("correct_tool", False))

    clean_rate = clean / len(valid_results) if valid_results else 0

    print(f"\n{'=' * 40}")
    print("SUMMARY")
    print(f"{'=' * 40}")
    print(f"Total requests: {total}")
    print(f"Errors: {errors}")
    print(f"Valid responses: {len(valid_results)}")
    print(f"Clean parses: {clean} ({clean_rate:.1%})")
    print(f"Tool calls returned: {has_tool}")
    print(f"Correct tool selected: {correct_tool}")
    print(f"Thinking contamination: {thinking}")

    if valid_results:
        print(f"\nISSUES FOUND:")
        all_issues = []
        for r in valid_results:
            all_issues.extend(r.get("issues", []))

        from collections import Counter

        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common():
            print(f"  {issue}: {count}")

    save_evidence(clean_rate, len(valid_results), clean, thinking)

    if clean_rate >= 0.60:
        print(f"\n✓ SUCCESS: Clean parse rate >= 60%")
        return 0
    else:
        print(f"\n✗ FAILURE: Clean parse rate < 60%")
        print("Recommendation: Try smaller model (4B) or different quantization")
        return 1


def save_evidence(clean_rate, total_valid, clean_parses, thinking_count):
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(exist_ok=True)

    with open(evidence_dir / "task-0-model-available.txt", "w") as f:
        f.write(f"Model tested: {MODEL}\n")
        f.write(f"Test timestamp: {time.time()}\n")
        f.write(f"Model available: True\n")
        f.write(f"Ollama version: 0.13.4\n")
        f.write(f"Note: Qwen3.5 9B not available, using Qwen3 32B\n")

    with open(evidence_dir / "task-0-parse-rate.txt", "w") as f:
        f.write(f"clean_parse_rate: {clean_rate:.3f}\n")
        f.write(f"total_valid_requests: {total_valid}\n")
        f.write(f"clean_parses: {clean_parses}\n")
        f.write(f"thinking_contamination: {thinking_count}\n")
        f.write(f"model_used: {MODEL}\n")
        f.write(
            f"recommendation: {'PASS' if clean_rate >= 0.60 else 'FAIL - try smaller model'}\n"
        )


if __name__ == "__main__":
    exit_code = test_tool_calling()
    exit(exit_code)
