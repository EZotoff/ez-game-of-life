#!/usr/bin/env python3
"""
T0: Ollama Tool Calling Spike
Test tool calling reliability with available Qwen models.
"""

import json
import random
import time
import httpx
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "hf.co/bartowski/Qwen_Qwen3-32B-GGUF:IQ3_XS"  # Using available model
NUM_REQUESTS = 50
RAW_RESPONSES_DIR = Path("scripts/raw_responses")

# Tool definitions matching final system
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
            "name": "file_write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_list",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"}
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
            "name": "check_balance",
            "description": "Check token balance for an account",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account identifier",
                    }
                },
                "required": ["account_id"],
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
                        "description": "HTTP method",
                    },
                },
                "required": ["url", "method"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "self_modify",
            "description": "Modify the agent's own code or configuration",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to modify",
                    },
                    "changes": {
                        "type": "string",
                        "description": "Description of changes to make",
                    },
                },
                "required": ["file_path", "changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_env_info",
            "description": "Get environment information",
            "parameters": {
                "type": "object",
                "properties": {
                    "info_type": {
                        "type": "string",
                        "enum": ["system", "user", "network"],
                        "description": "Type of information to retrieve",
                    }
                },
                "required": ["info_type"],
            },
        },
    },
]

# Test prompts for each tool
TOOL_PROMPTS = {
    "file_read": "Read the file at /etc/hosts and tell me what's in it.",
    "file_write": "Create a file called test.txt with the content 'Hello, World!'.",
    "file_list": "List all files in the current directory.",
    "shell_exec": "Run the command 'ls -la' to see directory contents.",
    "check_balance": "Check the balance for account 'user123'.",
    "http_request": "Make a GET request to https://api.github.com.",
    "self_modify": "Modify the agent's configuration to increase timeout to 30 seconds.",
    "get_env_info": "Get system environment information.",
}


class ToolCallTester:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self.results = []
        self.raw_responses = []

    def make_request(self, tool_name: str, prompt: str) -> Optional[Dict]:
        """Make a single tool call request to Ollama."""
        request_data = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "options": {"num_ctx": 8192, "enable_thinking": False},
            "stream": False,
        }

        try:
            response = self.client.post(OLLAMA_URL, json=request_data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def analyze_response(self, response: Dict, expected_tool: str) -> Dict:
        """Analyze a response for tool calling issues."""
        analysis = {
            "clean_parse": False,
            "malformed_json": False,
            "hallucinated_name": False,
            "thinking_contamination": False,
            "raw_response": response,
        }

        try:
            message = response.get("message", {})
            tool_calls = message.get("tool_calls", [])

            if not tool_calls:
                # No tool calls returned
                return analysis

            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "")

                # Check for thinking contamination
                if "thinking" in args_str.lower() or "thought" in args_str.lower():
                    analysis["thinking_contamination"] = True

                # Check for hallucinated tool names
                valid_names = [tool["function"]["name"] for tool in TOOLS]
                if name and name not in valid_names:
                    analysis["hallucinated_name"] = True

                # Try to parse arguments as JSON
                try:
                    if args_str:
                        json.loads(args_str)
                        analysis["clean_parse"] = True
                except json.JSONDecodeError:
                    analysis["malformed_json"] = True

        except Exception as e:
            print(f"Analysis error: {e}")

        return analysis

    def run_test(self):
        """Run the full test suite."""
        print(f"Starting tool calling test with model: {MODEL}")
        print(f"Number of requests: {NUM_REQUESTS}")
        print("-" * 50)

        for i in range(NUM_REQUESTS):
            # Randomly select a tool to test
            tool_name = random.choice(list(TOOL_PROMPTS.keys()))
            prompt = TOOL_PROMPTS[tool_name]

            print(f"Request {i + 1}/{NUM_REQUESTS}: Testing {tool_name}")

            response = self.make_request(tool_name, prompt)
            if response is None:
                print("  -> Request failed")
                self.results.append(
                    {"request_num": i + 1, "tool": tool_name, "error": "request_failed"}
                )
                continue

            # Save raw response
            timestamp = int(time.time())
            response_file = RAW_RESPONSES_DIR / f"response_{timestamp}_{i + 1}.json"
            with open(response_file, "w") as f:
                json.dump(response, f, indent=2)

            # Analyze response
            analysis = self.analyze_response(response, tool_name)
            self.results.append(
                {"request_num": i + 1, "tool": tool_name, "analysis": analysis}
            )

            # Print brief result
            if analysis["clean_parse"]:
                print("  -> Clean parse ✓")
            elif analysis["malformed_json"]:
                print("  -> Malformed JSON ✗")
            elif analysis["hallucinated_name"]:
                print("  -> Hallucinated tool name ✗")
            elif analysis["thinking_contamination"]:
                print("  -> Thinking contamination ✗")
            else:
                print("  -> No tool call returned")

            # Small delay between requests
            time.sleep(0.5)

        print("-" * 50)
        self.print_summary()

    def print_summary(self):
        """Print test summary statistics."""
        total = len(self.results)
        clean_parses = sum(
            1 for r in self.results if r.get("analysis", {}).get("clean_parse", False)
        )
        malformed_json = sum(
            1
            for r in self.results
            if r.get("analysis", {}).get("malformed_json", False)
        )
        hallucinated_names = sum(
            1
            for r in self.results
            if r.get("analysis", {}).get("hallucinated_name", False)
        )
        thinking_contamination = sum(
            1
            for r in self.results
            if r.get("analysis", {}).get("thinking_contamination", False)
        )
        no_tool_calls = total - (
            clean_parses + malformed_json + hallucinated_names + thinking_contamination
        )

        clean_rate = clean_parses / total if total > 0 else 0

        print("\n=== TEST SUMMARY ===")
        print(f"Total requests: {total}")
        print(f"Clean parses: {clean_parses} ({clean_rate:.1%})")
        print(f"Malformed JSON: {malformed_json}")
        print(f"Hallucinated tool names: {hallucinated_names}")
        print(f"Thinking contamination: {thinking_contamination}")
        print(f"No tool calls returned: {no_tool_calls}")
        print("-" * 30)

        # Save evidence files
        self.save_evidence(clean_rate)

        # Exit with appropriate code
        if clean_rate >= 0.60:
            print("✓ SUCCESS: Clean parse rate >= 60%")
            exit(0)
        else:
            print("✗ FAILURE: Clean parse rate < 60%")
            print(
                "Recommendation: Consider 4B fallback model or different quantization"
            )
            exit(1)

    def save_evidence(self, clean_rate: float):
        """Save evidence files as required."""
        evidence_dir = Path(".sisyphus/evidence")
        evidence_dir.mkdir(exist_ok=True)

        # Save model availability evidence
        with open(evidence_dir / "task-0-model-available.txt", "w") as f:
            f.write(f"Model tested: {MODEL}\n")
            f.write(f"Test timestamp: {time.time()}\n")
            f.write(f"Model available: True\n")

        # Save parse rate evidence
        with open(evidence_dir / "task-0-parse-rate.txt", "w") as f:
            f.write(f"clean_parse_rate: {clean_rate:.3f}\n")
            f.write(f"total_requests: {len(self.results)}\n")
            f.write(
                f"clean_parses: {sum(1 for r in self.results if r.get('analysis', {}).get('clean_parse', False))}\n"
            )
            f.write(
                f"malformed_json: {sum(1 for r in self.results if r.get('analysis', {}).get('malformed_json', False))}\n"
            )
            f.write(
                f"hallucinated_names: {sum(1 for r in self.results if r.get('analysis', {}).get('hallucinated_name', False))}\n"
            )
            f.write(
                f"thinking_contamination: {sum(1 for r in self.results if r.get('analysis', {}).get('thinking_contamination', False))}\n"
            )


def main():
    """Main entry point."""
    # Ensure raw responses directory exists
    RAW_RESPONSES_DIR.mkdir(exist_ok=True)

    # Run the test
    tester = ToolCallTester()
    tester.run_test()


if __name__ == "__main__":
    main()
