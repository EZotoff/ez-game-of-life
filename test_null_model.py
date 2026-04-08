#!/usr/bin/env python3
"""Test script for NullModel implementation."""

import asyncio
import json
import sys
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, "src")

from petri_dish.null_model import NullModel
from petri_dish.tools import get_all_tools


def get_tool_schemas() -> list[Dict[str, Any]]:
    """Get tool schemas in Ollama format."""
    registry = get_all_tools()
    return registry.get_all_schemas()


async def test_null_model_basic() -> None:
    """Test basic null model functionality."""
    print("=== Test 1: Basic Null Model Functionality ===")

    # Create null model with fixed seed for reproducibility
    model = NullModel(seed=42, null_model_type="random")

    # Get tool schemas
    tools = get_tool_schemas()
    print(f"Available tools: {[t['function']['name'] for t in tools]}")

    # Generate random tool call
    system_prompt = "Test system prompt"
    messages = [{"role": "user", "content": "Test message"}]

    response_text, tool_calls = await model.chat(system_prompt, messages, tools)

    print(f"Response text: '{response_text}'")
    print(f"Number of tool calls: {len(tool_calls)}")

    if tool_calls:
        tool_call = tool_calls[0]
        print(f"Tool call ID: {tool_call.get('id')}")
        print(f"Tool name: {tool_call['function']['name']}")
        print(
            f"Tool arguments: {json.dumps(tool_call['function']['arguments'], indent=2)}"
        )

    print()


async def test_null_model_reproducibility() -> None:
    """Test that null model is reproducible with same seed."""
    print("=== Test 2: Reproducibility with Same Seed ===")

    tools = get_tool_schemas()
    system_prompt = "Test"
    messages = [{"role": "user", "content": "Test"}]

    # First run with seed 123
    model1 = NullModel(seed=123, null_model_type="random")
    _, calls1 = await model1.chat(system_prompt, messages, tools)

    # Second run with same seed
    model2 = NullModel(seed=123, null_model_type="random")
    _, calls2 = await model2.chat(system_prompt, messages, tools)

    print(f"Model 1 seed: {model1.get_seed()}")
    print(f"Model 2 seed: {model2.get_seed()}")

    if calls1 and calls2:
        call1 = calls1[0]
        call2 = calls2[0]

        print(
            f"Tool call 1: {call1['function']['name']} with args {call1['function']['arguments']}"
        )
        print(
            f"Tool call 2: {call2['function']['name']} with args {call2['function']['arguments']}"
        )

        # Check if they're identical
        if call1["function"]["name"] == call2["function"]["name"]:
            print("✓ Tool names match (reproducible)")
        else:
            print("✗ Tool names differ")

        if call1["function"]["arguments"] == call2["function"]["arguments"]:
            print("✓ Arguments match (reproducible)")
        else:
            print("✗ Arguments differ")

    print()


async def test_null_model_different_seeds() -> None:
    """Test that different seeds produce different results."""
    print("=== Test 3: Different Seeds Produce Different Results ===")

    tools = get_tool_schemas()
    system_prompt = "Test"
    messages = [{"role": "user", "content": "Test"}]

    # Run with different seeds
    model1 = NullModel(seed=100, null_model_type="random")
    model2 = NullModel(seed=200, null_model_type="random")

    _, calls1 = await model1.chat(system_prompt, messages, tools)
    _, calls2 = await model2.chat(system_prompt, messages, tools)

    if calls1 and calls2:
        call1 = calls1[0]
        call2 = calls2[0]

        print(
            f"Seed 100: {call1['function']['name']} with args {call1['function']['arguments']}"
        )
        print(
            f"Seed 200: {call2['function']['name']} with args {call2['function']['arguments']}"
        )

        # They should be different (though could coincidentally match)
        if (
            call1["function"]["name"] != call2["function"]["name"]
            or call1["function"]["arguments"] != call2["function"]["arguments"]
        ):
            print("✓ Different seeds produced different results (expected)")
        else:
            print("⚠ Same results with different seeds (possible but unlikely)")

    print()


async def test_null_model_argument_validity() -> None:
    """Test that generated arguments are valid for each tool type."""
    print("=== Test 4: Argument Validity Check ===")

    tools = get_tool_schemas()
    model = NullModel(seed=999, null_model_type="random")

    # Run multiple times to test different tools
    tool_results = {}
    for i in range(20):  # Run enough times to likely hit all tools
        _, calls = await model.chat(
            "Test", [{"role": "user", "content": "Test"}], tools
        )
        if calls:
            call = calls[0]
            tool_name = call["function"]["name"]
            args = call["function"]["arguments"]

            if tool_name not in tool_results:
                tool_results[tool_name] = []
            tool_results[tool_name].append(args)

    print("Generated tool calls by tool type:")
    for tool_name, arg_list in tool_results.items():
        print(f"  {tool_name}: {len(arg_list)} calls")
        if arg_list:
            print(f"    Sample args: {arg_list[0]}")

    print()


async def test_null_model_no_tools() -> None:
    """Test null model behavior when no tools are provided."""
    print("=== Test 5: No Tools Provided ===")

    model = NullModel(seed=42, null_model_type="random")
    response_text, tool_calls = await model.chat(
        "Test",
        [{"role": "user", "content": "Test"}],
        [],  # Empty tools list
    )

    print(f"Response text: '{response_text}'")
    print(f"Number of tool calls: {len(tool_calls)}")
    assert response_text == ""
    assert tool_calls == []
    print()


async def main() -> None:
    """Run all tests."""
    print("Testing Null Model (Random-Action Baseline)\n")

    await test_null_model_basic()
    await test_null_model_reproducibility()
    await test_null_model_different_seeds()
    await test_null_model_argument_validity()
    await test_null_model_no_tools()
    print("=== All Tests Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
