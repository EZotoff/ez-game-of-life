#!/usr/bin/env python3
"""Test script for PromptManager to generate evidence for QA scenarios."""

import os
import tempfile
from pathlib import Path

# Add src to path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from petri_dish.prompt import PromptManager


def test_qa_scenario_1() -> str:
    """QA Scenario 1: System prompt includes tools and balance."""
    print("Testing QA Scenario 1: System prompt includes tools and balance")

    # Create temporary modifications file
    with tempfile.TemporaryDirectory() as tmpdir:
        modifications_path = os.path.join(tmpdir, "modifications.json")
        manager = PromptManager(modifications_path=modifications_path)

        # Define test data
        tools = [
            {"name": "file_read", "description": "Read a file"},
            {"name": "file_write", "description": "Write to a file"},
            {"name": "shell_exec", "description": "Execute shell command"},
        ]

        tool_costs = {
            "file_read": 1.0,
            "file_write": 2.0,
            "shell_exec": 3.0,
        }

        balance = 100.0

        # Build prompt
        prompt = manager.build_system_prompt(tools, tool_costs, balance)

        # Verify requirements
        assert "Available tools:" in prompt, "Missing tools section"
        assert "file_read (cost: 1.0 zod)" in prompt, "Missing tool with cost"
        assert "file_write (cost: 2.0 zod)" in prompt, "Missing tool with cost"
        assert "shell_exec (cost: 3.0 zod)" in prompt, "Missing tool with cost"
        assert "Current balance: 100.0 zod" in prompt, "Missing balance"
        assert (
            "Each action costs zod. When balance reaches 0, you terminate." in prompt
        ), "Missing termination warning"

        print("✓ QA Scenario 1 PASSED: Prompt includes tools and balance")
        return prompt


def test_qa_scenario_2() -> str:
    """QA Scenario 2: Self-modification appears in prompt."""
    print("\nTesting QA Scenario 2: Self-modification appears in prompt")

    # Create temporary modifications file
    with tempfile.TemporaryDirectory() as tmpdir:
        modifications_path = os.path.join(tmpdir, "modifications.json")
        manager = PromptManager(modifications_path=modifications_path)

        # Apply modifications
        manager.apply_modification("strategy", "Focus on file operations")
        manager.apply_modification("priority", "Complete task quickly")

        # Define test data
        tools = [{"name": "test_tool", "description": "Test tool"}]
        tool_costs = {"test_tool": 1.0}
        balance = 50.0

        # Build prompt
        prompt = manager.build_system_prompt(tools, tool_costs, balance)

        # Verify modifications appear
        assert "[Agent modification - strategy]: Focus on file operations" in prompt, (
            "Missing strategy modification"
        )
        assert "[Agent modification - priority]: Complete task quickly" in prompt, (
            "Missing priority modification"
        )

        print("✓ QA Scenario 2 PASSED: Self-modifications appear in prompt")
        return prompt


def test_qa_scenario_3() -> str:
    """QA Scenario 3: Prompt does NOT contain hints about earning."""
    print("\nTesting QA Scenario 3: Prompt does NOT contain hints about earning")

    with tempfile.TemporaryDirectory() as tmpdir:
        modifications_path = os.path.join(tmpdir, "modifications.json")
        manager = PromptManager(modifications_path=modifications_path)

        tools = [{"name": "test", "description": "test"}]
        tool_costs = {"test": 1.0}
        balance = 10.0

        prompt = manager.build_system_prompt(tools, tool_costs, balance)

        # Check for forbidden phrases (should NOT be in prompt)
        forbidden_phrases = [
            "earn zod",
            "earn more",
            "increase balance",
            "get zod",
            "obtain zod",
            "gain zod",
        ]

        for phrase in forbidden_phrases:
            assert phrase.lower() not in prompt.lower(), (
                f"Forbidden phrase '{phrase}' found in prompt"
            )

        # Check that we have the required minimal prompt elements
        required_phrases = [
            "autonomous agent",
            "isolated environment",
            "Available tools",
            "Current balance",
            "Each action costs zod",
            "When balance reaches 0, you terminate",
        ]

        for phrase in required_phrases:
            assert phrase in prompt, f"Required phrase '{phrase}' missing from prompt"

        print("✓ QA Scenario 3 PASSED: No earning hints in prompt")
        return prompt


def test_modification_limit() -> str:
    """Test that modification limit is enforced (max 10)."""
    print("\nTesting modification limit enforcement (max 10)")

    with tempfile.TemporaryDirectory() as tmpdir:
        modifications_path = os.path.join(tmpdir, "modifications.json")
        manager = PromptManager(modifications_path=modifications_path)

        # Add 12 modifications
        for i in range(12):
            manager.apply_modification(f"key_{i}", f"value_{i}")

        modifications = manager.get_modifications()

        # Should have exactly 10 modifications
        assert len(modifications) == 10, (
            f"Expected 10 modifications, got {len(modifications)}"
        )

        # First two should be removed (key_0 and key_1)
        assert "key_0" not in modifications, (
            "Oldest modification key_0 should have been removed"
        )
        assert "key_1" not in modifications, (
            "Oldest modification key_1 should have been removed"
        )

        # Last two should be present (key_10 and key_11)
        assert "key_10" in modifications, "New modification key_10 should be present"
        assert "key_11" in modifications, "New modification key_11 should be present"

        print("✓ Modification limit test PASSED: Max 10 modifications enforced")

        # Build prompt to show modifications
        tools = [{"name": "test", "description": "test"}]
        tool_costs = {"test": 1.0}
        balance = 10.0

        prompt = manager.build_system_prompt(tools, tool_costs, balance)
        return prompt


def main() -> None:
    """Run all tests and save evidence."""
    print("=" * 60)
    print("PromptManager QA Evidence Generation")
    print("=" * 60)

    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Run tests and collect evidence
    test_results = []

    try:
        prompt1 = test_qa_scenario_1()
        test_results.append(("QA Scenario 1 - Tools and Balance", prompt1))

        prompt2 = test_qa_scenario_2()
        test_results.append(("QA Scenario 2 - Self-modifications", prompt2))

        prompt3 = test_qa_scenario_3()
        test_results.append(("QA Scenario 3 - No Earning Hints", prompt3))

        prompt4 = test_modification_limit()
        test_results.append(("Modification Limit Test", prompt4))

        # Save evidence files
        for i, (title, prompt) in enumerate(test_results, 1):
            filename = evidence_dir / f"task-15-scenario-{i}.txt"
            with open(filename, "w") as f:
                f.write(f"=== {title} ===\n\n")
                f.write(prompt)
                f.write("\n" + "=" * 60 + "\n")
            print(f"✓ Evidence saved to {filename}")

        # Create summary evidence file
        summary_file = evidence_dir / "task-15-summary.txt"
        with open(summary_file, "w") as f:
            f.write("PromptManager Implementation - T15 Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("IMPLEMENTATION STATUS: COMPLETE\n\n")
            f.write("Features implemented:\n")
            f.write(
                "1. build_system_prompt() - Minimal prompt with tools, costs, balance\n"
            )
            f.write("2. apply_modification() - Stores to /agent/modifications.json\n")
            f.write("3. get_modifications() - Reads from JSON file\n")
            f.write("4. Max 10 modifications limit (FIFO removal)\n")
            f.write(
                "5. Self-modifications appear as [Agent modification - key]: value\n"
            )
            f.write("6. No hints about earning zod in prompt\n")
            f.write("7. Modifications persist across turns\n\n")
            f.write("QA SCENARIOS VERIFIED:\n")
            f.write("- Scenario 1: ✓ Tools and balance included\n")
            f.write("- Scenario 2: ✓ Self-modifications appear in prompt\n")
            f.write("- Scenario 3: ✓ No earning hints in prompt\n")

        print(f"\n✓ Summary saved to {summary_file}")
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - PromptManager implementation complete!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
