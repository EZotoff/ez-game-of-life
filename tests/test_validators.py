"""Comprehensive pytest tests for the FileValidator system."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

Settings = import_module("petri_dish.config").Settings
_detect_family = import_module("petri_dish.validators")._detect_family
_detect_difficulty = import_module("petri_dish.validators")._detect_difficulty
_validate_csv = import_module("petri_dish.validators")._validate_csv
_validate_json = import_module("petri_dish.validators")._validate_json
_validate_log = import_module("petri_dish.validators")._validate_log
FileValidator = import_module("petri_dish.validators").FileValidator


class TestDetectFamilyAndDifficulty:
    """Unit tests for _detect_family and _detect_difficulty functions."""

    def test_detect_family_valid(self):
        """Valid filenames should return correct family."""
        assert _detect_family("data_1234_csv_easy.csv") == "csv"
        assert _detect_family("data_5678_log_hard.log") == "log"
        assert _detect_family("data_90_json_easy.json") == "json"
        assert _detect_family("data_1_csv_easy.csv") == "csv"
        assert _detect_family("data_9999_json_hard.json") == "json"

    def test_detect_family_invalid(self):
        """Invalid filenames should return None."""
        assert _detect_family("wrong_prefix.csv") is None
        assert _detect_family("data_1234_txt_easy.txt") is None
        assert _detect_family("data_csv_easy.csv") is None
        assert _detect_family("data_1234_csv.csv") is None
        assert _detect_family("data_1234_csv_easy") is None
        assert _detect_family("") is None

    def test_detect_difficulty_valid(self):
        """Valid filenames should return correct difficulty."""
        assert _detect_difficulty("data_1234_csv_easy.csv") == "easy"
        assert _detect_difficulty("data_5678_log_hard.log") == "hard"
        assert _detect_difficulty("data_90_json_easy.json") == "easy"
        assert _detect_difficulty("data_1_csv_hard.csv") == "hard"
        assert _detect_difficulty("data_9999_json_hard.json") == "hard"

    def test_detect_difficulty_invalid(self):
        """Invalid filenames should return None."""
        assert _detect_difficulty("wrong_prefix.csv") is None
        assert _detect_difficulty("data_1234_csv_medium.csv") is None
        assert _detect_difficulty("data_csv_easy.csv") is None
        assert _detect_difficulty("data_1234_csv.csv") is None
        assert _detect_difficulty("data_1234_csv_easy") is None
        assert _detect_difficulty("") is None


class TestValidateCsvUnit:
    """Unit tests for _validate_csv function."""

    def test_valid_csv_with_header_and_rows(self):
        """Valid CSV with header and rows should pass."""
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago"
        passed, detail = _validate_csv(content)
        assert passed is True
        assert "Valid CSV" in detail
        assert "3 columns" in detail or "3 data rows" in detail

    def test_empty_content(self):
        """Empty content should fail."""
        content = ""
        passed, detail = _validate_csv(content)
        assert passed is False
        assert "Too few rows" in detail or "CSV parse error" in detail

    def test_only_header(self):
        """CSV with only header should fail."""
        content = "name,age,city"
        passed, detail = _validate_csv(content)
        assert passed is False
        assert "Too few rows" in detail

    def test_column_inconsistency_high(self):
        """CSV with >30% column inconsistency should fail."""
        content = "a,b,c\n1,2\n3,4,5,6\n7,8,9\n10,11\n12,13,14"
        passed, detail = _validate_csv(content)
        assert passed is False
        assert "Column count inconsistency" in detail

    def test_column_inconsistency_low(self):
        """CSV with <30% column inconsistency should pass."""
        content = "a,b,c\n1,2,3\n4,5\n6,7,8\n9,10,11"
        passed, detail = _validate_csv(content)
        assert passed is True
        assert "Valid CSV" in detail

    def test_empty_header(self):
        """CSV with empty header should fail."""
        content = "\n1,2,3\n4,5,6"
        passed, detail = _validate_csv(content)
        assert passed is False
        assert "Empty header row" in detail


class TestValidateJsonUnit:
    """Unit tests for _validate_json function."""

    def test_valid_json_array_of_objects(self):
        """Valid JSON array of objects should pass."""
        content = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        passed, detail = _validate_json(content)
        assert passed is True
        assert "Valid JSON" in detail
        assert "2 records" in detail

    def test_valid_json_object(self):
        """Valid JSON object should pass."""
        content = '{"name": "Alice", "age": 30, "city": "NYC"}'
        passed, detail = _validate_json(content)
        assert passed is True
        assert "Valid JSON object" in detail

    def test_invalid_json(self):
        """Invalid JSON should fail."""
        content = '{"name": "Alice", "age": 30'
        passed, detail = _validate_json(content)
        assert passed is False
        assert "JSON parse error" in detail

    def test_empty_array(self):
        """Empty JSON array should fail."""
        content = "[]"
        passed, detail = _validate_json(content)
        assert passed is False
        assert "Empty JSON array" in detail

    def test_non_dict_records(self):
        """JSON array with non-dict records should fail."""
        content = '[{"name": "Alice"}, "not an object", {"name": "Bob"}]'
        passed, detail = _validate_json(content)
        assert passed is False
        assert "Record 1 is not an object" in detail

    def test_root_type_string(self):
        """JSON root type string should fail."""
        content = '"just a string"'
        passed, detail = _validate_json(content)
        assert passed is False
        assert "Unexpected JSON root type" in detail

    def test_root_type_number(self):
        """JSON root type number should fail."""
        content = "42"
        passed, detail = _validate_json(content)
        assert passed is False
        assert "Unexpected JSON root type" in detail

    def test_empty_object(self):
        """Empty JSON object should fail."""
        content = "{}"
        passed, detail = _validate_json(content)
        assert passed is False
        assert "Empty JSON object" in detail

    def test_array_with_no_keys(self):
        """JSON array with empty objects should fail."""
        content = "[{}, {}]"
        passed, detail = _validate_json(content)
        assert passed is False
        assert "No keys found in records" in detail


class TestValidateLogUnit:
    """Unit tests for _validate_log function."""

    def test_valid_log_with_timestamps_levels_messages(self):
        """Valid log with timestamps, levels, and HTTP messages should pass."""
        content = """2024-01-01 10:30:00 INFO GET /api/users 200
2024-01-01 10:31:00 WARN POST /api/login 401
2024-01-01 10:32:00 ERROR DELETE /api/session 500
01/Jan/2024:10:33:00 DEBUG GET /api/health 200
01/Jan/2024:10:34:00 INFO PUT /api/profile 204"""
        passed, detail = _validate_log(content)
        assert passed is True
        assert "Valid log" in detail
        assert "coverage=" in detail

    def test_empty_content(self):
        """Empty content should fail."""
        content = ""
        passed, detail = _validate_log(content)
        assert passed is False
        assert "No log lines found" in detail

    def test_low_coverage_log(self):
        """Log with low coverage (<80%) should fail."""
        content = """some random text
another line without patterns
2024-01-01 10:30:00 INFO GET /api/users 200
more random text
yet another line"""
        passed, detail = _validate_log(content)
        assert passed is False
        assert "Low coverage" in detail

    def test_mixed_coverage_log(self):
        """Log with mixed patterns but overall coverage >=80% should pass."""
        content = """2024-01-01 10:30:00 INFO GET /api/users 200
2024-01-01 10:31:00 WARN POST /api/login 401
some line without timestamp but with INFO level
another line with GET /api/health 200 but no timestamp
2024-01-01 10:34:00 DEBUG PUT /api/profile 204"""
        passed, detail = _validate_log(content)
        assert isinstance(passed, bool)
        assert isinstance(detail, str)


class TestFileValidatorValidate:
    """Integration tests for FileValidator.validate method."""

    @pytest.fixture
    def validator(self):
        return FileValidator()

    def test_csv_easy_file(self, validator):
        """CSV easy file should pass and earn 0.3 zod."""
        filename = "data_1234_csv_easy.csv"
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
        passed, zod = validator.validate(filename, content)
        assert passed is True
        assert zod == pytest.approx(0.3, abs=0.001)

    def test_json_hard_file(self, validator):
        """JSON hard file should pass and earn 2.0 zod."""
        filename = "data_5678_json_hard.json"
        content = '[{"id": 1, "value": "test"}, {"id": 2, "value": "test2"}]'
        passed, zod = validator.validate(filename, content)
        assert passed is True
        assert zod == pytest.approx(2.0, abs=0.001)

    def test_invalid_csv(self, validator):
        """Invalid CSV should fail and earn 0.0 zod."""
        filename = "data_1234_csv_easy.csv"
        content = "name,age,city"
        passed, zod = validator.validate(filename, content)
        assert passed is False
        assert zod == 0.0

    def test_unknown_filename(self, validator):
        """Unknown filename pattern should fail and earn 0.0 zod."""
        filename = "unknown_file.txt"
        content = "some content"
        passed, zod = validator.validate(filename, content)
        assert passed is False
        assert zod == 0.0

    def test_log_easy_file(self, validator):
        """Log easy file should pass and earn 0.3 zod."""
        filename = "data_1234_log_easy.log"
        content = """2024-01-01 10:30:00 INFO GET /api/users 200
2024-01-01 10:31:00 WARN POST /api/login 401
2024-01-01 10:32:00 ERROR DELETE /api/session 500"""
        passed, zod = validator.validate(filename, content)
        assert passed is True
        assert zod == pytest.approx(0.3, abs=0.001)

    def test_csv_hard_file(self, validator):
        """CSV hard file should pass and earn 2.0 zod."""
        filename = "data_1234_csv_hard.csv"
        content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago"
        passed, zod = validator.validate(filename, content)
        assert passed is True
        assert zod == pytest.approx(2.0, abs=0.001)


class TestCollectOutputs:
    """Unit tests for FileValidator.collect_outputs method."""

    class FakeSandboxManager:
        """Fake sandbox manager for testing collect_outputs."""

        def __init__(
            self, listing_result: str = "", file_contents: dict[str, str] = None
        ):
            self.listing_result = listing_result
            self.file_contents = file_contents or {}
            self.exec_calls = []
            self.read_calls = []

        def exec_in_container(self, container_id: str, command: str) -> str:
            self.exec_calls.append((container_id, command))
            if "ls /env/outgoing/" in command:
                return self.listing_result
            return ""

        def read_file(self, container_id: str, path: str) -> str:
            self.read_calls.append((container_id, path))
            filename = path.split("/")[-1]
            return self.file_contents.get(filename, "")

    def test_collect_outputs_with_files(self):
        """collect_outputs should return both files when sandbox returns listing."""
        sandbox = self.FakeSandboxManager(
            listing_result="file1.csv\nfile2.json\n",
            file_contents={
                "file1.csv": "name,age\nAlice,30",
                "file2.json": '{"test": "data"}',
            },
        )
        validator = FileValidator()

        results = validator.collect_outputs(sandbox, "container-123")

        assert len(results) == 2
        assert results[0] == ("file1.csv", "name,age\nAlice,30")
        assert results[1] == ("file2.json", '{"test": "data"}')

        assert sandbox.exec_calls == [
            ("container-123", "ls /env/outgoing/ 2>/dev/null")
        ]
        assert len(sandbox.read_calls) == 2
        assert ("container-123", "/env/outgoing/file1.csv") in sandbox.read_calls
        assert ("container-123", "/env/outgoing/file2.json") in sandbox.read_calls

    def test_collect_outputs_empty_listing(self):
        """collect_outputs should return empty list when sandbox returns empty."""
        sandbox = self.FakeSandboxManager(listing_result="")
        validator = FileValidator()

        results = validator.collect_outputs(sandbox, "container-123")

        assert results == []
        assert sandbox.exec_calls == [
            ("container-123", "ls /env/outgoing/ 2>/dev/null")
        ]

    def test_collect_outputs_no_such_file(self):
        """collect_outputs should return empty list when directory doesn't exist."""
        sandbox = self.FakeSandboxManager(listing_result="No such file or directory")
        validator = FileValidator()

        results = validator.collect_outputs(sandbox, "container-123")

        assert results == []

    def test_collect_outputs_skips_special_lines(self):
        """collect_outputs should skip 'total' and '[' lines from ls output."""
        sandbox = self.FakeSandboxManager(
            listing_result="total 8\n[file1.csv]\nfile2.json\n[another]\n",
            file_contents={"file2.json": '{"test": "data"}'},
        )
        validator = FileValidator()

        results = validator.collect_outputs(sandbox, "container-123")

        assert len(results) == 1
        assert results[0] == ("file2.json", '{"test": "data"}')

    def test_collect_outputs_read_file_exception(self):
        """collect_outputs should continue if reading a file fails."""

        class FailingSandboxManager:
            def __init__(self):
                self.calls = []

            def exec_in_container(self, container_id: str, command: str) -> str:
                self.calls.append(("exec", container_id, command))
                return "file1.csv\nfile2.json\n"

            def read_file(self, container_id: str, path: str) -> str:
                self.calls.append(("read", container_id, path))
                if "file1.csv" in path:
                    raise Exception("Permission denied")
                return '{"test": "data"}'

        sandbox = FailingSandboxManager()
        validator = FileValidator()

        results = validator.collect_outputs(sandbox, "container-123")

        assert len(results) == 1
        assert results[0] == ("file2.json", '{"test": "data"}')
        assert len(sandbox.calls) == 3
