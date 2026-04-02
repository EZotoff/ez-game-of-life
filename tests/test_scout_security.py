from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

build_scout_report = import_module("petri_dish.scout_report").build_scout_report
GamingDetector = import_module("petri_dish.scout_validator").GamingDetector


def _make_report(
    *,
    query: str,
    confidence: float,
    snippet: str = "public docs",
):
    return build_scout_report(
        requesting_agent_id="agent-sec",
        claimed_pattern="pattern",
        output_summary="summary",
        file_family="log",
        queries_executed=[query],
        result_snippets=[snippet],
        confidence=confidence,
        verdict="supported",
        reasoning="security test",
        suggested_bonus=0.1,
    )


def test_detects_repeated_query_patterns() -> None:
    detector = GamingDetector(repeated_query_threshold=3)
    reports = [
        _make_report(query="json schema validate", confidence=0.6),
        _make_report(query="json schema validate", confidence=0.7),
        _make_report(query="json schema validate", confidence=0.8),
        _make_report(query="json schema validate", confidence=0.9),
    ]
    alerts = detector.detect_suspicious_patterns(
        reports,
        blocklisted_domains=["localhost", "internal.local"],
    )
    repeated = [a for a in alerts if a.type == "repeated_query"]
    assert len(repeated) == 1
    assert repeated[0].severity == "warning"
    assert "repeated 4 times" in repeated[0].detail


def test_detects_suspicious_confidence_concentration() -> None:
    detector = GamingDetector(
        high_confidence_threshold=0.95,
        high_confidence_ratio_threshold=0.8,
        min_reports_for_confidence_check=5,
    )
    reports = [
        _make_report(query=f"q{i}", confidence=0.97 if i < 5 else 0.4) for i in range(6)
    ]
    alerts = detector.detect_suspicious_patterns(
        reports,
        blocklisted_domains=["localhost"],
    )
    high_conf = [a for a in alerts if a.type == "suspicious_confidence"]
    assert len(high_conf) == 1
    assert high_conf[0].severity == "critical"
    assert "5/6" in high_conf[0].detail


def test_detects_blocklisted_domain_evidence_in_snippets_and_queries() -> None:
    detector = GamingDetector()
    reports = [
        _make_report(
            query="inspect https://localhost/admin",
            confidence=0.7,
            snippet="example text",
        ),
        _make_report(
            query="normal query",
            confidence=0.7,
            snippet="found note at https://debug.internal.local/path",
        ),
    ]
    alerts = detector.detect_suspicious_patterns(
        reports,
        blocklisted_domains=["localhost", "internal.local"],
    )
    blocked = [a for a in alerts if a.type == "blocklisted_domain"]
    assert len(blocked) == 2
    assert all(a.severity == "critical" for a in blocked)
    details = [a.detail for a in blocked]
    assert any("localhost" in detail for detail in details)
    assert any("internal.local" in detail for detail in details)
