from __future__ import annotations

from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass
import re
from statistics import mean
from typing import Callable, Protocol

from petri_dish.scout_report import FileFamily, ScoutReport


class _ScoutReportStore(Protocol):
    def get_scout_reports(
        self,
        run_id: str,
        *,
        agent_id: str | None = None,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class ReplayResult:
    consistent: bool
    avg_confidence: float
    confidence_variance: float
    promotion_recommended: bool
    replay_count: int
    verdicts: list[str]
    replay_queries: list[str]


@dataclass(frozen=True)
class GeneralizationResult:
    pattern: str
    file_family: FileFamily
    queries_used: list[str]
    verdicts: list[str]
    avg_confidence: float
    supportive_ratio: float
    generalized: bool


@dataclass(frozen=True)
class PromotionCandidate:
    report_id: str
    file_family: FileFamily
    pattern: str
    evidence_summary: str
    replay_result: ReplayResult
    generalization_result: GeneralizationResult
    risk_assessment: str
    promotion_candidate: bool


@dataclass(frozen=True)
class GamingAlert:
    type: str
    severity: str
    detail: str


_URL_RE = re.compile(r"https?://([^/\s\]\)]+)", re.IGNORECASE)
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,63}\b", re.IGNORECASE)


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = mean(values)
    return sum((v - avg) ** 2 for v in values) / len(values)


def _extract_hosts(text: str) -> list[str]:
    hosts: list[str] = []
    for match in _URL_RE.finditer(text):
        host = match.group(1).split(":", 1)[0].lower().rstrip(".")
        if host:
            hosts.append(host)
    for match in _HOST_RE.finditer(text):
        hosts.append(match.group(0).lower().rstrip("."))
    return hosts


def _is_blocklisted_host(host: str, blocklisted_domains: Iterable[str]) -> bool:
    normalized = host.lower().rstrip(".")
    for blocked_raw in blocklisted_domains:
        blocked = blocked_raw.lower().strip().lstrip("*.")
        if not blocked:
            continue
        if normalized == blocked or normalized.endswith(f".{blocked}"):
            return True
    return False


def scout_reports_from_db(
    logging_db: _ScoutReportStore,
    run_id: str,
    *,
    agent_id: str | None = None,
) -> list[ScoutReport]:
    rows = logging_db.get_scout_reports(run_id, agent_id=agent_id)
    reports: list[ScoutReport] = []
    for row in rows:
        raw_json = row.get("report_json")
        if isinstance(raw_json, str) and raw_json.strip():
            reports.append(ScoutReport.from_json(raw_json))
            continue

        report_payload = {
            "report_id": row.get("report_id", ""),
            "requesting_agent_id": row.get("requesting_agent_id", "unknown"),
            "timestamp": row.get("timestamp", "1970-01-01T00:00:00+00:00"),
            "claimed_pattern": row.get("claimed_pattern", ""),
            "output_summary": row.get("output_summary", ""),
            "file_family": row.get("file_family", "csv"),
            "queries_executed": row.get("queries_executed", []),
            "results_found": row.get("results_found", 0),
            "result_snippets": row.get("result_snippets", []),
            "confidence": row.get("confidence", 0.0),
            "verdict": row.get("verdict", "no_sources"),
            "reasoning": row.get("reasoning", ""),
            "suggested_bonus": row.get("suggested_bonus", 0.0),
        }
        reports.append(ScoutReport.from_dict(report_payload))
    return reports


class ScoutReplayValidator:
    def __init__(
        self,
        execute_scout: Callable[[str, str, FileFamily], ScoutReport],
        *,
        min_avg_confidence: float = 0.85,
        max_confidence_variance: float = 0.05,
    ) -> None:
        self._execute_scout: Callable[[str, str, FileFamily], ScoutReport] = (
            execute_scout
        )
        self._min_avg_confidence: float = min_avg_confidence
        self._max_confidence_variance: float = max_confidence_variance

    def validate_replay(
        self,
        original_report: ScoutReport,
        num_replay_queries: int = 3,
    ) -> ReplayResult:
        paraphrased = self._paraphrase_queries(original_report.queries_executed)
        replay_queries = paraphrased[: max(0, num_replay_queries)]
        replay_reports = [
            self._execute_scout(
                query,
                original_report.claimed_pattern,
                original_report.file_family,
            )
            for query in replay_queries
        ]

        verdicts = [report.verdict for report in replay_reports]
        confidences = [report.confidence for report in replay_reports]
        avg_confidence = mean(confidences) if confidences else 0.0
        conf_variance = _variance(confidences)
        consistent = (
            bool(verdicts) and len(set(verdicts)) == 1 and verdicts[0] == "supported"
        )
        promotion_recommended = (
            consistent
            and avg_confidence >= self._min_avg_confidence
            and conf_variance <= self._max_confidence_variance
        )

        return ReplayResult(
            consistent=consistent,
            avg_confidence=avg_confidence,
            confidence_variance=conf_variance,
            promotion_recommended=promotion_recommended,
            replay_count=len(replay_reports),
            verdicts=verdicts,
            replay_queries=replay_queries,
        )

    def check_generalization(
        self,
        pattern: str,
        file_family: FileFamily,
    ) -> GeneralizationResult:
        queries = [
            f"{file_family} format examples",
            f"{file_family} validation rules",
            f"{file_family} best practices",
        ]
        reports = [
            self._execute_scout(query, pattern, file_family) for query in queries
        ]

        verdicts = [report.verdict for report in reports]
        confidences = [report.confidence for report in reports]
        supported_count = sum(1 for verdict in verdicts if verdict == "supported")
        supportive_ratio = supported_count / len(reports) if reports else 0.0
        avg_confidence = mean(confidences) if confidences else 0.0
        generalized = supportive_ratio >= 2 / 3 and avg_confidence >= 0.7

        return GeneralizationResult(
            pattern=pattern,
            file_family=file_family,
            queries_used=queries,
            verdicts=verdicts,
            avg_confidence=avg_confidence,
            supportive_ratio=supportive_ratio,
            generalized=generalized,
        )

    def generate_promotion_candidate(
        self,
        report: ScoutReport,
        replay_result: ReplayResult,
        generalization_result: GeneralizationResult,
    ) -> PromotionCandidate:
        promotion_candidate = (
            replay_result.promotion_recommended and generalization_result.generalized
        )
        risk_assessment = (
            "low"
            if promotion_candidate
            else "medium"
            if replay_result.consistent
            else "high"
        )
        evidence_summary = (
            f"report={report.report_id}, verdict={report.verdict}, "
            f"confidence={report.confidence:.2f}, "
            f"replay_consistent={replay_result.consistent}, "
            f"generalized={generalization_result.generalized}"
        )

        return PromotionCandidate(
            report_id=report.report_id,
            file_family=report.file_family,
            pattern=report.claimed_pattern,
            evidence_summary=evidence_summary,
            replay_result=replay_result,
            generalization_result=generalization_result,
            risk_assessment=risk_assessment,
            promotion_candidate=promotion_candidate,
        )

    @staticmethod
    def _paraphrase_queries(queries: list[str]) -> list[str]:
        expanded: list[str] = []
        for query in queries:
            normalized = " ".join(query.split())
            if not normalized:
                continue
            expanded.append(normalized)
            expanded.append(f"best practice {normalized}")
            expanded.append(f"official guidance {normalized}")
        if not expanded:
            return [
                "validation best practices",
                "format requirements",
                "schema guidance",
            ]
        deduped: list[str] = []
        seen: set[str] = set()
        for query in expanded:
            key = _normalize_query(query)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped


class GamingDetector:
    def __init__(
        self,
        *,
        repeated_query_threshold: int = 3,
        high_confidence_threshold: float = 0.95,
        high_confidence_ratio_threshold: float = 0.8,
        min_reports_for_confidence_check: int = 5,
    ) -> None:
        self._repeated_query_threshold: int = repeated_query_threshold
        self._high_confidence_threshold: float = high_confidence_threshold
        self._high_confidence_ratio_threshold: float = high_confidence_ratio_threshold
        self._min_reports_for_confidence_check: int = min_reports_for_confidence_check

    def detect_suspicious_patterns(
        self,
        reports: list[ScoutReport],
        *,
        blocklisted_domains: list[str],
    ) -> list[GamingAlert]:
        alerts: list[GamingAlert] = []
        alerts.extend(self._detect_repeated_queries(reports))
        alerts.extend(self._detect_confidence_concentration(reports))
        alerts.extend(
            self._detect_blocklisted_domain_evidence(reports, blocklisted_domains)
        )
        return alerts

    def _detect_repeated_queries(self, reports: list[ScoutReport]) -> list[GamingAlert]:
        normalized_queries: list[str] = []
        for report in reports:
            normalized_queries.extend(
                _normalize_query(q) for q in report.queries_executed if q.strip()
            )
        counts = Counter(normalized_queries)
        alerts: list[GamingAlert] = []
        for query, count in sorted(counts.items()):
            if count > self._repeated_query_threshold:
                alerts.append(
                    GamingAlert(
                        type="repeated_query",
                        severity="warning",
                        detail=f"Query repeated {count} times: {query[:80]}",
                    )
                )
        return alerts

    def _detect_confidence_concentration(
        self,
        reports: list[ScoutReport],
    ) -> list[GamingAlert]:
        if len(reports) < self._min_reports_for_confidence_check:
            return []
        high_conf_count = sum(
            1
            for report in reports
            if report.confidence >= self._high_confidence_threshold
        )
        ratio = high_conf_count / len(reports)
        if ratio >= self._high_confidence_ratio_threshold:
            return [
                GamingAlert(
                    type="suspicious_confidence",
                    severity="critical",
                    detail=(
                        f"High-confidence concentration {high_conf_count}/{len(reports)} "
                        f"({ratio:.0%})"
                    ),
                )
            ]
        return []

    def _detect_blocklisted_domain_evidence(
        self,
        reports: list[ScoutReport],
        blocklisted_domains: list[str],
    ) -> list[GamingAlert]:
        offending_hosts: set[str] = set()
        for report in reports:
            for text in [*report.queries_executed, *report.result_snippets]:
                for host in _extract_hosts(text):
                    if _is_blocklisted_host(host, blocklisted_domains):
                        offending_hosts.add(host)
        return [
            GamingAlert(
                type="blocklisted_domain",
                severity="critical",
                detail=f"Blocklisted domain evidence detected: {host}",
            )
            for host in sorted(offending_hosts)
        ]
