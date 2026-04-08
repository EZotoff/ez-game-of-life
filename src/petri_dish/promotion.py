from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from petri_dish.config import Settings

if TYPE_CHECKING:
    from petri_dish.logging_db import LoggingDB

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PromotionRule:
    rule_id: str
    claimed_pattern: str
    file_family: str
    hit_count: int
    promoted: bool
    promoted_at: datetime | None
    bonus_multiplier: float = 1.5
    created_at: datetime = field(default_factory=datetime.now)
    last_hit_at: datetime | None = None


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _rule_from_row(row: dict[str, Any]) -> PromotionRule:
    return PromotionRule(
        rule_id=str(row.get("rule_id", "")),
        claimed_pattern=str(row.get("claimed_pattern", "")),
        file_family=str(row.get("file_family", "")),
        hit_count=int(row.get("hit_count", 0)),
        promoted=bool(row.get("promoted", False)),
        promoted_at=_parse_datetime(row.get("promoted_at")),
        bonus_multiplier=float(row.get("bonus_multiplier", 1.5)),
        created_at=_parse_datetime(row.get("created_at")) or datetime.now(),
        last_hit_at=_parse_datetime(row.get("last_hit_at")),
    )


class PromotionEngine:
    def __init__(self, logging_db: "LoggingDB", settings: Settings):
        self._logging_db = logging_db
        self._settings = settings

    def record_hit(self, claimed_pattern: str, file_family: str) -> PromotionRule:
        rule = self._logging_db.record_scout_hit(
            claimed_pattern,
            file_family,
            bonus_multiplier=float(self._settings.promotion_bonus_multiplier),
        )
        if not self._settings.promotion_enabled:
            return rule

        if rule.promoted:
            return rule

        promoted_count = len(self._logging_db.get_promotion_rules(promoted_only=True))
        if promoted_count >= int(self._settings.max_promoted_rules):
            logger.warning(
                "Promotion cap reached (%s); skipping promotion for %s/%s",
                self._settings.max_promoted_rules,
                claimed_pattern,
                file_family,
            )
            return rule

        if rule.hit_count >= int(self._settings.promotion_threshold):
            self._logging_db.promote_rule(rule.rule_id)
            promoted_row = self._logging_db.get_promotion_rule(rule.rule_id)
            if promoted_row is not None:
                return _rule_from_row(promoted_row)
            rule.promoted = True
            rule.promoted_at = datetime.now()
        return rule

    def get_promoted_multiplier(self, claimed_pattern: str, file_family: str) -> float:
        rules = self._logging_db.get_promoted_rules_for_family(file_family)
        matched = [
            float(rule.get("bonus_multiplier", 1.0))
            for rule in rules
            if str(rule.get("claimed_pattern", "")) == claimed_pattern
        ]
        if not matched:
            return 1.0
        return max(matched)

    def get_promoted_rules_for_family(self, file_family: str) -> list[dict[str, Any]]:
        return self._logging_db.get_promoted_rules_for_family(file_family)

    def get_all_promoted_rules(self) -> list[PromotionRule]:
        rows = self._logging_db.get_promotion_rules(promoted_only=True)
        return [_rule_from_row(row) for row in rows]
