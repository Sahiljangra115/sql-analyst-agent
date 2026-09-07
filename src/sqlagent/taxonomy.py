"""Heuristic failure taxonomy. Compares a wrong prediction to the gold SQL (when known) and to the
execution outcome. Heuristic by design: regex on SQL text, not a full parser. Spot-check before quoting."""
from __future__ import annotations

import re

CATEGORIES = ["wrong_join", "wrong_filter", "hallucinated_column", "wrong_aggregation", "timeout",
              "syntax_error", "sandbox_rejected", "other_wrong_result"]
_STR = re.compile(r"'(?:[^']|'')*'")
_AGG = re.compile(r"\b(count|sum|avg|min|max|total)\s*\(", re.I)


def _tables(sql: str) -> set[str]:
    return {m.lower() for m in re.findall(r"\b(?:from|join)\s+[`\"\[]?(\w+)", sql, re.I)}


def _aggs(sql: str) -> list[str]:
    s = _STR.sub("''", sql)
    extra = [k for k in ("group by", "distinct", "having") if k in s.lower()]
    return sorted([m.lower() for m in _AGG.findall(s)] + extra)


def _filters(sql: str) -> set[str]:
    sql = sql.strip().rstrip(";")  # a trailing ; used to make every last filter differ
    m = re.search(r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|$)", sql, re.I | re.S)
    if not m:
        return set()
    body = re.sub(r"\b\w+\.", "", m.group(1))  # drop table aliases
    parts = re.split(r"\band\b|\bor\b", body, flags=re.I)
    return {re.sub(r"[\s`\"\[\]()]", "", p).lower() for p in parts if p.strip()}


def classify(pred_sql: str, kind: str, error: str = "", gold_sql: str | None = None) -> str:
    """kind: sandbox ExecResult.kind of the prediction. Call only for incorrect answers."""
    if kind == "timeout":
        return "timeout"
    if kind == "rejected":
        return "sandbox_rejected"
    if kind == "error":
        e = error.lower()
        if "no such column" in e or "no such table" in e or "ambiguous column" in e:
            return "hallucinated_column"
        return "syntax_error"  # includes made-up functions (DIVIDE, SUBTRACT copied from BIRD evidence)
    if not gold_sql:
        return "other_wrong_result"
    if _tables(pred_sql) != _tables(gold_sql):
        return "wrong_join"
    if len(re.findall(r"\bjoin\b", pred_sql, re.I)) != len(re.findall(r"\bjoin\b", gold_sql, re.I)):
        return "wrong_join"
    if _aggs(pred_sql) != _aggs(gold_sql):
        return "wrong_aggregation"
    if _filters(pred_sql) != _filters(gold_sql):
        return "wrong_filter"
    return "other_wrong_result"
