"""Schema introspection and lexical schema retrieval (no embeddings, no extra deps)."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .sandbox import connect_ro, is_duckdb

_TOK = re.compile(r"[a-z0-9]+")
_STOP = frozenset("the a an of in on for to and or is are was were what which who how many much list name "
                  "with by from that this give show all each per than has have had".split())


def toks(s: str) -> set[str]:
    out = set()
    for w in _TOK.findall(re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()):
        if w not in _STOP:
            out.add(w)
            if w.endswith("s") and len(w) > 3:
                out.add(w[:-1])
    return out


@dataclass
class Table:
    name: str
    cols: list[tuple[str, str, bool]]  # name, type, is_pk
    fks: list[tuple[str, str, str]] = field(default_factory=list)  # col, ref_table, ref_col
    samples: dict[str, list] = field(default_factory=dict)

    def ddl(self, only: set[str] | None = None) -> str:
        lines = [f'  "{c}" {t}{" PRIMARY KEY" if pk else ""}' + (f"  -- e.g. {self.samples[c]}" if c in self.samples else "")
                 for c, t, pk in self.cols if only is None or c in only or pk]
        lines += [f'  FOREIGN KEY ("{c}") REFERENCES "{rt}"("{rc}")' for c, rt, rc in self.fks
                  if only is None or c in only]
        return f'CREATE TABLE "{self.name}" (\n' + ",\n".join(lines) + "\n);"


# Home Credit join keys (the parquet views carry no declared constraints)
HOMECREDIT_FKS = {
    "bureau": [("SK_ID_CURR", "application_train", "SK_ID_CURR")],
    "bureau_balance": [("SK_ID_BUREAU", "bureau", "SK_ID_BUREAU")],
    "previous_application": [("SK_ID_CURR", "application_train", "SK_ID_CURR")],
    "pos_cash_balance": [("SK_ID_PREV", "previous_application", "SK_ID_PREV")],
    "credit_card_balance": [("SK_ID_PREV", "previous_application", "SK_ID_PREV")],
    "installments_payments": [("SK_ID_PREV", "previous_application", "SK_ID_PREV")],
}


def introspect_duckdb(db_path: str | Path, n_samples: int = 2) -> dict[str, Table]:
    """Schema loader for the DuckDB demo target. Samples come from one small LIMIT scan per table."""
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    out: dict[str, Table] = {}
    try:
        names = [r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main' ORDER BY 1").fetchall()]
        for n in names:
            t = out[n] = Table(n, [])
            t.cols = [(r[0], r[1], False) for r in con.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema='main' AND table_name=? ORDER BY ordinal_position", [n]).fetchall()]
            t.fks = [f for f in HOMECREDIT_FKS.get(n, []) if f[1] in names]
            cur = con.execute(f'SELECT * FROM "{n}" LIMIT 200')
            head = cur.fetchall()
            for i, (c, _, _) in enumerate(t.cols):
                vals = list(dict.fromkeys(r[i] for r in head if r[i] is not None))[:n_samples]
                t.samples[c] = [v[:30] if isinstance(v, str) else (str(v) if not isinstance(v, (int, float)) else v)
                                for v in vals]
    finally:
        con.close()
    return out


def introspect(db_path: str | Path, n_samples: int = 2) -> dict[str, Table]:
    if is_duckdb(db_path):
        return introspect_duckdb(db_path, n_samples)
    con = connect_ro(db_path)
    out: dict[str, Table] = {}
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        con.set_authorizer(None)  # introspection only, connection is still mode=ro + query_only
        for n in names:
            t = out[n] = Table(n, [])
            t.cols = [(r[1], r[2] or "", bool(r[5])) for r in con.execute(f'PRAGMA table_info("{n}")')]
            t.fks = [(r[3], r[2], r[4]) for r in con.execute(f'PRAGMA foreign_key_list("{n}")')]
            for c, _, _ in t.cols:
                try:
                    vals = [r[0] for r in con.execute(
                        f'SELECT DISTINCT "{c}" FROM "{n}" WHERE "{c}" IS NOT NULL LIMIT {n_samples}')]
                    t.samples[c] = [v[:30] if isinstance(v, str) else v for v in vals]
                except sqlite3.Error:
                    pass
    finally:
        con.close()
    return out


def render(schema: dict[str, Table], keep: dict[str, set[str]] | None = None) -> str:
    return "\n\n".join(t.ddl(keep.get(n) if keep else None) for n, t in schema.items() if keep is None or n in keep)


def retrieve(schema: dict[str, Table], question: str, evidence: str = "",
             top_tables: int = 4, top_cols: int = 12) -> dict[str, set[str]]:
    """Score tables and columns by token overlap with question+evidence, then add FK neighbours
    so joins stay possible. Returns {table: columns to show}."""
    q = toks(question + " " + evidence)
    tscore = {n: len(q & toks(n)) * 2 + sum(len(q & toks(c)) for c, _, _ in t.cols) for n, t in schema.items()}
    ranked = sorted(schema, key=lambda n: -tscore[n])
    chosen = [n for n in ranked[:top_tables] if tscore[n] > 0] or ranked[:top_tables]
    for n in list(chosen):  # FK closure, one hop
        for _, rt, _ in schema[n].fks:
            if rt in schema and rt not in chosen:
                chosen.append(rt)
    keep: dict[str, set[str]] = {}
    for n in chosen:
        t = schema[n]
        cs = sorted(t.cols, key=lambda c: -len(q & toks(c[0])))
        cols = {c for c, _, _ in cs[:top_cols]}
        cols |= {c for c, _, _ in t.fks} | {c for c, _, pk in t.cols if pk}
        cols |= {c for c, _, _ in t.cols if len(q & toks(c)) > 0}
        keep[n] = cols
    # FK columns pointing into chosen tables must exist on both sides
    for n in chosen:
        for c, rt, rc in schema[n].fks:
            if rt in keep:
                keep[rt].add(rc)
    return keep
