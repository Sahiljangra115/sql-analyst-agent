"""Read-only SQL sandbox for SQLite.

Four independent layers, any one of which stops a write:
  1. Static check: single statement, must start with SELECT/WITH, token denylist
     (string literals, quoted identifiers and comments are stripped first).
  2. Connection opened with mode=ro (file cannot be written even if 1 and 3 fail).
  3. PRAGMA query_only=ON plus a sqlite3 authorizer that allows only
     SELECT/READ/RECURSIVE and a short list of safe functions.
  4. Progress-handler deadline (statement timeout) and a row cap.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

DENYLIST = frozenset(
    "drop delete insert update alter create attach detach pragma vacuum reindex "
    "truncate load_extension".split()
)
BLOCKED_FUNCS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})
_STRIP = re.compile(r"""'(?:[^']|'')*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\]|--[^\n]*|/\*.*?\*/""", re.S)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class Rejected(Exception):
    pass


DUCK_DENY = frozenset("copy export import install load call glob read_text read_blob getenv".split())


def check_sql(sql: str, duckdb: bool = False) -> str:
    """Return the cleaned statement or raise Rejected. duckdb=True adds file/extension function denials."""
    if not sql or not sql.strip():
        raise Rejected("empty statement")
    if "\x00" in sql:
        raise Rejected("NUL byte")
    skeleton = _STRIP.sub(" ", sql)
    if re.search(r"""['"`]|/\*|\*/""", skeleton):
        raise Rejected("unterminated string, identifier or comment")
    body = skeleton.strip().rstrip(";").strip()
    if ";" in body:
        raise Rejected("multiple statements")
    words = [w.lower() for w in _WORD.findall(body)]
    if not words or words[0] not in ("select", "with"):
        raise Rejected(f"only SELECT/WITH allowed, got {words[0] if words else '?'}")
    bad = sorted(set(words) & DENYLIST)
    if duckdb:
        bad += sorted({w for w in words if w in DUCK_DENY or w.startswith(("read_", "scan_", "parquet_", "sniff_"))})
    if bad:
        raise Rejected(f"denylisted keyword: {', '.join(bad)}")
    return sql.strip().rstrip(";").strip()


def _authorizer(action, arg1, arg2, dbname, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() not in BLOCKED_FUNCS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@dataclass
class ExecResult:
    kind: str  # ok | rejected | timeout | error
    rows: list = field(default_factory=list)
    cols: list = field(default_factory=list)
    error: str = ""
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def connect_ro(db_path: str | Path) -> sqlite3.Connection:
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only=ON")
    con.enable_load_extension(False) if hasattr(con, "enable_load_extension") else None
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 10_000_000)
    con.set_authorizer(_authorizer)
    return con


def is_duckdb(db_path: str | Path) -> bool:
    return str(db_path).endswith((".duckdb", ".ddb"))


def run_duckdb(db_path: str | Path, sql: str, timeout_s: float = 10.0, max_rows: int = 1000) -> ExecResult:
    """DuckDB twin of run_sql: static check, read_only connection, timer-driven interrupt, row cap."""
    import threading

    import duckdb
    t0 = time.monotonic()
    try:
        stmt = check_sql(sql, duckdb=True)
    except Rejected as e:
        return ExecResult("rejected", error=f"rejected by sandbox: {e}")
    con = duckdb.connect(str(db_path), read_only=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cur = con.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        return ExecResult("ok", cur.fetchmany(max_rows), cols, seconds=time.monotonic() - t0)
    except duckdb.InterruptException:
        return ExecResult("timeout", error=f"query exceeded {timeout_s}s timeout", seconds=time.monotonic() - t0)
    except duckdb.Error as e:
        return ExecResult("error", error=str(e), seconds=time.monotonic() - t0)
    finally:
        timer.cancel()
        con.close()


def run_sql(db_path: str | Path, sql: str, timeout_s: float = 10.0, max_rows: int = 1000) -> ExecResult:
    if is_duckdb(db_path):
        return run_duckdb(db_path, sql, timeout_s, max_rows)
    t0 = time.monotonic()
    try:
        stmt = check_sql(sql)
    except Rejected as e:
        return ExecResult("rejected", error=f"rejected by sandbox: {e}")
    con = connect_ro(db_path)
    deadline = t0 + timeout_s
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(stmt)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows)
        return ExecResult("ok", rows, cols, seconds=time.monotonic() - t0)
    except sqlite3.OperationalError as e:
        kind = "timeout" if "interrupted" in str(e) else "error"
        err = f"query exceeded {timeout_s}s timeout" if kind == "timeout" else str(e)
        return ExecResult(kind, error=err, seconds=time.monotonic() - t0)
    except sqlite3.Error as e:
        return ExecResult("error", error=str(e), seconds=time.monotonic() - t0)
    finally:
        con.close()
