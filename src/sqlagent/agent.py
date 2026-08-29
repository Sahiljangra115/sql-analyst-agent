"""LangGraph workflow: retrieve schema -> plan -> write SQL -> execute -> (bounded) self-correct."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import schema as sch
from . import trace
from .llm import LLM
from .sandbox import ExecResult, is_duckdb, run_sql

SYSTEM = ("You are a careful data analyst writing {dialect} queries. Use only tables and columns present in the "
          "schema. Return exactly one read-only SELECT statement, nothing else, inside a ```sql fenced block.")
SUMMARY_SYSTEM = "You are a data analyst. Answer in plain prose from the given query result. Do not write SQL."
DIALECTS = {False: "SQLite", True: "DuckDB"}


@dataclass
class Config:
    retrieval: bool = True
    use_plan: bool = True
    max_retries: int = 2  # 0 disables self-correction
    use_evidence: bool = True
    timeout_s: float = 10.0
    max_rows: int = 1000
    summarize: bool = False


class State(TypedDict, total=False):
    question: str
    evidence: str
    schema_text: str
    plan: str
    sql: str
    result: ExecResult
    attempts: int
    history: list
    events: list
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cost: float
    answer: str


@dataclass
class Answer:
    sql: str
    result: ExecResult
    attempts: int
    history: list  # [(sql, kind, error)]
    events: list
    llm_calls: int
    tokens_in: int
    tokens_out: int
    cost: float
    steps: int
    summary: str = ""


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sqlite|sql)?\s*(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:sqlite|sql)?\s*(.*)", text, re.S | re.I)  # unterminated fence
    s = (m.group(1) if m else text).strip()
    m = re.search(r"\b(select|with)\b.*", s, re.S | re.I)
    return (m.group(0) if m else s).strip()


class Agent:
    def __init__(self, llm: LLM, db_path: str | Path, cfg: Config | None = None, schema=None):
        self.llm, self.db, self.cfg = llm, str(db_path), cfg or Config()
        self.schema = schema or sch.introspect(db_path)
        self.system = SYSTEM.format(dialect=DIALECTS[is_duckdb(db_path)])
        self.graph = self._build()

    # --- helpers
    def _call(self, st: State, node: str, user: str, system: str | None = None) -> str:
        with trace.span(node, input=user, generation=True, model=self.llm.name) as done:
            r = self.llm.complete(system or self.system, user)
            done(r.text, {"input": r.tokens_in, "output": r.tokens_out})
        st["llm_calls"] = st.get("llm_calls", 0) + 1
        st["tokens_in"] = st.get("tokens_in", 0) + r.tokens_in
        st["tokens_out"] = st.get("tokens_out", 0) + r.tokens_out
        st["cost"] = st.get("cost", 0.0) + (r.tokens_in * self.llm.price_in + r.tokens_out * self.llm.price_out) / 1e6
        return r.text

    def _ev(self, st: State, node: str, t0: float, **detail):
        st["events"] = st.get("events", []) + [{"node": node, "s": round(time.monotonic() - t0, 4), **detail}]

    def _ctx(self, st: State) -> str:
        ev = f"\nEvidence: {st['evidence']}" if self.cfg.use_evidence and st.get("evidence") else ""
        return f"Schema:\n{st['schema_text']}\n\nQuestion: {st['question']}{ev}"

    # --- nodes
    def _retrieve(self, st: State) -> State:
        t0 = time.monotonic()
        with trace.span("retrieve_schema", input=st["question"]) as done:
            ev = st.get("evidence", "") if self.cfg.use_evidence else ""
            keep = sch.retrieve(self.schema, st["question"], ev) if self.cfg.retrieval else None
            st["schema_text"] = sch.render(self.schema, keep)
            done(sorted(keep) if keep else "full schema")
        self._ev(st, "retrieve", t0, tables=len(keep) if keep else len(self.schema))
        return st

    def _plan(self, st: State) -> State:
        t0 = time.monotonic()
        st["plan"] = self._call(st, "plan", self._ctx(st) +
                                "\n\nWrite a short numbered plan (tables to join, filters, aggregation). No SQL yet.")
        self._ev(st, "plan", t0)
        return st

    def _write(self, st: State) -> State:
        t0 = time.monotonic()
        plan = f"\n\nPlan:\n{st['plan']}" if st.get("plan") else ""
        st["sql"] = extract_sql(self._call(st, "write_sql", self._ctx(st) + plan + "\n\nSQL:"))
        self._ev(st, "write", t0)
        return st

    def _execute(self, st: State) -> State:
        t0 = time.monotonic()
        with trace.span("execute", input=st["sql"]) as done:
            r = run_sql(self.db, st["sql"], self.cfg.timeout_s, self.cfg.max_rows)
            done({"kind": r.kind, "error": r.error, "rows": len(r.rows)})
        st["result"] = r
        st["history"] = st.get("history", []) + [(st["sql"], r.kind, r.error)]
        self._ev(st, "execute", t0, kind=r.kind, rows=len(r.rows))
        return st

    def _route(self, st: State) -> str:
        r = st["result"]
        if r.ok and r.rows:
            return "done"
        if st.get("attempts", 0) >= self.cfg.max_retries:
            return "done"
        return "correct"

    def _correct(self, st: State) -> State:
        t0 = time.monotonic()
        r = st["result"]
        problem = r.error if not r.ok else "The query ran but returned zero rows. Re-check filter values and joins."
        if r.kind == "timeout":
            problem += " Avoid cross joins and unindexed correlated subqueries."
        st["attempts"] = st.get("attempts", 0) + 1
        user = (self._ctx(st) + f"\n\nPrevious SQL:\n{st['sql']}\n\nProblem: {problem}\n\n"
                "Write a corrected SQL query.\nSQL:")
        st["sql"] = extract_sql(self._call(st, "correct", user))
        self._ev(st, "correct", t0, attempt=st["attempts"])
        return st

    def _summarize(self, st: State) -> State:
        t0 = time.monotonic()
        r = st["result"]
        user = (f"Question: {st['question']}\nSQL: {st['sql']}\nResult (first rows): {r.rows[:20]}\n"
                "Answer the question in one or two sentences using only this result.")
        st["answer"] = re.sub(r"```.*?```", "", self._call(st, "summarize", user, SUMMARY_SYSTEM), flags=re.S).strip()
        self._ev(st, "summarize", t0)
        return st

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("write", self._write)
        g.add_node("execute", self._execute)
        g.add_node("correct", self._correct)
        g.add_edge(START, "retrieve")
        if self.cfg.use_plan:
            g.add_node("plan", self._plan)
            g.add_edge("retrieve", "plan")
            g.add_edge("plan", "write")
        else:
            g.add_edge("retrieve", "write")
        g.add_edge("write", "execute")
        g.add_edge("correct", "execute")
        end = "summarize" if self.cfg.summarize else END
        if self.cfg.summarize:
            g.add_node("summarize", self._summarize)
            g.add_edge("summarize", END)
        g.add_conditional_edges("execute", self._route, {"done": end, "correct": "correct"})
        return g.compile()

    def ask(self, question: str, evidence: str = "") -> Answer:
        with trace.span("question", input=question):
            st = self.graph.invoke({"question": question, "evidence": evidence, "attempts": 0,
                                    "events": [], "history": [], "llm_calls": 0, "tokens_in": 0,
                                    "tokens_out": 0, "cost": 0.0}, {"recursion_limit": 50})
        trace.flush()
        return Answer(st["sql"], st["result"], st["attempts"], st["history"], st["events"], st["llm_calls"],
                      st["tokens_in"], st["tokens_out"], st["cost"], len(st["events"]), st.get("answer", ""))
