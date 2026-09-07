"""BIRD Mini-Dev harness: execution accuracy (EX), failure taxonomy, cost/steps/latency per config."""
from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import schema as sch
from .agent import Agent, Config
from .llm import LLM
from .sandbox import run_sql
from .taxonomy import classify

PRESETS = {
    "baseline": dict(retrieval=False, max_retries=0),
    "retrieval": dict(retrieval=True, max_retries=0),
    "correction": dict(retrieval=False, max_retries=2),
    "full": dict(retrieval=True, max_retries=2),
}


def db_file(db_root: Path, db_id: str) -> Path:
    return db_root / db_id / f"{db_id}.sqlite"


def _norm(rows):
    """BIRD EX compares result sets; floats rounded to 6 places to avoid representation noise."""
    return {tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows}


def load_questions(path: Path, limit: int | None = None, seed: int = 0) -> list[dict]:
    qs = json.loads(Path(path).read_text())
    if limit and limit < len(qs):
        qs = random.Random(seed).sample(qs, limit)
    return qs


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))] if xs else 0.0


def run_config(llm: LLM, questions: list[dict], db_root: Path, name: str, cfg: Config,
               out_dir: Path | None = None, gold_timeout: float = 60.0, log=print) -> dict:
    schemas, rows_out = {}, []
    part = None
    if out_dir:  # written per question so a crashed run keeps its progress
        out_dir.mkdir(parents=True, exist_ok=True)
        part = (out_dir / f"{name}.jsonl").open("w")
    for i, q in enumerate(questions):
        db = db_file(db_root, q["db_id"])
        if q["db_id"] not in schemas:
            schemas[q["db_id"]] = sch.introspect(db)
        gold = run_sql(db, q["SQL"], gold_timeout, max_rows=1_000_000)
        t0 = time.monotonic()
        try:
            a = Agent(llm, db, cfg, schemas[q["db_id"]]).ask(q["question"], q.get("evidence", ""))
            wall = time.monotonic() - t0
            correct = a.result.ok and gold.ok and _norm(a.result.rows) == _norm(gold.rows)
            cat = None if correct else classify(a.sql, a.result.kind, a.result.error, q["SQL"])
            rec = dict(question_id=q.get("question_id", i), db_id=q["db_id"], difficulty=q.get("difficulty", ""),
                       correct=correct, category=cat, sql=a.sql, kind=a.result.kind, error=a.result.error[:200],
                       first_kind=a.history[0][1] if a.history else "", attempts=a.attempts,
                       steps=a.steps, llm_calls=a.llm_calls, tokens_in=a.tokens_in, tokens_out=a.tokens_out,
                       cost=a.cost, wall=wall, gold_ok=gold.ok)
        except Exception as e:  # LLM/transport failure: count as wrong, keep the run going
            rec = dict(question_id=q.get("question_id", i), db_id=q["db_id"], difficulty=q.get("difficulty", ""),
                       correct=False, category="agent_error", sql="", kind="agent_error", error=repr(e)[:200],
                       attempts=0, steps=0, llm_calls=0, tokens_in=0, tokens_out=0, cost=0.0,
                       wall=time.monotonic() - t0, gold_ok=gold.ok)
        rows_out.append(rec)
        if part:
            part.write(json.dumps(rec, default=str) + "\n")
            part.flush()
        log(f"[{name}] {i + 1}/{len(questions)} {'OK ' if rec['correct'] else 'BAD'} {rec['category'] or ''}")
    summ = summarize(name, cfg, llm.name, rows_out)
    if out_dir:
        part.close()
        (out_dir / f"{name}.summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def summarize(name, cfg, model, recs) -> dict:
    n = len(recs)
    by_diff = {}
    for d in sorted({r["difficulty"] for r in recs}):
        sub = [r for r in recs if r["difficulty"] == d]
        by_diff[d] = dict(n=len(sub), ex=round(sum(r["correct"] for r in sub) / len(sub), 4))
    return dict(config=name, settings=asdict(cfg), model=model, n=n,
                ex=round(sum(r["correct"] for r in recs) / n, 4) if n else 0.0,
                ex_by_difficulty=by_diff,
                failures=dict(Counter(r["category"] for r in recs if r["category"])),
                mean_steps=round(statistics.fmean(r["steps"] for r in recs), 2) if n else 0,
                mean_llm_calls=round(statistics.fmean(r["llm_calls"] for r in recs), 2) if n else 0,
                mean_cost_usd=round(statistics.fmean(r["cost"] for r in recs), 6) if n else 0,
                mean_tokens=round(statistics.fmean(r["tokens_in"] + r["tokens_out"] for r in recs)) if n else 0,
                p95_wall_s=round(p95([r["wall"] for r in recs]), 2),
                gold_failed=sum(not r["gold_ok"] for r in recs))


def selfcheck(questions: list[dict], db_root: Path, log=print) -> dict:
    """Run every gold SQL through the sandbox. Gold must be accepted (no false positives on the denylist)."""
    rej, err = [], []
    for q in questions:
        r = run_sql(db_file(db_root, q["db_id"]), q["SQL"], 60.0, max_rows=10)
        if r.kind == "rejected":
            rej.append((q.get("question_id"), r.error))
        elif not r.ok:
            err.append((q.get("question_id"), r.kind, r.error[:80]))
    return dict(n=len(questions), rejected=rej, errors=err)


def markdown_table(summaries: list[dict]) -> str:
    h = "| config | n | EX | mean steps | mean LLM calls | mean cost USD | p95 wall s | failures |\n|---|---|---|---|---|---|---|---|\n"
    return h + "\n".join(f"| {s['config']} | {s['n']} | {s['ex']:.1%} | {s['mean_steps']} | {s['mean_llm_calls']} | "
                         f"{s['mean_cost_usd']} | {s['p95_wall_s']} | {s['failures']} |" for s in summaries)
