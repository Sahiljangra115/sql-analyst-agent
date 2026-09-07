"""Offline schema-retrieval recall on the eval sample (no LLM). For each question, does the
retrieved schema still contain every table and column the gold SQL uses?
Usage: PYTHONPATH=src python scripts/retrieval_recall.py [--limit 200] [--seed 0] [--jsonl results/retrieval.jsonl]"""
import argparse
import json
import re
from pathlib import Path

from sqlagent import schema as sch
from sqlagent.evalharness import db_file, load_questions

p = argparse.ArgumentParser()
p.add_argument("--questions", default="data/minidev/mini_dev_sqlite.json")
p.add_argument("--db-root", default="data/minidev/dbs")
p.add_argument("--limit", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--jsonl", nargs="*", default=[], help="eval jsonl files to cross-tab against recall")
a = p.parse_args()

qs = load_questions(Path(a.questions), a.limit, a.seed)
schemas, rec = {}, {}
for q in qs:
    db = db_file(Path(a.db_root), q["db_id"])
    s = schemas.setdefault(q["db_id"], sch.introspect(db))
    keep = sch.retrieve(s, q["question"], q.get("evidence", ""))
    gold = re.sub(r"'(?:[^']|'')*'", "''", q["SQL"]).lower()
    words = set(re.findall(r"[`\"]([^`\"]+)[`\"]|(\w+)", gold))
    words = {x or y for x, y in words}
    g_tabs = {t for t in s if t.lower() in words}
    g_cols = {(t, c) for t in g_tabs for c, _, _ in s[t].cols if c.lower() in words}
    miss_t = sorted(g_tabs - set(keep))
    miss_c = sorted(f"{t}.{c}" for t, c in g_cols if t in keep and c not in keep[t])
    rec[q["question_id"]] = dict(tables_ok=not miss_t, cols_ok=not miss_t and not miss_c,
                                 miss_t=miss_t, miss_c=miss_c,
                                 kept_tables=len(keep), all_tables=len(s))
n = len(rec)
t_ok = sum(r["tables_ok"] for r in rec.values())
c_ok = sum(r["cols_ok"] for r in rec.values())
print(f"n={n} seed={a.seed}: all gold tables kept {t_ok}/{n} ({t_ok / n:.1%}), "
      f"all gold tables and columns kept {c_ok}/{n} ({c_ok / n:.1%})")
for f in a.jsonl:
    rows = [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
    for ok in (True, False):
        sub = [r for r in rows if rec[r["question_id"]]["cols_ok"] is ok]
        if sub:
            print(f"  {Path(f).name}: recall {'complete' if ok else 'incomplete'} n={len(sub)} "
                  f"EX={sum(r['correct'] for r in sub) / len(sub):.1%}")
