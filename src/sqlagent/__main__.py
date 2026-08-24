"""CLI: python -m sqlagent {selfcheck|eval|ask|chat|build-homecredit|build-smogcast|report} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, Config
from .evalharness import PRESETS, load_questions, markdown_table, run_config, selfcheck, db_file
from .llm import make_llm


def main(argv=None):
    p = argparse.ArgumentParser(prog="sqlagent")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("selfcheck", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--questions", default="data/minidev/mini_dev_sqlite.json")
        s.add_argument("--db-root", required=True)
        s.add_argument("--limit", type=int)
        s.add_argument("--seed", type=int, default=0)
        if name == "eval":
            s.add_argument("--configs", default="baseline,full")
            s.add_argument("--out", default="results")
            s.add_argument("--backend")
            s.add_argument("--timeout", type=float, default=10.0)
            s.add_argument("--no-plan", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("db"); a.add_argument("question"); a.add_argument("--evidence", default="")
    a.add_argument("--backend")
    c = sub.add_parser("chat", help="interactive demo REPL over one database")
    c.add_argument("db", nargs="?", default="data/homecredit.duckdb")
    c.add_argument("--backend")
    b = sub.add_parser("build-homecredit", help="build the DuckDB views over the Home Credit parquet files")
    b.add_argument("--src"); b.add_argument("--out", default="data/homecredit.duckdb")
    sm = sub.add_parser("build-smogcast", help="build data/smogcast.sqlite from the smogcast raw CSVs")
    sm.add_argument("--src"); sm.add_argument("--out", default="data/smogcast.sqlite")
    r = sub.add_parser("report", help="render results/ into markdown tables")
    r.add_argument("--results", default="results"); r.add_argument("--write", help="also write to this file")
    r.add_argument("--questions", default="data/minidev/mini_dev_sqlite.json", help="gold SQL, to recount failures")
    args = p.parse_args(argv)

    if args.cmd == "build-homecredit":
        from . import homecredit
        print(homecredit.build(Path(args.src) if args.src else homecredit.DEFAULT_SRC, Path(args.out)))
        return
    if args.cmd == "build-smogcast":
        from . import smogcast
        print(smogcast.build(Path(args.src) if args.src else smogcast.DEFAULT_SRC, Path(args.out)))
        return
    if args.cmd == "report":
        from .report import render
        md = render(Path(args.results), Path(args.questions))
        print(md)
        if args.write:
            Path(args.write).write_text(md)
        return
    if args.cmd == "chat":
        agent = Agent(make_llm(args.backend), args.db, Config(summarize=True))
        total = 0.0
        print(f"{agent.llm.name} on {args.db}. Empty line or Ctrl-D to quit.")
        while True:
            try:
                q = input("question> ").strip()
            except EOFError:
                break
            if not q:
                break
            if not sys.stdin.isatty():  # piped questions: echo them so the transcript reads well
                print(q)
            ans = agent.ask(q)
            total += ans.cost
            print(f"SQL: {ans.sql}\n[{ans.result.kind}] cols={ans.result.cols}")
            for row in ans.result.rows[:10]:
                print("  ", row)
            if ans.summary:
                print(f"Answer: {ans.summary}")
            print(f"retries={ans.attempts} llm_calls={ans.llm_calls} tokens={ans.tokens_in}+{ans.tokens_out} "
                  f"cost=${ans.cost:.6f} (session ${total:.6f})")
        return

    if args.cmd == "ask":
        ans = Agent(make_llm(args.backend), args.db).ask(args.question, args.evidence)
        print(ans.sql, "\n", ans.result.kind, ans.result.rows[:10])
        print(f"cost=${ans.cost:.6f} tokens={ans.tokens_in}+{ans.tokens_out} llm_calls={ans.llm_calls}")
        return
    qs = load_questions(Path(args.questions), args.limit, args.seed)
    if args.cmd == "selfcheck":
        print(json.dumps(selfcheck(qs, Path(args.db_root)), indent=2, default=str))
        return
    llm = make_llm(args.backend)
    sums = []
    for name in args.configs.split(","):
        # predictions get the same row cap as gold, else a correct >1000-row answer scores wrong
        cfg = Config(**PRESETS[name], timeout_s=args.timeout, use_plan=not args.no_plan, max_rows=1_000_000)
        sums.append(run_config(llm, qs, Path(args.db_root), name, cfg, Path(args.out)))
    print(markdown_table(sums))


if __name__ == "__main__":
    main()
