"""Render eval output (results/*.summary.json + *.jsonl) into README-ready markdown tables."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .taxonomy import CATEGORIES, classify

ORDER = ["baseline", "retrieval", "correction", "full"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a: dict, b: dict) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired correctness {question_id: bool}. Returns (only_a, only_b, p)."""
    ids = a.keys() & b.keys()
    oa = sum(a[i] and not b[i] for i in ids)
    ob = sum(b[i] and not a[i] for i in ids)
    m = oa + ob
    if not m:
        return oa, ob, 1.0
    return oa, ob, min(1.0, 2 * sum(math.comb(m, k) for k in range(min(oa, ob) + 1)) / 2 ** m)


def _load(d: Path, questions: Path | None = None) -> list[dict]:
    out = [json.loads(p.read_text()) for p in d.glob("*.summary.json")]
    if questions and questions.exists():  # recount failures with the current taxonomy.py
        gold = {q["question_id"]: q["SQL"] for q in json.loads(questions.read_text())}
        for s in out:
            jl = d / f"{s['config']}.jsonl"
            recs = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.exists() else []
            s["paired"] = {r["question_id"]: bool(r["correct"]) for r in recs}
            fired = [r for r in recs if r.get("attempts", 0) > 0]
            s["retry_line"] = f"{s['config']}: retry fired on {len(fired)}/{len(recs)}, {sum(r['correct'] for r in fired)} of those ended correct"
            if recs and all("error" in r for r in recs):
                s["failures"] = dict(Counter(
                    r["category"] if r["kind"] == "agent_error" else classify(r["sql"], r["kind"], r["error"], gold.get(r["question_id"]))
                    for r in recs if not r["correct"]))
    return sorted(out, key=lambda s: ORDER.index(s["config"]) if s["config"] in ORDER else 99)


def render(results_dir: Path, questions: Path | None = None) -> str:
    sums = _load(results_dir, questions)
    if not sums:
        return f"No *.summary.json found in {results_dir}. Results pending, run the eval first.\n"
    lines = ["| config | model | n | EX | 95% CI | mean steps | mean LLM calls | mean cost USD | p95 wall s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in sums:
        lo, hi = wilson(round(s["ex"] * s["n"]), s["n"])
        lines.append(f"| {s['config']} | {s['model']} | {s['n']} | {s['ex']:.1%} | {lo:.1%} to {hi:.1%} | "
                     f"{s['mean_steps']} | {s['mean_llm_calls']} | {s['mean_cost_usd']:.6f} | {s['p95_wall_s']} |")
    diffs = sorted({d for s in sums for d in s["ex_by_difficulty"]})
    lines += ["", "EX by difficulty (n in brackets)", "",
              "| config | " + " | ".join(diffs) + " |", "|---|" + "---|" * len(diffs)]
    for s in sums:
        cells = [f"{s['ex_by_difficulty'][d]['ex']:.1%} ({s['ex_by_difficulty'][d]['n']})"
                 if d in s["ex_by_difficulty"] else "-" for d in diffs]
        lines.append(f"| {s['config']} | " + " | ".join(cells) + " |")
    cats = CATEGORIES + sorted({c for s in sums for c in s["failures"]} - set(CATEGORIES))
    lines += ["", "Failure taxonomy (count of wrong answers, heuristic)", "",
              "| category | " + " | ".join(s["config"] for s in sums) + " |", "|---|" + "---|" * len(sums)]
    for c in cats:
        lines.append(f"| {c} | " + " | ".join(str(s["failures"].get(c, 0)) for s in sums) + " |")
    retry = [s["retry_line"] for s in sums if s.get("settings", {}).get("max_retries") and "retry_line" in s]
    if retry:
        lines += ["", "Self-correction activity (retry fires on error, timeout, rejection or zero rows)", ""] + [f"- {x}" for x in retry]
    if len(sums) > 1:
        base = sums[0]
        lines += ["", f"Delta EX vs {base['config']}: " +
                  ", ".join(f"{s['config']} {100 * (s['ex'] - base['ex']):+.1f} pts" for s in sums[1:])]
    pairs = [(x, y) for i, x in enumerate(sums) for y in sums[i + 1:] if "paired" in x and "paired" in y]
    if pairs:
        lines += ["", "Paired exact McNemar test (same questions; only_A = A right and B wrong)", "",
                  "| A | B | shared n | only_A | only_B | delta B-A pts | p |", "|---|---|---|---|---|---|---|"]
        for x, y in pairs:
            n = len(x["paired"].keys() & y["paired"].keys())
            oa, ob, pv = mcnemar(x["paired"], y["paired"])
            lines.append(f"| {x['config']} | {y['config']} | {n} | {oa} | {ob} | {100 * (ob - oa) / n:+.1f} | {pv:.3f} |")
    return "\n".join(lines) + "\n"
