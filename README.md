# sql-analyst-agent

Text-to-SQL analyst agent over SQLite with a measured self-correction lift. Evaluated on BIRD Mini-Dev (500 questions, 11 databases, SQLite dialect).

Status: code complete. **Eval numbers are pending, run later.** No accuracy figure has been measured for the final code, so none is quoted here.

## What it does

LangGraph workflow (`src/sqlagent/agent.py`):

`retrieve schema -> plan -> write SQL -> execute (sandbox) -> [retry with error feedback, bounded] -> done`

- `schema.py`: introspects tables, columns, FKs and sample values. Lexical schema retrieval (token overlap of question + BIRD evidence with table/column names, plus one-hop FK closure). No embeddings.
- `sandbox.py`: read-only execution. Static check (single SELECT/WITH, token denylist after stripping strings and comments), `mode=ro` connection, `query_only`, a sqlite authorizer, a progress-handler timeout and a row cap.
- `agent.py`: retry fires when the query errors, times out, is rejected, or returns zero rows. `max_retries=0` disables self-correction.
- `llm.py`: backends are Anthropic, OpenAI-compatible, native Ollama (sets `num_ctx`, disables thinking) and a deterministic fake for tests. Prices are not hardcoded (`SQLAGENT_PRICE_IN/OUT`); local models cost 0.
- `trace.py`: optional Langfuse tracing, active only when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set. Never exercised in this repo so far.
- `taxonomy.py`: heuristic failure categories (wrong join, wrong filter, hallucinated column, wrong aggregation, timeout, syntax error, sandbox rejected, other). Regex-based comparison against the gold SQL, so spot-check before quoting breakdowns.
- `evalharness.py`: execution accuracy (result-set equality, floats rounded to 6 places), per-difficulty EX, failure counts, mean steps, mean LLM calls, mean cost, p95 wall-clock.

Eval presets: `baseline` (full schema, no retries), `retrieval`, `correction` (full schema, 2 retries), `full` (retrieval + 2 retries). Same model, same questions for all.

## Data

BIRD Mini-Dev 2025-07-03 package (`minidev_0703.zip`, 800 MB) from the official Google Drive link in the HF dataset card (`birdsql/bird_mini_dev`). The Aliyun mirror was too slow (about 11 KB/s). Extracted to `data/minidev/dbs/` (gitignored). `data/minidev/mini_dev_sqlite.json` is the 500-question file from that same package.

Checked: all 500 gold queries pass the sandbox without false rejections. One gold query (question_id 701) exceeds 60 s and counts as wrong in every config.

## Run

```
export ML_VENV=/home/ladliju/Developer/Machine-learning/.venv
source $ML_VENV/bin/activate
export PYTHONPATH=src
python -m sqlagent selfcheck --db-root data/minidev/dbs
python -m sqlagent ask path/to.sqlite "question"
./run_eval.sh          # 4 presets in parallel, LIMIT=200 random sample by default, needs Ollama + gemma4:e2b
```

### Home Credit DuckDB demo target

A second database the agent can query. `build-homecredit` creates `data/homecredit.duckdb` (gitignored) containing only views over the parquet files in `credit-risk-policy/data` (read only, nothing copied or written there). Tables: `application_train`, `bureau`, `bureau_balance`, `previous_application`, `pos_cash_balance`, `credit_card_balance`, `installments_payments`, `model_table`. The agent picks the DuckDB dialect from the `.duckdb` suffix; the sandbox uses a read-only connection, the same static check plus file and extension function denials, a timer interrupt and a row cap. Join keys are hand-listed in `schema.py` (parquet views declare none).

```
python -m sqlagent build-homecredit
python -m sqlagent ask data/homecredit.duckdb "How many applicants defaulted, by gender?"
python -m sqlagent chat                 # interactive demo REPL, prints SQL, rows, tokens, cost per question
```

### Hosted model (API backend)

`SQLAGENT_BACKEND=api` targets any OpenAI-compatible hosted endpoint. The key comes from an environment variable only (`SQLAGENT_API_KEY`, or the variable named in `SQLAGENT_API_KEY_ENV`); no file is read. Set `SQLAGENT_BASE_URL`, `SQLAGENT_MODEL`, and the prices from the provider page for cost tracking. `anthropic` backend reads `ANTHROPIC_API_KEY` via the SDK.

```
export SQLAGENT_BACKEND=api SQLAGENT_BASE_URL=https://.../v1 SQLAGENT_MODEL=<model>
export SQLAGENT_API_KEY=...             # from your shell, never committed
export SQLAGENT_PRICE_IN=<usd per 1M in> SQLAGENT_PRICE_OUT=<usd per 1M out>
python -m sqlagent chat
```

Cost per question is `tokens_in * price_in + tokens_out * price_out`, summed over every LLM call of the question (plan, write, corrections). It is printed by `ask` and `chat` and stored in the eval jsonl and summary (`mean_cost_usd`). Without prices set it reads 0 and a warning is printed.

### Render results

After an eval run, turn `results/*.summary.json` into README-ready tables (config table, EX by difficulty, failure taxonomy per config, delta vs baseline):

```
python -m sqlagent report --results results --write results/tables.md
```

`SQLAGENT_BACKEND=anthropic|api|openai|ollama`, `SQLAGENT_MODEL`, `SQLAGENT_NUM_CTX` (default 12288; Ollama's default 4096 would silently truncate full-schema prompts).

## Results

Pending, run later. Planned: gemma4:e2b via local Ollama, 200-question seeded random sample (full 500 if time allows), 4 presets.

| config | n | EX | mean steps | mean LLM calls | p95 wall s | failures |
|---|---|---|---|---|---|---|
| baseline | pending | pending | pending | pending | pending | pending |
| retrieval | pending | pending | pending | pending | pending | pending |
| correction | pending | pending | pending | pending | pending | pending |
| full | pending | pending | pending | pending | pending | pending |

Expect a weak absolute number from a ~2B-active-parameter model. The claim to test is the delta between rows, not the absolute EX. An 8-question smoke run earlier gave full below baseline, but n=8 is noise and that run used a 4096 context; it is not a result.

## Known gaps

- Eval not run on final code. Tests exist in `tests/test_all.py` (written, passed 16/16 once before the llm.py Ollama change); rerun later.
- Home Credit DuckDB target is implemented but untested against a real LLM (only a fake-LLM smoke run). The Smogcast Postgres target from the plan is not implemented.
- API backend and cost tracking never run against a hosted model. The `report` script was only checked on synthetic summary files.
- No web UI; the demo is the `chat` CLI (Streamlit is not installed).
- Langfuse path untested. Cost is 0 for local models, so cost-per-question is only meaningful with an API backend.
- Taxonomy is heuristic.
