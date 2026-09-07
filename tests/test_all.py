import sqlite3
import pytest
from sqlagent.agent import Agent, Config, extract_sql
from sqlagent.llm import FakeLLM
from sqlagent.sandbox import check_sql, Rejected, run_sql
from sqlagent.taxonomy import classify
from sqlagent import schema as sch


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""create table city(id integer primary key, name text, pop integer);
    create table person(id integer primary key, name text, city_id integer references city(id));
    insert into city values (1,'Delhi',30),(2,'Pune',7);
    insert into person values (1,'a',1),(2,'b',1),(3,'c',2);""")
    c.commit(); c.close()
    return p


@pytest.mark.parametrize("sql", ["DROP TABLE city", "select 1; select 2", "delete from city",
                                 "with x as (select 1) insert into city values (9,'x',1)",
                                 "select 1 -- ok\n; drop table city", "pragma writable_schema=1", "select 'abc"])
def test_rejects(sql):
    with pytest.raises(Rejected):
        check_sql(sql)


def test_allows_denylist_words_in_strings():
    assert check_sql("select * from city where name = 'drop delete'")


def test_run_ok_and_readonly(db):
    assert run_sql(db, "select count(*) from person").rows == [(3,)]
    assert run_sql(db, "update city set pop=0").kind == "rejected"
    assert run_sql(db, "select * from nope").kind == "error"


def test_timeout(db):
    sql = ("with recursive c(x) as (select 1 union all select x+1 from c) select count(*) from c")
    r = run_sql(db, sql, timeout_s=0.5)
    assert r.kind == "timeout"


def test_extract_sql():
    assert extract_sql("```sql\nselect 1\n```") == "select 1"
    assert extract_sql("blah select 2 from t") == "select 2 from t"
    assert extract_sql("```sqlite\nselect 3\n```") == "select 3"  # was "ite\nselect 3"


def test_retrieval_picks_table(db):
    s = sch.introspect(db)
    keep = sch.retrieve(s, "what is the population of Delhi city", top_tables=1)
    assert "city" in keep


def test_self_correction_fixes_bad_column(db):
    llm = FakeLLM(lambda sys_, u: "```sql\nselect count(*) from person\n```" if "Problem:" in u
                  else "```sql\nselect count(nope) from person\n```")
    a = Agent(llm, db, Config(use_plan=False, max_retries=2)).ask("how many persons")
    assert a.result.ok and a.result.rows == [(3,)] and a.attempts == 1


def test_no_correction_when_disabled(db):
    llm = FakeLLM(["```sql\nselect nope from person\n```"])
    a = Agent(llm, db, Config(use_plan=False, max_retries=0)).ask("x")
    assert not a.result.ok and a.attempts == 0 and a.llm_calls == 1


def test_retry_bounded(db):
    llm = FakeLLM(["```sql\nselect nope from person\n```"])
    a = Agent(llm, db, Config(use_plan=False, max_retries=2)).ask("x")
    assert a.attempts == 2 and a.llm_calls == 3


def test_taxonomy():
    assert classify("select x from t", "error", "no such column: x") == "hallucinated_column"
    assert classify("select 1", "timeout") == "timeout"
    assert classify("select a from t join u", "ok", "", "select a from t") == "wrong_join"
    assert classify("select count(a) from t", "ok", "", "select sum(a) from t") == "wrong_aggregation"
    assert classify("select a from t where b=1", "ok", "", "select a from t where b=2") == "wrong_filter"
    assert classify("select divide(1, 2)", "error", "no such function: divide") == "syntax_error"
    assert classify("select b from t where a=1;", "ok", "", "select c from t where a = 1") == "other_wrong_result"


def test_build_smogcast(tmp_path):
    from sqlagent import smogcast
    src = tmp_path / "raw"
    src.mkdir()
    head = "time," + ",".join(smogcast.COLS) + "\n"
    (src / "rohini.csv").write_text(head + "2025-10-20 20:00:00,100,150,25,80,3,300,50,990\n"
                                    "2025-10-20 21:00:00,200,250,24,82,2,310,,991\n")
    db = smogcast.build(src, tmp_path / "s.sqlite")
    r = run_sql(db, "select date_ist, pm2_5_mean, hours from daily")  # 20:00 and 21:00 UTC are 21 Oct IST
    assert r.ok and r.rows == [("2025-10-21", 150.0, 2)]
    assert run_sql(db, "select boundary_layer_height from hourly where pm2_5 = 200").rows == [(None,)]


def test_report_stats():
    from sqlagent.report import mcnemar, wilson
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    a = {i: True for i in range(10)}
    b = {i: i >= 3 for i in range(10)}  # 3 discordant, all one way: p = 2 * 1/8
    assert mcnemar(a, b) == (3, 0, 0.25)
