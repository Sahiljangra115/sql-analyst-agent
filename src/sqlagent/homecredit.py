"""Home Credit DuckDB demo target. Builds a DuckDB file of VIEWS over the credit-risk-policy parquet
files (read only, nothing is copied or written there). Default output: data/homecredit.duckdb."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb

DEFAULT_SRC = Path(os.getenv("HOMECREDIT_DATA", Path(__file__).resolve().parents[3] / "credit-risk-policy/data"))
DEFAULT_DB = Path("data/homecredit.duckdb")
VIEWS = {  # view name -> path under the data folder
    "application_train": "raw/application_train_dated",
    "bureau": "raw/bureau",
    "bureau_balance": "raw/bureau_balance",
    "previous_application": "raw/previous_application",
    "pos_cash_balance": "raw/POS_CASH_balance",
    "credit_card_balance": "raw/credit_card_balance",
    "installments_payments": "raw/installments_payments",
    "model_table": "interim",
}


def build(src: Path = DEFAULT_SRC, db: Path = DEFAULT_DB) -> Path:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    try:
        for name, rel in VIEWS.items():
            files = sorted((src / rel).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet under {src / rel}")
            glob = str((src / rel / "*.parquet").resolve()).replace("'", "''")
            con.execute(f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{glob}\')')
    finally:
        con.close()
    return db
