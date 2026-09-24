"""Smogcast demo target. Builds data/smogcast.sqlite from the smogcast raw CSVs (read only).
The plan wanted the Smogcast Postgres; there is no Postgres on this machine, so the same data goes
into SQLite. Source: Open-Meteo CAMS PM2.5/PM10 + weather, hourly, UTC, one CSV per location."""
from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_SRC = Path(os.getenv("SMOGCAST_DATA", Path(__file__).resolve().parents[3] / "smogcast/data/raw"))
DEFAULT_DB = Path("data/smogcast.sqlite")
# copied from smogcast/config.py SITES (query points for Open-Meteo)
COORDS = {"delhi_centre": (28.61, 77.21), "anand_vihar": (28.65, 77.32), "dwarka": (28.59, 77.05),
          "rohini": (28.74, 77.12), "noida": (28.57, 77.32), "gurugram": (28.46, 77.03),
          "faridabad": (28.41, 77.31), "panipat": (29.39, 76.97), "meerut": (28.98, 77.71),
          "rewari": (28.20, 76.62)}
COLS = ["pm2_5", "pm10", "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m",
        "boundary_layer_height", "surface_pressure"]
IST = timedelta(hours=5, minutes=30)

DDL = """
CREATE TABLE locations (location TEXT PRIMARY KEY, latitude REAL, longitude REAL);
CREATE TABLE hourly (
  location TEXT REFERENCES locations(location), time_utc TEXT, time_ist TEXT, date_ist TEXT,
  pm2_5 REAL, pm10 REAL, temperature_2m REAL, relative_humidity_2m REAL, wind_speed_10m REAL,
  wind_direction_10m REAL, boundary_layer_height REAL, surface_pressure REAL,
  PRIMARY KEY (location, time_utc));
CREATE TABLE daily (
  location TEXT REFERENCES locations(location), date_ist TEXT, pm2_5_mean REAL, pm2_5_max REAL,
  pm10_mean REAL, temperature_mean REAL, wind_speed_mean REAL, hours INTEGER,
  PRIMARY KEY (location, date_ist));
"""


def _f(v: str):
    return float(v) if v not in ("", "nan", "NaN") else None


def build(src: Path = DEFAULT_SRC, db: Path = DEFAULT_DB) -> Path:
    files = sorted(src.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs under {src}")
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    try:
        con.executescript(DDL)
        for f in files:
            loc = f.stem
            con.execute("INSERT INTO locations VALUES (?,?,?)", (loc, *COORDS.get(loc, (None, None))))
            rows = []
            with f.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    ist = datetime.fromisoformat(r["time"]) + IST
                    rows.append((loc, r["time"], ist.strftime("%Y-%m-%d %H:%M:%S"), ist.strftime("%Y-%m-%d"),
                                 *(_f(r[c]) for c in COLS)))
            con.executemany(f"INSERT INTO hourly VALUES ({','.join('?' * 12)})", rows)
        con.execute("""INSERT INTO daily SELECT location, date_ist, ROUND(AVG(pm2_5), 1), MAX(pm2_5),
                       ROUND(AVG(pm10), 1), ROUND(AVG(temperature_2m), 1), ROUND(AVG(wind_speed_10m), 1), COUNT(*)
                       FROM hourly GROUP BY location, date_ist""")
        con.commit()
    finally:
        con.close()
    return db
