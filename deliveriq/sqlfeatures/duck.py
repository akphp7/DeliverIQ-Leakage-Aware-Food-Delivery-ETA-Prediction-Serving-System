"""DuckDB helpers: run the project's SQL files against the cleaned orders.

DuckDB runs in-process (no server), reads pandas DataFrames directly and
speaks standard analytical SQL (CTEs, window functions, QUALIFY), so the
same queries port easily to a warehouse such as Snowflake/BigQuery/Spark SQL.
"""
from __future__ import annotations

from functools import reduce

import duckdb
import pandas as pd

from deliveriq import config

SQL_HISTORY_FILES = ["03_rider_history_features.sql", "04_restaurant_history_features.sql"]
SQL_HISTORY_COLUMNS = ["rider_prev_orders", "rider_prev_avg_minutes", "rider_prev_slow_pct",
                       "rider_days_since_active", "rest_orders_prev7d", "rest_avg_minutes_prev7d"]


def connect(clean: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    orders = clean.reset_index(drop=True).assign(row_id=lambda d: d.index.astype("int64"))
    for c in orders.select_dtypes(include="string").columns:
        orders[c] = orders[c].astype(object)
    con.register("orders_df", orders)
    con.execute("CREATE TABLE orders AS SELECT * FROM orders_df")
    return con


def read_sql(name: str) -> str:
    return (config.SQL_DIR / name).read_text()


def run(con, name: str) -> pd.DataFrame:
    return con.execute(read_sql(name)).df()


def history_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Rider + restaurant history from previous days, aligned to clean's rows."""
    con = connect(clean)
    parts = [run(con, f) for f in SQL_HISTORY_FILES]
    con.close()
    out = reduce(lambda a, b: a.merge(b, on="row_id"), parts).sort_values("row_id")
    out.index = clean.index
    return out[SQL_HISTORY_COLUMNS].astype(float)
