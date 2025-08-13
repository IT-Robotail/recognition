from typing import List, Tuple
import sqlite3

def db_connect(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def db_init(conn: sqlite3.Connection):

    conn.execute("""
        CREATE TABLE IF NOT EXISTS last_seen (
            name TEXT PRIMARY KEY,   -- сотрудник
            last_ip TEXT NOT NULL,   -- alias камеры (или ip, если alias нет)
            last_ts TEXT NOT NULL    -- ISO8601
        );
    """)
    conn.commit()

def db_update_last_seen(conn: sqlite3.Connection, rows: List[Tuple[str, str, str]]):
    """
    rows: список кортежей (name, last_ip_or_alias, last_ts)
    """
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO last_seen(name, last_ip, last_ts)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            last_ip=excluded.last_ip,
            last_ts=excluded.last_ts
        """,
        rows
    )
    conn.commit()