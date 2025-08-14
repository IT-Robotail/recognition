from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json

import psycopg2
from psycopg2.extras import execute_values


# config.json лежит в корне проекта, рядом с этим файлом
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


def _load_pg_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Прочитать параметры подключения к Postgres из config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    pg = cfg.get("postgres", {}) or {}

    # поддержим опциональный port, по умолчанию 5432
    host = pg.get("host", "localhost")
    dbname = pg.get("name", "recognition")
    user = pg.get("user", "postgres")
    password = pg.get("password", "")
    port = int(pg.get("port", 5432)) if str(pg.get("port", "")).strip() else 5432

    return {"host": host, "dbname": dbname, "user": user, "password": password, "port": port}


def db_connect(_unused_db_path: Optional[str] = None):
    """
    Соединение с PostgreSQL.
    Аргумент db_path оставлен для совместимости и игнорируется.
    """
    params = _load_pg_config()
    conn = psycopg2.connect(**params)
    # как и в sqlite-версии, коммитим вручную
    conn.autocommit = False
    return conn


def db_init(conn) -> None:
    """
    Создаёт таблицу, если её нет.
    Типы оставлены текстовыми, чтобы не менять формат last_ts в существующем коде.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS last_seen (
                name    TEXT PRIMARY KEY,  -- сотрудник
                last_ip TEXT NOT NULL,     -- alias камеры (или ip)
                last_ts TEXT NOT NULL      -- ISO8601 строка
            );
            """
        )
    conn.commit()


def db_update_last_seen(conn, rows: List[Tuple[str, str, str]]) -> None:
    """
    rows: список кортежей (name, last_ip_or_alias, last_ts)
    Поведение идентично sqlite-версии: UPSERT по ключу name.
    """
    if not rows:
        return

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO last_seen (name, last_ip, last_ts)
            VALUES %s
            ON CONFLICT (name) DO UPDATE SET
                last_ip = EXCLUDED.last_ip,
                last_ts = EXCLUDED.last_ts
            """,
            rows,
        )
    conn.commit()