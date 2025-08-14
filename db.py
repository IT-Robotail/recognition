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
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS last_seen (
                    name    TEXT PRIMARY KEY,
                    last_ip TEXT NOT NULL,
                    last_ts TEXT NOT NULL
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sightings (
                    id          BIGSERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    camera      TEXT NOT NULL,
                    ts          TEXT NOT NULL,   -- ISO8601 строка
                    image_bytes BYTEA NOT NULL
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sightings_name_ts ON sightings(name, ts DESC);")
            # <<< ВАЖНО: одна строка на сотрудника
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_sightings_name ON sightings(name);")
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


# =========================
#   НИЖЕ — НОВЫЕ МЕТОДЫ
#   работа с изображениями (байты) в таблице sightings
# =========================

def db_insert_sighting_bytes(
    conn,
    name: str,
    camera: str,
    ts_iso: str,
    image_bytes: bytes,
) -> int:
    """
    Вставить одно наблюдение с изображением в байтах.
    Возвращает id вставленной записи.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("image_bytes должен быть bytes/bytearray/memoryview")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sightings (name, camera, ts, image_bytes)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (name, camera, ts_iso, psycopg2.Binary(image_bytes)),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def db_insert_sightings_bytes_bulk(
    conn,
    rows: List[Tuple[str, str, str, bytes]],
) -> None:
    """
    Пакетная вставка наблюдений.
    rows: [(name, camera, ts_iso, image_bytes), ...]
    """
    if not rows:
        return

    # Подготовим данные: завернём байты в Binary, чтобы execute_values не копировал лишнего
    prepared = []
    for name, camera, ts_iso, img in rows:
        if not isinstance(img, (bytes, bytearray, memoryview)):
            raise TypeError("image_bytes в rows должен быть bytes/bytearray/memoryview")
        prepared.append((name, camera, ts_iso, psycopg2.Binary(img)))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO sightings (name, camera, ts, image_bytes)
            VALUES %s
            """,
            prepared,
        )
    conn.commit()


def db_get_last_sightings_bytes(
    conn,
    name: str,
    limit: int = 3,
) -> List[Tuple[int, str, str, str, bytes]]:
    """
    Получить последние N наблюдений для сотрудника.
    Возвращает список кортежей: (id, name, camera, ts, image_bytes)
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, camera, ts, image_bytes
            FROM sightings
            WHERE name = %s
            ORDER BY ts DESC
            LIMIT %s
            """,
            (name, limit),
        )
        rows = cur.fetchall()
    # psycopg2 вернёт memoryview для BYTEA — приведём к bytes для удобства
    result: List[Tuple[int, str, str, str, bytes]] = []
    for _id, _n, cam, ts, blob in rows:
        result.append((_id, _n, cam, ts, bytes(blob) if isinstance(blob, memoryview) else blob))
    return result


def db_get_sighting_bytes_by_id(conn, sighting_id: int) -> Optional[Tuple[int, str, str, str, bytes]]:
    """
    Получить одно наблюдение по id.
    Возвращает (id, name, camera, ts, image_bytes) либо None.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, name, camera, ts, image_bytes
            FROM sightings
            WHERE id = %s
            """,
            (sighting_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    _id, _n, cam, ts, blob = row
    return (_id, _n, cam, ts, bytes(blob) if isinstance(blob, memoryview) else blob)


def db_upsert_sighting_bytes(conn, name: str, camera: str, ts_iso: str, image_bytes: bytes) -> int:
    """
    Обновить/вставить запись по name (последний кадр сотрудника).
    Возвращает id строки.
    """
    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("image_bytes должен быть bytes/bytearray/memoryview")

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sightings (name, camera, ts, image_bytes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE
              SET camera = EXCLUDED.camera,
                  ts     = EXCLUDED.ts,
                  image_bytes = EXCLUDED.image_bytes
            RETURNING id
            """,
            (name, camera, ts_iso, psycopg2.Binary(image_bytes)),
        )
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


def db_upsert_sightings_bytes_bulk(conn, rows: List[Tuple[str, str, str, bytes]]) -> None:
    """
    Пакетный UPSERT по name.
    rows: [(name, camera, ts_iso, image_bytes), ...]
    """
    if not rows:
        return
    prepared = []
    for name, camera, ts_iso, blob in rows:
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise TypeError("image_bytes в rows должен быть bytes/bytearray/memoryview")
        prepared.append((name, camera, ts_iso, psycopg2.Binary(blob)))

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO sightings (name, camera, ts, image_bytes)
            VALUES %s
            ON CONFLICT (name) DO UPDATE
              SET camera = EXCLUDED.camera,
                  ts     = EXCLUDED.ts,
                  image_bytes = EXCLUDED.image_bytes
            """,
            prepared,
        )
    conn.commit()