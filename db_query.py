import sqlite3
import argparse
from tabulate import tabulate


# Показать всё:
# python db_query.py recognition.db


# Фильтр по имени:
# python db_query.py settings.db --name Serega

# Фильтр по алиасу/камере:
# python db_query.py settings.db --camera Вход

def query_last_seen(db_path, name=None, camera=None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sql = "SELECT name, last_ip, last_ts FROM last_seen"
    params = []
    conditions = []

    if name:
        conditions.append("name LIKE ?")
        params.append(f"%{name}%")
    if camera:
        conditions.append("last_ip LIKE ?")
        params.append(f"%{camera}%")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    sql += " ORDER BY last_ts DESC;"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def main():
    parser = argparse.ArgumentParser(description="Просмотр таблицы last_seen")
    parser.add_argument("db_path", help="Путь к SQLite базе (например, settings.db)")
    parser.add_argument("--name", help="Фильтр по имени сотрудника")
    parser.add_argument("--camera", help="Фильтр по алиасу или IP камеры")

    args = parser.parse_args()

    rows = query_last_seen(args.db_path, args.name, args.camera)

    if rows:
        print(tabulate(rows, headers=["Имя", "Камера (alias)", "Время"], tablefmt="pretty"))
    else:
        print("Нет данных по фильтрам.")

if __name__ == "__main__":
    main()