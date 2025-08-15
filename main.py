# main.py
import time
import re
import cv2
from pathlib import Path
from config_loader import load_config
from camera_fetcher import make_session_no_retries, fetch_camera_image  # <— новый cam_fetcher

from face_recognizer import (
    init_insightface,
    build_facebank,
    build_facebank_from_config,
    recognize_on_image,
    draw_results,
    format_result_list,
)
from db import db_connect, db_init, db_update_last_seen, db_upsert_sighting_bytes


def sanitize_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-я_\-\.]+", "_", name)


def process_camera(
    ip, cam_name, port, sess,
    username, password,
    ch_http, ch_rtsp, HTTP_TIMEOUT,
    RTSP_TIMEOUT_MS, RTSP_ATTEMPTS,   # ← вот так
    app, names, embs, threshold, save_labeled, out_dir, conn
):
    """
    Если порт == 80 → HTTP snapshot (ISAPI).
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        img, msg = fetch_camera_image(
            session=sess,
            ip=ip,
            username=username,
            password=password,
            port=port,
            http_snapshot_path=ch_http,     # для 80 порта
            rtsp_path=ch_rtsp,              # для 554 порта (теперь 101)
            http_timeout=HTTP_TIMEOUT,
            rtsp_timeout_ms=RTSP_TIMEOUT_MS,  # ← новое имя параметра
            rtsp_attempts=RTSP_ATTEMPTS,      # ← есть в новом cam_fetcher
            prefer_transport="tcp",           # сначала TCP, при фейле упадёт на UDP внутри
        )
    except Exception as e:
        print(f"❌ {cam_name} ({ip}:{port}): исключение при получении кадра: {e}")
        return []

    if img is None:
        print(f"❌ {cam_name} ({ip}:{port}): {msg}")
        return []

    try:
        results = recognize_on_image(app, names, embs, img, threshold)
    except Exception as e:
        print(f"❌ {cam_name} ({ip}:{port}): ошибка распознавания: {e}")
        return []

    print(f"✅ {cam_name} ({ip}:{port}): {format_result_list(results)}")
    known = [r for r in results if r["name"] != "UNKNOWN"]
    if not known:
        return []

    # разметка
    try:
        labeled = draw_results(img, results)
    except Exception as e:
        print(f"[WARN] draw_results failed: {e}")
        labeled = img  # fallback

    # JPEG bytes
    try:
        ok, buf = cv2.imencode(".jpg", labeled, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpg_bytes = buf.tobytes() if ok else None
    except Exception as e:
        print(f"[WARN] JPEG encode failed: {e}")
        jpg_bytes = None

    # в БД
    if jpg_bytes:
        for r in known:
            try:
                db_upsert_sighting_bytes(conn, name=r["name"], camera=cam_name, ts_iso=ts, image_bytes=jpg_bytes)
            except Exception as e:
                print(f"[DB] insert sighting failed for {r['name']}: {e}")

    # в файловую систему
    if save_labeled:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for r in known:
                person = sanitize_name(r["name"])
                out_path = out_dir / f"{person}.jpg"
                try:
                    cv2.imwrite(str(out_path), labeled)
                except Exception as e:
                    print(f"[WARN] write {out_path} failed: {e}")
        except Exception as e:
            print(f"[WARN] cannot ensure out_dir: {e}")

    return [(r["name"], cam_name, ts) for r in known]


def main():
    cfg = load_config("config.json")
    settings = cfg["settings"]

    cameras = cfg.get("cameras", [])
    print("Cams", cameras)
    ips = [c["ip"] for c in cameras]
    aliases = {c["ip"]: c.get("alias", c["ip"]) for c in cameras}
    # если в конфиге не указан порт — берём 80 (HTTPS мы не используем)
    ports = {c["ip"]: int(c.get("port", 80)) for c in cameras}

    INTERVAL = settings["interval_sec"]
    HTTP_TIMEOUT = (settings["timeout_connect"], settings["timeout_read"])

    print("[INIT] InsightFace…")
    app = init_insightface(model_name=settings["model_name"], gpu=True)

    print("[INIT] Facebank…")
    names, embs = build_facebank_from_config(app, cfg.get("people", []))
    if len(names) == 0:
        print("[INFO] В config.people нет валидных фото — сканирую папку employees/")
        names, embs = build_facebank(app, "employees")
    print(f"[INIT] База сотрудников: {len(names)}")

    conn = db_connect(settings["db_path"])
    db_init(conn)
    print(f"[DB] SQLite → {settings['db_path']}")

    # логин/пароль теперь передаём строками, а не HTTPDigestAuth
    USERNAME = cfg.get("user", "admin")
    PASSWORD = cfg.get("password", "1Qaz2Wsx")

    sess = make_session_no_retries()
    out_dir = Path("output")

    threshold = settings["threshold"]
    save_labeled = settings["save_labeled"]
    ch_http = settings.get("http_snapshot_path", "/ISAPI/Streaming/channels/101/picture")
    ch_rtsp = settings.get("rtsp_channel_path", "Streaming/Channels/101")  # субпоток по умолчанию
    GAP = float(settings.get("gap_between_requests", 0.2))

    # Параметры RTSP через VLC (есть дефолты, можно положить в settings)
    RTSP_TIMEOUT_MS = int(settings.get("rtsp_timeout_ms", 1000))
    RTSP_ATTEMPTS = int(settings.get("rtsp_attempts", 2))  # 2–3 достаточно

    print(f"[LOOP] Опрос каждые {INTERVAL} сек. Ctrl+C для остановки.")
    try:
        while True:
            timerStart = time.time()
            t0 = time.time()
            last_rows = []

            # последовательно обходим камеры
            for ip in ips:
                rows = process_camera(
                    ip, aliases[ip], ports[ip],
                    sess,
                    USERNAME, PASSWORD,
                    ch_http, ch_rtsp, HTTP_TIMEOUT,
                    RTSP_TIMEOUT_MS, RTSP_ATTEMPTS,   # ← так
                    app, names, embs, threshold, save_labeled, out_dir, conn
                )
                if rows:
                    last_rows.extend(rows)

                # пауза между запросами к разным камерам
                if GAP > 0:
                    time.sleep(GAP)

            # дедупликация и запись last_seen
            if last_rows:
                deduped = {}
                for name, camera, ts in last_rows:
                    deduped[(name, camera)] = ts
                last_rows = [(name, camera, ts) for (name, camera), ts in deduped.items()]
                db_update_last_seen(conn, last_rows)

            elapsed = time.time() - t0
            sleep_left = max(0.0, INTERVAL - elapsed)
            if sleep_left > 0:
                time.sleep(sleep_left)

            print("ОБОШЕЛ ВСЕ ЗА", time.time() - timerStart)

    except KeyboardInterrupt:
        print("\n[STOP] Остановлено пользователем.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
