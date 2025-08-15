import time
import re
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.auth import HTTPDigestAuth
from config_loader import load_config
from camera_fetcher import make_session_no_retries, fetch_camera_image
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


def process_camera(ip, cam_name, port, sess, auth, ch_http, ch_rtsp, TIMEOUT,
                   app, names, embs, threshold, save_labeled, out_dir, conn):
    """
    Если порт == 80 → HTTP snapshot (ISAPI).
    Иначе → RTSP (быстрее брать субпоток 102).
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Выбор канала/протокола по порту
    if port == 80:
        channel_path = ch_http            # пример: /ISAPI/Streaming/channels/101/picture
        use_https = False
    elif port == 443:
        # если вдруг есть HTTPS-снапшоты
        channel_path = ch_http
        use_https = True
    else:
        channel_path = ch_rtsp            # пример: Streaming/Channels/102
        use_https = False                 # для RTSP флаг не важен

    img, msg = fetch_camera_image(
        session=sess,
        ip=ip,
        auth=auth,
        channel_path=channel_path,
        timeout=TIMEOUT,
        use_https=use_https,
        port=port,
        transport="tcp",                  # RTSP по TCP стабильнее
        try_ffmpeg_fallback=(port not in (80, 443)),  # ← вкл. фоллбек только для RTSP
        ffmpeg_path="ffmpeg",
        ffmpeg_timeout_s=10,
    )

    if img is None:
        print(f"❌ {cam_name} ({ip}:{port}): {msg}")
        return []

    results = recognize_on_image(app, names, embs, img, threshold)
    print(f"✅ {cam_name} ({ip}:{port}): {format_result_list(results)}")

    known = [r for r in results if r["name"] != "UNKNOWN"]
    if not known:
        return []

    labeled = draw_results(img, results)
    ok, buf = cv2.imencode(".jpg", labeled, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        jpg_bytes = buf.tobytes()
        for r in known:
            try:
                db_upsert_sighting_bytes(conn, name=r["name"], camera=cam_name, ts_iso=ts, image_bytes=jpg_bytes)
            except Exception as e:
                print(f"[DB] insert sighting failed for {r['name']}: {e}")
    else:
        print("[WARN] JPEG encode failed; skip DB image insert")

    if save_labeled:
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in known:
            person = sanitize_name(r["name"])
            out_path = out_dir / f"{person}.jpg"
            cv2.imwrite(str(out_path), labeled)

    return [(r["name"], cam_name, ts) for r in known]



def main():
    cfg = load_config("config.json")
    settings = cfg["settings"]

    cameras = cfg.get("cameras", [])
    print("Cams", cameras)
    ips = [c["ip"] for c in cameras]
    aliases = {c["ip"]: c.get("alias", c["ip"]) for c in cameras}
    ports = {c["ip"]: int(c.get("port", 443 if settings["https"] else 80)) for c in cameras}

    INTERVAL = settings["interval_sec"]
    TIMEOUT = (settings["timeout_connect"], settings["timeout_read"])

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

    auth = HTTPDigestAuth(cfg.get("user", "admin"), cfg.get("password", "1Qaz2Wsx"))
    sess = make_session_no_retries()
    out_dir = Path("output")

    ch_path = settings["channel_path"]
    use_https = settings["https"]
    threshold = settings["threshold"]
    save_labeled = settings["save_labeled"]
    ch_http = settings.get("http_snapshot_path", "/ISAPI/Streaming/channels/101/picture")
    # ch_rtsp = settings.get("rtsp_channel_path", "Streaming/Channels/101") 
    ch_rtsp = settings.get("rtsp_channel_path", "Streaming/Channels/102") 
    GAP = float(settings.get("gap_between_requests", 0.2))

    max_workers = min(8, len(ips))  # например, максимум 8 потоков
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
                    sess, auth,
                    ch_http, ch_rtsp, TIMEOUT,
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