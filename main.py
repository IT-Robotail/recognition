import time
import threading
from queue import Queue, Empty
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
        try_ffmpeg_fallback=False         # максимально быстро
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

def camera_worker(ip, cam_name, port, settings, app, names, embs, auth, stop_event: threading.Event, dbq: Queue):
    """Опрос одной камеры с собственным интервалом."""
    sess = make_session_no_retries()
    out_dir = Path("output")
    interval = settings["interval_sec"]
    timeout = (settings["timeout_connect"], settings["timeout_read"])
    ch_path = settings["channel_path"]
    use_https = settings["https"]
    threshold = settings["threshold"]
    save_labeled = settings["save_labeled"]

    # Если видишь странности с параллельным инференсом — раскомментируй LOCK
    global _infer_lock
    _infer_lock = globals().setdefault("_infer_lock", threading.Lock())

    while not stop_event.is_set():
        t0 = time.time()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        img, msg = fetch_camera_image(
            session=sess,
            ip=ip,
            auth=auth,
            channel_path=ch_path,
            timeout=timeout,
            use_https=use_https,
            port=port,
        )

        if img is None:
            print(f"❌ {cam_name} ({ip}:{port}): {msg}")
        else:
            # Распознавание (при необходимости оберни в lock)
            # with _infer_lock:
            results = recognize_on_image(app, names, embs, img, threshold)

            print(f"✅ {cam_name} ({ip}:{port}): {format_result_list(results)}")

            known = [r for r in results if r["name"] != "UNKNOWN"]
            if known:
                # Обновим last_seen батчем через очередь
                best = max(known, key=lambda r: r["score"])
                dbq.put(("last_seen", (best["name"], cam_name, ts)))

                # Один раз рисуем и кодируем JPEG
                labeled = draw_results(img, results)
                ok, buf = cv2.imencode(".jpg", labeled, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    jpg_bytes = buf.tobytes()
                    for r in known:
                        # UPSERT «последнего кадра» в БД
                        dbq.put(("sighting_upsert", (r["name"], cam_name, ts, jpg_bytes)))

                        # (опционально) сохраняем файл «Имя.jpg»
                        if save_labeled:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            person = sanitize_name(r["name"])
                            cv2.imwrite(str((out_dir / f"{person}.jpg").with_suffix(".jpg")), labeled)
                else:
                    print("[WARN] JPEG encode failed; skip DB image insert")

        # Поддержим индивидуальный интервал: спим остаток
        elapsed = time.time() - t0
        left = max(0.0, interval - elapsed)
        if stop_event.wait(left):
            break

def db_writer(settings, stop_event: threading.Event, dbq: Queue):
    """Единый писатель в БД: батчит last_seen и делает upsert изображений."""
    conn = db_connect(settings["db_path"])
    db_init(conn)

    batch_last = []
    last_flush = time.time()

    FLUSH_EVERY = 1.0   # сек
    BATCH_SIZE  = 50

    while not stop_event.is_set() or not dbq.empty():
        try:
            typ, payload = dbq.get(timeout=0.5)
        except Empty:
            typ = None

        if typ == "last_seen":
            batch_last.append(payload)

        elif typ == "sighting_upsert":
            name, cam, ts, jpg = payload
            try:
                db_upsert_sighting_bytes(conn, name=name, camera=cam, ts_iso=ts, image_bytes=jpg)
            except Exception as e:
                print(f"[DB] upsert sighting error for {name}: {e}")

        # Периодический флеш батча last_seen
        now = time.time()
        if batch_last and (len(batch_last) >= BATCH_SIZE or (now - last_flush) >= FLUSH_EVERY):
            try:
                db_update_last_seen(conn, batch_last)
            except Exception as e:
                print(f"[DB] last_seen batch error: {e}")
            batch_last.clear()
            last_flush = now

    # финальный флеш
    if batch_last:
        try:
            db_update_last_seen(conn, batch_last)
        except Exception as e:
            print(f"[DB] last_seen final flush error: {e}")

    conn.close()

def main():
    cfg = load_config("config.json")
    settings = cfg["settings"]

    cameras = cfg.get("cameras", [])
    ips = [c["ip"] for c in cameras]
    aliases = {c["ip"]: c.get("alias", c["ip"]) for c in cameras}
    ports = {c["ip"]: int(c.get("port", 443 if settings["https"] else 80)) for c in cameras}

    print("[INIT] InsightFace…")
    app = init_insightface(model_name=settings["model_name"], gpu=True)
    print("[INIT] Facebank…")

    print("[INIT] Facebank…")
    names, embs = build_facebank_from_config(app, cfg.get("people", []))
    if len(names) == 0:
        print("[INFO] В config.people нет валидных фото — сканирую папку employees/")
        names, embs = build_facebank(app, "employees")
    print(f"[INIT] База сотрудников: {len(names)}")

    # HTTP auth
    auth = HTTPDigestAuth(cfg.get("user", "admin"), cfg.get("password", "1Qaz2Wsx"))

    # Очередь событий в БД и стоп-флаг
    dbq = Queue()
    stop_event = threading.Event()

    # Стартуем писателя в БД
    db_thread = threading.Thread(target=db_writer, args=(settings, stop_event, dbq), daemon=True)
    db_thread.start()

    # Стартуем воркеры камер
    workers = []
    for ip in ips:
        cam_name = aliases.get(ip, ip)
        port     = ports.get(ip, 443 if settings["https"] else 80)

        t = threading.Thread(
            target=camera_worker,
            args=(ip, cam_name, port, settings, app, names, embs, auth, stop_event, dbq),
            daemon=True
        )
        t.start()
        workers.append(t)

    print(f"[LOOP] Параллельный опрос {len(workers)} камер. Ctrl+C для остановки.")

    try:
        # основной поток просто “живёт”, пока не прервут
        while any(t.is_alive() for t in workers):
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOP] Останавливаемся…")
    finally:
        stop_event.set()
        for t in workers:
            t.join(timeout=5)
        # дождёмся, пока писатель вычитает очередь и закроет соединение
        db_thread.join(timeout=5)


if __name__ == "__main__":
    main()