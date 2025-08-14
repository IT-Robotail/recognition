import time
import threading
from queue import Queue, Empty
import re
import cv2
from pathlib import Path
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
    # для имени в файловой системе (уберём опасные символы)
    return re.sub(r"[^0-9A-Za-zА-Яа-я_\-\.]+", '_', name)

def crop_face(img, bbox, pad: int = 4):
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()

def save_unknown_face(base_dir: Path, ts: str, cam_alias: str, face_img, score: float, idx: int):
    """
    base_dir/date/cam_alias/ts_cam_idx_score.jpg
    """
    date_str = ts.split("T", 1)[0] if "T" in ts else ts.split(" ", 1)[0]
    cam_dir = base_dir / date_str / sanitize_name(cam_alias)
    cam_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{ts.replace(':','-').replace('T','_')}_{idx:02d}_{int(score*100):02d}.jpg"
    out_path = cam_dir / fname
    cv2.imwrite(str(out_path), face_img)
    return out_path

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
    # global _infer_lock
    # _infer_lock = globals().setdefault("_infer_lock", threading.Lock())

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
    ports   = {c["ip"]: int(c.get("port", 443 if settings["https"] else 80)) for c in cameras}

    print("[INIT] InsightFace…")
    app = init_insightface(model_name=settings["model_name"], gpu=True)
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
