import time
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
from db import db_connect, db_init, db_update_last_seen

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



def main():
    cfg = load_config("config.json")
    settings = cfg["settings"]

    # камеры и алиасы
    cameras = cfg.get("cameras", [])
    ips = [c["ip"] for c in cameras]
    aliases = {c["ip"]: c.get("alias", c["ip"]) for c in cameras}
    ports   = {c["ip"]: int(c.get("port", 443 if settings["https"] else 80)) for c in cameras}

    # тайминги
    INTERVAL = settings["interval_sec"]
    GAP = settings["gap_between_requests"]
    TIMEOUT = (settings["timeout_connect"], settings["timeout_read"])

    # распознавание
    print("[INIT] InsightFace…")
    app = init_insightface(model_name=settings["model_name"], gpu=True)  # при желании gpu=False
    print("[INIT] Facebank…")

    names, embs = build_facebank_from_config(app, cfg.get("people", []))
    if len(names) == 0:
        print("[INFO] В config.people нет валидных фото — сканирую папку employees/")
        names, embs = build_facebank(app, "employees")
    print(f"[INIT] База сотрудников: {len(names)}")

    # БД
    conn = db_connect(settings["db_path"])
    db_init(conn)
    print(f"[DB] SQLite → {settings['db_path']}")

    # HTTP
    auth = HTTPDigestAuth(cfg.get("user", "admin"), cfg.get("password", "1Qaz2Wsx"))
    sess = make_session_no_retries()
    out_dir = Path("output")

    ch_path = settings["channel_path"]
    use_https = settings["https"]
    threshold = settings["threshold"]
    save_labeled = settings["save_labeled"]

    save_unknown = settings["save_unknown_faces"]
    unknown_dir = Path(settings["unknown_dir"])

    print(f"[LOOP] Опрос каждые {INTERVAL} сек. Ctrl+C для остановки.")
    try:
        while True:
            t0 = time.time()
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            last_rows = []

            for ip in ips:
                port = ports.get(ip, 443 if use_https else 80)
                img, msg = fetch_camera_image(
                    session=sess,
                    ip=ip,
                    auth=auth,
                    channel_path=ch_path,
                    timeout=TIMEOUT,
                    use_https=use_https,
                    port=port,                 # ← передаём порт
                )
                cam_name = aliases.get(ip, ip)
                # в сообщениях можно для ясности показывать и порт:
                if img is None:
                    print(f"❌ {cam_name} ({ip}:{port}): {msg}")
                else:
                    results = recognize_on_image(app, names, embs, img, threshold)
                    print(f"✅ {cam_name} ({ip}:{port}): {format_result_list(results)}")

                    if results:
                        # берём самое уверенное лицо с кадра
                        best = max(results, key=lambda r: r["score"])
                        if best["name"] != "UNKNOWN":
                            last_rows.append((best["name"], cam_name, ts))

                        if save_unknown:
                            for idx, r in enumerate(results):
                                if r["name"] == "UNKNOWN":
                                    face = crop_face(img, r["bbox"])
                                    if face is not None:
                                        save_unknown_face(unknown_dir, ts, cam_name, face, r["score"], idx)

                        if save_labeled:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            labeled = draw_results(img, results)
                            out_path = out_dir / f"labeled_{ip.replace('.', '_')}.jpg"
                            import cv2
                            cv2.imwrite(str(out_path), labeled)

                time.sleep(GAP)

            if last_rows:
                db_update_last_seen(conn, last_rows)

            elapsed = time.time() - t0
            sleep_left = max(0.0, INTERVAL - elapsed)
            if sleep_left > 0:
                time.sleep(sleep_left)
    except KeyboardInterrupt:
        print("\n[STOP] Остановлено пользователем.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
