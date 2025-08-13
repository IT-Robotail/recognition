import json, os
from copy import deepcopy

DEFAULTS = {
    "cameras": [],
    "people": [],  # [{"name": "Ivan", "photos": ["employees/Ivan/1.jpg", ...]}]
    "settings": {
        "interval_sec": 10,
        "threshold": 0.35,
        "https": False,
        "channel_path": "/ISAPI/Streaming/channels/101/picture",
        "timeout_connect": 3.0,
        "timeout_read": 5.0,
        "gap_between_requests": 0.2,
        "model_name": "buffalo_l",
        "db_path": "recognition.db",
        "save_labeled": False,
        "save_unknown_faces": True,
        "unknown_dir": "unknown_faces",
    }
}

def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst

def load_config(path="config.json"):
    cfg = deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        deep_merge(cfg, file_cfg)

    # нормализация типов
    s = cfg["settings"]
    s["interval_sec"] = int(s.get("interval_sec", 10))
    s["threshold"] = float(s.get("threshold", 0.35))
    s["https"] = bool(s.get("https", False))
    s["timeout_connect"] = float(s.get("timeout_connect", 3.0))
    s["timeout_read"] = float(s.get("timeout_read", 5.0))
    s["gap_between_requests"] = float(s.get("gap_between_requests", 0.2))
    s["model_name"] = s.get("model_name", "buffalo_l") or "buffalo_l"
    s["db_path"] = s.get("db_path", "recognition.db") or "recognition.db"
    s["save_labeled"] = bool(s.get("save_labeled", False))
    s["save_unknown_faces"] = bool(s.get("save_unknown_faces", False))
    s["unknown_dir"] = s.get("unknown_dir", "unknown_faces") or "unknown_faces"

    # 🔧 Миграция камер: добавим port, если отсутствует
    default_port = 443 if s["https"] else 80
    for cam in cfg.get("cameras", []):
        try:
            cam["port"] = int(cam.get("port", default_port))
        except (TypeError, ValueError):
            cam["port"] = default_port
        # страховка на диапазон
        cam["port"] = max(1, min(65535, cam["port"]))

    return cfg