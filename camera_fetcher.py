# cam_fetcher.py
from typing import Optional, Tuple
from pathlib import Path
import subprocess
import numpy as np
import cv2
import requests
from requests.auth import HTTPDigestAuth


# =========================
# Публичный API
# =========================

def make_session_no_retries() -> requests.Session:
    """requests.Session без ретраев (достаточно для снапшотов)."""
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=0, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.headers.update({
        "Connection": "close",
        "User-Agent": "CamFetcher/ffmpeg-1.0",
    })
    return s


def fetch_camera_image(
    session: requests.Session,
    ip: str,
    username: str,
    password: str,
    port: int,
    *,
    # — по умолчанию канал 101 —
    http_snapshot_path: str = "/ISAPI/Streaming/channels/101/picture",
    rtsp_path: str = "Streaming/Channels/101",
    http_timeout: Tuple[float, float] = (4.0, 6.0),   # (connect, read)
    rtsp_timeout_ms: int = 1000,                      # общий таймаут на захват кадра
    rtsp_attempts: int = 2,                           # попыток ffmpeg (на транспорте ниже)
    prefer_transport: str = "tcp",                    # "tcp" или "udp"
    ffmpeg_path: str = "ffmpeg",                      # путь к ffmpeg
) -> Tuple[Optional[np.ndarray], str]:
    """
    Возвращает (img, msg).
      - img: numpy.ndarray (BGR) или None.
      - msg: краткое описание результата.

    Правила:
      - port == 80  -> HTTP ISAPI снапшот (только http, без https)
      - port == 554 -> RTSP через ffmpeg (pipe), по умолчанию TCP, затем фоллбэк на UDP
    """
    if port == 80:
        return _http_try_snapshot(
            session=session,
            ip=ip,
            username=username,
            password=password,
            path=http_snapshot_path,
            timeout=http_timeout,
        )
    elif port == 554:
        rtsp_url = _build_rtsp_url(ip, 554, rtsp_path, username, password)
        # порядок транспортов: сначала предпочтительный, потом альтернативный
        order = [prefer_transport.lower()]
        if "tcp" in order:
            order.append("udp")
        else:
            order.append("tcp")

        last_err = ""
        for transport in order[:2]:
            for attempt in range(max(1, rtsp_attempts)):
                img, msg = _ffmpeg_pipe_grab(
                    rtsp_url=rtsp_url,
                    transport=transport,
                    timeout_ms=max(1000, rtsp_timeout_ms),
                    ffmpeg_path=ffmpeg_path,
                )
                if img is not None:
                    return img, f"{ip}: ffmpeg OK via {transport.upper()} ({msg})"
                last_err = f"via {transport.upper()}: {msg}"
        return None, f"{ip}: ffmpeg failed ({last_err})"
    else:
        return None, f"{ip}:{port}: не поддерживаемый порт (разрешены только 80 и 554)"


# =========================
# Внутренняя реализация
# =========================

def _http_try_snapshot(
    session: requests.Session,
    ip: str,
    username: str,
    password: str,
    path: str,
    timeout: Tuple[float, float],
) -> Tuple[Optional[np.ndarray], str]:
    """
    Быстрый снимок по HTTP ISAPI (без HTTPS).
    """
    auth = HTTPDigestAuth(username or "", password or "")
    path = (path or "").lstrip("/")
    url = f"http://{ip}:80/{path}"

    try:
        r = session.get(url, auth=auth, timeout=timeout, stream=False)
    except requests.RequestException as e:
        return None, f"{ip}:80: HTTP запрос не удался ({e})"

    if r.status_code != 200:
        txt = (r.text or "").strip()
        if len(txt) > 200:
            txt = txt[:200] + "..."
        return None, f"{ip}:80: HTTP {r.status_code}{' — ' + txt if txt else ''}"

    data = r.content or b""
    if not data:
        return None, f"{ip}:80: пустой ответ (HTTP)"

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"{ip}:80: не удалось декодировать изображение (HTTP)"
    return img, f"{ip}:80: OK HTTP ({img.shape[1]}x{img.shape[0]})"


def _build_rtsp_url(ip: str, port: int, path: str, username: str, password: str) -> str:
    p = (path or "").lstrip("/")
    u = username or ""
    pw = password or ""
    return f"rtsp://{u}:{pw}@{ip}:{port}/{p}"


def _ffmpeg_pipe_grab(
    rtsp_url: str,
    transport: str,
    timeout_ms: int,
    ffmpeg_path: str = "ffmpeg",
) -> Tuple[Optional[np.ndarray], str]:
    """
    Вытащить 1 кадр через ffmpeg → pipe.
    На твоём ffmpeg опция таймаута — '-timeout' (в микросекундах).
    """
    trans = "tcp" if transport.lower() == "tcp" else "udp"
    # -timeout ждёт МИКРОсекунды; используем общий лимит как upper bound.
    timeout_us = max(1_000_000, int(timeout_ms) * 1000)

    cmd = [
        ffmpeg_path,
        "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", trans,
        "-timeout", str(timeout_us),            # ← твоя сборка ffmpeg принимает -timeout
        "-fflags", "nobuffer", "-flags", "low_delay",
        "-analyzeduration", "100k", "-probesize", "32k",
        "-i", rtsp_url,
        "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=max(2, int(timeout_ms/1000) + 3),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "ffmpeg: timeout"

    if p.returncode != 0 or not p.stdout:
        tail = (p.stderr or b"")[-240:].decode(errors="ignore").strip()
        return None, f"ffmpeg: error {tail or 'no output'}"

    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "ffmpeg: decode failed"
    return img, "pipe OK"
