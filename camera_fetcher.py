from typing import Optional, Tuple
import os
import subprocess
import numpy as np
import cv2
import requests
from requests.auth import HTTPDigestAuth

# ========= Константы =========
DEFAULT_PORT = 554
CAP_PROP_BUFFERSIZE = getattr(cv2, "CAP_PROP_BUFFERSIZE", 38)
CAP_PROP_OPEN_TIMEOUT_MSEC = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", 10252)
CAP_PROP_READ_TIMEOUT_MSEC = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", 10253)


def make_session_no_retries() -> requests.Session:
    """Создаёт requests.Session без ретраев (для совместимости)."""
    from requests.adapters import HTTPAdapter
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=0, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "close", "User-Agent": "FaceSnap/1.3"})
    return s


def _append_query(url_path: str, query: str) -> str:
    if "?" in url_path:
        if url_path.endswith("?") or url_path.endswith("&"):
            return f"{url_path}{query}"
        return f"{url_path}&{query}"
    return f"{url_path}?{query}"


class _TempEnv:
    """Временная установка переменных окружения (для OPENCV_FFMPEG_CAPTURE_OPTIONS)."""
    def __init__(self, **env):
        self._env = env
        self._old = {}

    def __enter__(self):
        for k, v in self._env.items():
            self._old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        return self

    def __exit__(self, *exc):
        for k, v in self._old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _opencv_try_grab(
    url: str,
    task_timeout_ms: int,
    open_share: float = 0.6,
    read_share: float = 0.4,
    buffersize: int = 1,
) -> Tuple[Optional[np.ndarray], str]:
    """Попытка вытащить кадр через OpenCV (FFmpeg backend)."""
    cap = None
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        try:
            cap.set(CAP_PROP_BUFFERSIZE, buffersize)
        except Exception:
            pass
        try:
            cap.set(CAP_PROP_OPEN_TIMEOUT_MSEC, int(task_timeout_ms * open_share))
            cap.set(CAP_PROP_READ_TIMEOUT_MSEC, int(task_timeout_ms * read_share))
        except Exception:
            pass

        if not cap.isOpened():
            return None, "RTSP не открылся (OpenCV)"

        ok, frame = cap.read()
        if not ok or frame is None:
            return None, "кадр не получен (OpenCV)"
        return frame, "OK OpenCV"
    finally:
        if cap is not None:
            cap.release()


def _ffmpeg_try_grab(
    url: str,
    transport: str,
    ffmpeg: str = "ffmpeg",
    rw_timeout_us: int = 7_000_000,
    process_timeout_s: int = 10,
) -> Tuple[Optional[np.ndarray], str]:
    """Попытка вытащить кадр через ffmpeg, без временных файлов (pipe→numpy)."""
    trans_map = {"tcp": "tcp", "udp": "udp", "udp_multicast": "udp_multicast"}
    rtsp_transport = trans_map.get(transport.lower(), "tcp")

    cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", rtsp_transport,
        "-rw_timeout", str(rw_timeout_us),   # микросекунды
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-analyzeduration", "100k",
        "-probesize", "32k",
        "-i", url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "pipe:1",
    ]

    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            timeout=process_timeout_s,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
    except subprocess.TimeoutExpired:
        return None, "ffmpeg timeout"

    if p.returncode != 0:
        return None, f"ffmpeg error: {(p.stderr or b'').decode(errors='ignore')[-400:]}"

    buf = p.stdout or b""
    if not buf:
        return None, "ffmpeg pipe: пустой вывод"
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "ffmpeg pipe: не удалось декодировать"
    return img, "OK ffmpeg(pipe)"


def _http_try_snapshot(
    session: requests.Session,
    ip: str,
    auth: HTTPDigestAuth,
    path: str,
    timeout: tuple,
    use_https: bool,
    port: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Быстрый снимок по HTTP/HTTPS (ISAPI/кастомный endpoint).
    Ожидается, что endpoint вернёт JPEG/PNG байты.
    """
    scheme = "https" if (use_https or port == 443) else "http"
    path = (path or "").lstrip("/")
    url = f"{scheme}://{ip}:{port}/{path}"
    try:
        r = session.get(url, auth=auth, timeout=timeout, stream=False, verify=not use_https)
    except requests.RequestException as e:
        return None, f"{ip}:{port}: HTTP запрос не удался ({e})"

    if r.status_code != 200:
        txt = r.text.strip()
        if len(txt) > 200:
            txt = txt[:200] + "..."
        return None, f"{ip}:{port}: HTTP {r.status_code}{' — ' + txt if txt else ''}"

    ctype = r.headers.get("Content-Type", "").lower()
    if ("image" not in ctype) and (not r.content):
        return None, f"{ip}:{port}: 200, но не изображение"

    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"{ip}:{port}: не удалось декодировать изображение (HTTP)"
    return img, f"{ip}:{port}: OK HTTP ({img.shape[1]}x{img.shape[0]})"


def fetch_camera_image(
    session: requests.Session,
    ip: str,
    auth: HTTPDigestAuth,
    channel_path: str,
    timeout: tuple,
    use_https: bool = False,
    port: int = None,
    transport: str = "tcp",
    add_multicast_query: bool = False,
    task_timeout_ms: int = 4000,
    try_ffmpeg_fallback: bool = True,
    ffmpeg_path: str = "ffmpeg",
    ffmpeg_timeout_s: int = 10,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Если port == 80 (или 443) — берём HTTP snapshot по `channel_path`.
    Иначе — RTSP: OpenCV (низкая задержка) → ffmpeg(pipe) фолбэк (опционально).
    """
    username = getattr(auth, "username", None)
    password = getattr(auth, "password", None)

    # Выбор протокола по порту
    if port in (80, 443):
        return _http_try_snapshot(
            session=session,
            ip=ip,
            auth=auth,
            path=channel_path,
            timeout=timeout,
            use_https=use_https or (port == 443),
            port=port,
        )

    # === RTSP ===
    if port is None:
        port = DEFAULT_PORT

    path = (channel_path or "").lstrip("/")
    if add_multicast_query and transport.lower() == "udp_multicast":
        path = _append_query(path, "transportmode=multicast")

    url = f"rtsp://{username}:{password}@{ip}:{port}/{path}"

    # Подсказываем OpenCV/FFmpeg низкие задержки
    trans = {"tcp": "tcp", "udp": "udp", "udp_multicast": "udp_multicast"}.get(transport.lower(), "tcp")
    ocv_opts = [
        f"rtsp_transport;{trans}",
        f"stimeout;{max(1, int(task_timeout_ms*1000))}",   # мкс
        "fflags;nobuffer",
        "flags;low_delay",
        "analyzeduration;100k",
        "probesize;32k",
    ]
    ocv_env = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    with _TempEnv(**{ocv_env: "|".join(ocv_opts)}):
        img, msg = _opencv_try_grab(url, task_timeout_ms=task_timeout_ms)

    if img is not None:
        return img, f"{ip}: {msg}"

    if try_ffmpeg_fallback:
        img2, msg2 = _ffmpeg_try_grab(
            url=url,
            transport=transport,
            ffmpeg=ffmpeg_path,
            rw_timeout_us=max(1, int(task_timeout_ms * 1000)),
            process_timeout_s=ffmpeg_timeout_s,
        )
        if img2 is not None:
            return img2, f"{ip}: {msg2}"
        return None, f"{ip}: {msg} | {msg2}"

    return None, f"{ip}: {msg}"