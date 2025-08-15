from typing import Optional, Tuple
import time
import tempfile
from pathlib import Path
import ctypes

import numpy as np
import cv2
import requests
from requests.auth import HTTPDigestAuth
import vlc  # pip install python-vlc

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
        "User-Agent": "CamFetcher/1.0"
    })
    return s


def fetch_camera_image(
    session: requests.Session,
    ip: str,
    username: str,
    password: str,
    port: int,
    *,
    http_snapshot_path: str = "/ISAPI/Streaming/channels/101/picture",
    rtsp_path: str = "Streaming/Channels/102",
    http_timeout: Tuple[float, float] = (1, 1.5),   # (connect, read)
    rtsp_task_timeout_ms: int = 1000,
    rtsp_network_caching_ms: int = 800,
    rtsp_attempts: int = 1,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Возвращает (img, msg).
      - img: numpy.ndarray (BGR) или None, если не удалось получить кадр.
      - msg: краткое объяснение результата.

    Правила:
      - port == 80  -> HTTP ISAPI снапшот (только http, без https)
      - port == 554 -> RTSP через VLC (TCP), делаем snapshot
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
        return _rtsp_vlc_snapshot(
            ip=ip,
            username=username,
            password=password,
            port=554,
            path=rtsp_path,
            task_timeout_ms=rtsp_task_timeout_ms,
            network_caching_ms=rtsp_network_caching_ms,
            attempts=rtsp_attempts,
        )
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
    Быстрый снимок по HTTP ISAPI.
    HTTPS не используем — схема жёстко 'http://'.
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

    arr = np.frombuffer(r.content or b"", dtype=np.uint8)
    if arr.size == 0:
        return None, f"{ip}:80: пустой ответ (HTTP)"
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"{ip}:80: не удалось декодировать изображение (HTTP)"
    return img, f"{ip}:80: OK HTTP ({img.shape[1]}x{img.shape[0]})"


def _rtsp_vlc_snapshot(
    ip: str,
    username: str,
    password: str,
    port: int,
    path: str,
    *,
    task_timeout_ms: int,
    network_caching_ms: int,
    attempts: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Снимок по RTSP через libVLC (TCP).
    Делаем snapshot во временный файл, затем читаем его в numpy.
    """
    rtsp_url = _build_rtsp_url(ip, port, path, username, password)
    # connect_timeout_s = max(1.0, task_timeout_ms / 1000.0)
    connect_timeout_s = 2
    last_err = "не удалось получить кадр через VLC"

    with tempfile.TemporaryDirectory(prefix="vlc_snap_") as td:
        out = Path(td) / "snap.jpg"
        for i in range(1, max(1, attempts) + 1):
            try:
                ok, msg = _vlc_take_snapshot_to_file(
                    rtsp_url=rtsp_url,
                    outfile=out,
                    connect_timeout_s=connect_timeout_s,
                    network_caching_ms=network_caching_ms,
                )
                if not ok:
                    last_err = msg
                    time.sleep(0.1 * i)
                    continue

                # читаем jpeg как ndarray
                data = out.read_bytes()
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    last_err = "VLC snapshot есть, но не удалось декодировать JPEG"
                    time.sleep(0.1 * i)
                    continue

                return img, f"{ip}:{port}: {msg} ({img.shape[1]}x{img.shape[0]})"

            except Exception as e:
                last_err = f"exception: {e}"
                time.sleep(0.1 * i)

    return None, f"{ip}:{port}: {last_err}"


def _build_rtsp_url(ip: str, port: int, path: str, username: str, password: str) -> str:
    p = (path or "").lstrip("/")
    u = username or ""
    pw = password or ""
    # Простейший формат аутентификации user:pass@ — обычно ок для камер
    return f"rtsp://{u}:{pw}@{ip}:{port}/{p}"


def _vlc_take_snapshot_to_file(
    rtsp_url: str,
    outfile: Path,
    *,
    connect_timeout_s: float,
    network_caching_ms: int,
) -> Tuple[bool, str]:
    """
    Открывает RTSP (TCP) через libVLC в headless-режиме, ждёт первый кадр и делает snapshot.
    """
    vlc_args = [
        "--intf", "dummy",
        "--no-audio",
        "--no-video-title-show",
        "--vout", "dummy",
        "--no-xlib",
        "--rtsp-tcp",
        "--avcodec-hw=none",
        "--verbose=0",
        "--logmode", "text", "--logfile", "/dev/null",
        f"--network-caching={max(800, network_caching_ms)}",  # минимум 800 мс
        # дополнительные кэши для RTSP (иногда помогает на тяжёлых потоках)
        "--live-caching=1200",
    ]
    instance = vlc.Instance(*vlc_args)
    _suppress_vlc_logs(instance)

    player = instance.media_player_new()
    media = instance.media_new(rtsp_url)
    
    media.add_option(":rtsp-frame-buffer-size=1000000")
    media.add_option(":chroma=RV24")          # принудительно RGB24 в видеовыводе
    media.add_option(":snapshot-format=jpg")  # явный формат снапшота (на всякий случай)

    media.add_option(":rtsp-tcp")
    media.add_option(":avcodec-hw=none")
    media.add_option(f":network-caching={max(800, network_caching_ms)}")
    media.add_option(":demux=live555")
    player.set_media(media)

    try:
        if player.play() == -1:
            return False, "VLC: не удалось запустить воспроизведение"

        start = time.time()
        got = False
        width = height = 0
        while time.time() - start < connect_timeout_s:
            st = player.get_state()
            if st == vlc.State.Error:
                return False, "VLC: ошибка потока"
            if st in (vlc.State.Opening, vlc.State.Buffering, vlc.State.NothingSpecial):
                time.sleep(0.08)
                continue

            # при Playing проверяем, что vout активен и есть размер кадра
            if st == vlc.State.Playing:
                w, h = player.video_get_size(0)
                if w and h:
                    width, height = w, h
                    break
                time.sleep(0.08)
                continue

            if st == vlc.State.Ended:
                return False, "VLC: поток закончился до первого кадра"

        if not (width and height):
            return False, f"VLC: кадр не готов (нет размера), state={player.get_state()}"

        # 2) ДЕЛАЕМ НЕСКОЛЬКО ПОПЫТОК SNAPSHOT
        for attempt in range(6):
            # небольшая пауза: растёт шанс попасть на полноценный (key) кадр
            time.sleep(0.25 + 0.12 * attempt)
            rc = player.video_take_snapshot(0, str(outfile), 0, 0)
            if rc == 0 and outfile.exists() and outfile.stat().st_size > 0:
                return True, "VLC snapshot OK"
        return False, f"VLC: кадр не получен, state={player.get_state()}"

    finally:
        try:
            player.stop()
        except Exception:
            pass
        del player
        del media
        del instance


def _suppress_vlc_logs(instance: vlc.Instance) -> None:
    """
    Глушим болтливые логи libVLC. Оставляем только ошибки.
    """
    try:
        def _cb(udata, level, ctx, fmt, args):
            # 0=debug, 1=notice/info, 2=warning, 3=error
            if level == 3:
                try:
                    msg = ctypes.string_at(ctypes.cast(fmt, ctypes.c_char_p)).decode(errors="ignore")
                except Exception:
                    msg = "libVLC error"
                print(f"[libVLC] {msg.strip()}")
            return
        instance.log_set(_cb, None)
    except Exception:
        pass