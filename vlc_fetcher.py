# vlc_fetcher.py
from typing import Optional, Tuple
import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth
import vlc  # pip install python-vlc
import ctypes

DEFAULT_PORT = 554

def _suppress_vlc_logs(instance: vlc.Instance):
    """Глушим болтливые логи libVLC (info/debug)."""
    # libVLC позволяет вешать callback; в python-vlc 3.x можно дернуть low-level API:
    try:
        def _cb(udata, level, ctx, fmt, args):
            # level: 0=debug 1=notice/info 2=warning 3=error
            # Пропускаем всё, кроме ошибок (3)
            if level == 3:
                # Выведем кратко только ошибки
                try:
                    msg = ctypes.string_at(ctypes.cast(fmt, ctypes.c_char_p)).decode(errors="ignore")
                except Exception:
                    msg = "libVLC error"
                print(f"[libVLC] {msg.strip()}")
            return
        instance.log_set(_cb, None)
    except Exception:
        # Если не получилось — просто игнорируем
        pass

def _http_try_snapshot(
    session: requests.Session,
    ip: str,
    auth: HTTPDigestAuth,
    path: str,
    timeout: tuple,          # (connect, read)
    use_https: bool,
    port: int,
) -> Tuple[Optional[np.ndarray], str]:
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

    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"{ip}:{port}: не удалось декодировать изображение (HTTP)"
    return img, f"{ip}:{port}: OK HTTP ({img.shape[1]}x{img.shape[0]})"


def _vlc_take_snapshot_to_file(rtsp_url: str, outfile: Path,
                               connect_timeout_s: float = 10.0,
                               network_caching_ms: int = 800) -> Tuple[bool, str]:
    vlc_args = [
        "--intf", "dummy",
        "--no-video-title-show",
        "--no-audio",
        "--vout", "dummy",
        "--no-xlib",
        "--rtsp-tcp",
        "--avcodec-hw=none",          # <— отключаем HW-декодер, чтобы не было строк про D3D/VAAPI
        "--verbose=0",                # <— тише
        f"--network-caching={network_caching_ms}",
    ]
    instance = vlc.Instance(*vlc_args)
    _suppress_vlc_logs(instance)      # <— глушим логи
    player = instance.media_player_new()
    media = instance.media_new(rtsp_url)
    media.add_option(":rtsp-tcp")
    media.add_option(":avcodec-hw=none")
    media.add_option(f":network-caching={network_caching_ms}")
    player.set_media(media)

    try:
        if player.play() == -1:
            return False, "VLC: не удалось запустить плеер"

        start = time.time()
        got = False
        # Ожидаем состояние Playing (или до таймаута)
        while time.time() - start < connect_timeout_s:
            st = player.get_state()
            if st == vlc.State.Playing:
                time.sleep(0.4)  # даём декодеру вывести первый кадр
                rc = player.video_take_snapshot(0, str(outfile), 0, 0)
                if rc == 0 and outfile.exists() and outfile.stat().st_size > 0:
                    got = True
                break
            if st in (vlc.State.Opening, vlc.State.Buffering, vlc.State.NothingSpecial):
                time.sleep(0.15)
                continue
            if st in (vlc.State.Error, vlc.State.Ended):
                break

        if not got:
            return False, f"VLC: не удалось получить кадр, state={player.get_state()}"

        return True, "VLC snapshot OK"
    finally:
        try:
            player.stop()
        except Exception:
            pass
        del player
        del media
        del instance


def fetch_camera_image(
    session: requests.Session,
    ip: str,
    auth: HTTPDigestAuth,
    channel_path: str,
    timeout: tuple,            # (connect, read)
    use_https: bool = False,
    port: int = None,
    transport: str = "tcp",    # для совместимости: игнорируется, используем TCP
    add_multicast_query: bool = False,  # не используется в VLC-снимке
    task_timeout_ms: int = 4000,
    try_ffmpeg_fallback: bool = False,  # не нужен, мы используем VLC
    ffmpeg_path: str = "ffmpeg",
    ffmpeg_timeout_s: int = 10,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Если port == 80/443 — берём HTTP снапшот по channel_path.
    Иначе — RTSP через VLC (libVLC): создаём снапшот в temp-файл и читаем его в numpy.
    """
    username = getattr(auth, "username", None) or ""
    password = getattr(auth, "password", None) or ""

    # HTTP/HTTPS ветка
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

    # === RTSP через VLC ===
    if port is None:
        port = DEFAULT_PORT
    path = (channel_path or "").lstrip("/")
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{path}"

    # таймауты
    connect_timeout_s = max(1.0, task_timeout_ms / 1000.0)
    network_caching_ms = 800

    attempts = 3
    last_err = ""
    with tempfile.TemporaryDirectory(prefix="vlc_snap_") as td:
        out = Path(td) / "snap.jpg"
        for i in range(1, attempts + 1):
            try:
                ok, msg = _vlc_take_snapshot_to_file(
                    rtsp_url,
                    out,
                    connect_timeout_s=connect_timeout_s,
                    network_caching_ms=network_caching_ms,
                )
                if not ok:
                    last_err = msg
                    time.sleep(0.3 * i)  # экспоненциальный бэкофф
                    continue

                arr = np.fromfile(out, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    last_err = "VLC snapshot есть, но не удалось декодировать"
                    time.sleep(0.2 * i)
                    continue
                return img, f"{ip}: {msg} ({img.shape[1]}x{img.shape[0]})"
            except Exception as e:
                last_err = f"exception: {e}"
                time.sleep(0.3 * i)

    return None, f"{ip}: {last_err or 'не удалось получить кадр через VLC'}"
