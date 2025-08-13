from typing import Optional, Tuple
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth
import cv2

def make_session_no_retries() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(max_retries=0, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"Connection": "close", "User-Agent": "FaceSnap/1.3"})
    return s

def fetch_camera_image(
    session: requests.Session,
    ip: str,
    auth: HTTPDigestAuth,
    channel_path: str,
    timeout: tuple,
    use_https: bool = False,
    port: int = None,                 # ← новое
) -> Tuple[Optional[np.ndarray], str]:
    scheme = "https" if use_https else "http"
    # дефолт порта, если вдруг не передали
    if port is None:
        port = 443 if use_https else 80
    url = f"{scheme}://{ip}:{port}{channel_path}"   # ← порт в URL
    try:
        r = session.get(url, auth=auth, timeout=timeout, stream=False, verify=False if use_https else True)
    except requests.RequestException as e:
        return None, f"{ip}:{port}: запрос не удался ({e})"

    if r.status_code != 200:
        txt = r.text.strip()
        if len(txt) > 200:
            txt = txt[:200] + "..."
        return None, f"{ip}:{port}: HTTP {r.status_code}{' — ' + txt if txt else ''}"

    ctype = r.headers.get("Content-Type", "")
    if "image" not in ctype.lower() and not r.content:
        return None, f"{ip}: 200, но не картинка"

    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"{ip}: не удалось декодировать изображение"
    return img, f"{ip}: OK ({img.shape[1]}x{img.shape[0]})"