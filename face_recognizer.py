from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import os
from PIL import Image, ImageFont, ImageDraw

FONT_PATH = "DejaVuSans.ttf"  # проверь путь
FONT_SIZE = 20


def imread_unicode(path: str):
    """Надёжное чтение картинки по любому Unicode‑пути на Windows."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# --------- init ----------
def init_insightface(model_name: str = "buffalo_l", gpu: bool = True):
    import insightface
    app = insightface.app.FaceAnalysis(name=model_name, root="~/.insightface")
    ctx_id = 0 if gpu else -1
    try:
        providers = []
        if gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        app.prepare(ctx_id=ctx_id, det_size=(640, 640), providers=providers)
    except TypeError:
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / max(n, 1e-12)

# --------- facebank ----------
def build_facebank(app, emp_dir: str):
   


    emp_dir = Path(emp_dir)
    # берём файлы любого регистра расширения
    paths = [p for p in emp_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
    if not paths:
        raise RuntimeError(f"В '{emp_dir}' нет изображений сотрудников.")

    facebank = {}
    for p in paths:
        if p.stat().st_size == 0:
            print(f"[WARN] Пустой файл: {p}")
            continue
        img_bgr = imread_unicode(str(p))           # <-- вместо cv2.imread
        if img_bgr is None:
            print(f"[WARN] Не читается файл сотрудника: {p}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if not faces:
            print(f"[WARN] На фото сотрудника нет лиц. удаляю: {p}")
            os.remove(p)
            continue
        faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        emb = l2_normalize(faces[0].embedding.astype(np.float32))
        name = p.parent.name if p.parent != emp_dir else p.stem
        facebank.setdefault(name, []).append(emb)



    names, embs = [], []
    for name, vecs in facebank.items():
        if not vecs:
            continue
        mean_emb = l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))
        names.append(name)
        embs.append(mean_emb)
    if not embs:
        raise RuntimeError("Пустой facebank после обработки папки сотрудников.")
    return names, np.stack(embs, axis=0).astype(np.float32)

# --------- recognition ----------
def recognize_on_image(app, names: List[str], embs: np.ndarray, image_bgr: np.ndarray, threshold: float):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = app.get(img_rgb)
    results = []
    if len(embs) == 0:
        return results
    norms = np.linalg.norm(embs, axis=1)
    for f in faces:
        emb = l2_normalize(f.embedding.astype(np.float32))
        sims = (embs @ emb) / (norms * np.linalg.norm(emb) + 1e-12)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        label = names[idx] if score >= threshold else "UNKNOWN"
        results.append({"bbox": [int(x) for x in f.bbox], "name": label, "score": score})
    return results

# def draw_results(image_bgr: np.ndarray, results) -> np.ndarray:
#     out = image_bgr.copy()
#     for r in results:
#         x1, y1, x2, y2 = r["bbox"]
#         color = (0, 255, 0) if r["name"] != "UNKNOWN" else (0, 0, 255)
#         cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
#         text = f"{r['name']} ({r['score']:.2f})"
#         (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
#         cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
#         cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
#     return out

def draw_results(img, results):
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for r in results:
        x1, y1, x2, y2 = r["bbox"]
        name = r["name"]
        color = (0,255,0) if name != "UNKNOWN" else (255,0,0)

        # рамка
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # подпись (фон + текст)
        text = f"{name} {r['score']:.2f}"
        tw, th = draw.textlength(text, font=font), font.getbbox("Hg")[3]
        pad = 4
        draw.rectangle([x1, y1 - th - 2*pad, x1 + tw + 2*pad, y1], fill=(0,0,0))
        draw.text((x1 + pad, y1 - th - pad), text, fill=(255,255,255), font=font)

    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def format_result_list(results):
    return ", ".join([f"{r['name']}({r['score']:.2f})" for r in results]) if results else "лиц не найдено"




def build_facebank_from_config(app, people: list):

    facebank = {}
    for person in people:
        name = person.get("name", "").strip()
        if not name:
            continue
        for p in person.get("photos", []):
            img_path = Path(p)
            if not img_path.exists() or img_path.stat().st_size == 0:
                continue
            img_bgr = imread_unicode(str(img_path))  # <-- вместо cv2.imread
            if img_bgr is None:
                print(f"[WARN] Не читается файл сотрудника: {img_path}")
                continue




    names, embs = [], []
    for name, vecs in facebank.items():
        if not vecs:
            continue
        mean_emb = l2_normalize(np.mean(np.stack(vecs, axis=0), axis=0))
        names.append(name)
        embs.append(mean_emb)

    if not embs:
        # откат — пусть выше решают, чем заполнять
        return [], np.empty((0, 512), dtype=np.float32)
    return names, np.stack(embs, axis=0).astype(np.float32)