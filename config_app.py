import os, json, shutil, ipaddress, uuid
from pathlib import Path
import sys, subprocess
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QMessageBox, QFileDialog, QTabWidget, QCheckBox,
    QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSplitter, QInputDialog
)

APP_TITLE = "Настройки распознавания"
CONFIG_PATH = "config.json"
EMPLOYEES_DIR = Path("employees")
EMPLOYEES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IP_BASE = "192.168.254"
DEFAULT_IP_RANGE = (101, 150)

def default_config():
    return {
        "cameras": [{"ip": f"{DEFAULT_IP_BASE}.{i}", "alias": f"Cam {i}"} for i in range(DEFAULT_IP_RANGE[0], DEFAULT_IP_RANGE[1] + 1)],
        "people": [],
        "settings": {
            "interval_sec": 10,
            "threshold": 0.35,
            "https": False,
            "channel_path": "/ISAPI/Streaming/channels/101/picture",
            "timeout_connect": 3,
            "timeout_read": 5,
            "gap_between_requests": 0.2,
            "model_name": "buffalo_l",
            "db_path": "recognition.db",
            "save_labeled": False
        }
    }

def load_config():
    if not os.path.exists(CONFIG_PATH):
        cfg = default_config()
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        return cfg
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def validate_ip(ip: str) -> bool:
    try:
        ipaddress.IPv4Address(ip)
        return True
    except ipaddress.AddressValueError:
        return False

class CamerasTab(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        layout = QVBoxLayout(self)

        # список камер
        self.list = QListWidget()
        self.refresh_list()
        layout.addWidget(QLabel("Камеры:"))
        layout.addWidget(self.list)

        # кнопки
        btns = QHBoxLayout()
        self.btn_add = QPushButton("Добавить")
        self.btn_edit = QPushButton("Изменить")
        self.btn_delete = QPushButton("Удалить")
        btns.addWidget(self.btn_add)
        btns.addWidget(self.btn_edit)
        btns.addWidget(self.btn_delete)
        layout.addLayout(btns)

        self.btn_add.clicked.connect(self.add_camera)
        self.btn_edit.clicked.connect(self.edit_camera)
        self.btn_delete.clicked.connect(self.delete_camera)

    def refresh_list(self):
        self.list.clear()
        cams = sorted(self.cfg["cameras"], key=lambda c: list(map(int, c["ip"].split("."))))
        for cam in cams:
            port = cam.get("port", 80)
            item = QListWidgetItem(f"{cam.get('alias', cam['ip'])}  ({cam['ip']}:{port})")
            item.setData(Qt.UserRole, cam["ip"])   # ключ — ip
            self.list.addItem(item)
        
    def _find_camera_index(self, ip: str) -> int:
        for i, c in enumerate(self.cfg["cameras"]):
            if c["ip"] == ip:
                return i
        return -1
    
    def add_camera(self):
        ip, ok = QInputDialog.getText(self, "Новая камера", "IP-адрес:")
        if not ok: return
        ip = ip.strip()
        if not ip:
            QMessageBox.warning(self, APP_TITLE, "Пустой IP.")
            return
        if not validate_ip(ip):
            QMessageBox.warning(self, APP_TITLE, "Некорректный IP.")
            return
        if any(c["ip"] == ip for c in self.cfg["cameras"]):
            QMessageBox.warning(self, APP_TITLE, "Такая камера уже есть.")
            return
        alias, ok2 = QInputDialog.getText(self, "Новая камера", "Псевдоним:")
        if not ok2: return
        alias = alias.strip() or ip

        # спросим порт (дефолт 80; если в Settings включён HTTPS, можно подставить 443)
        from recognition.config_loader import load_config
        cfg_full = load_config("config.json")
        default_port = 443 if cfg_full["settings"].get("https", False) else 80
        port_str, ok3 = QInputDialog.getText(self, "Новая камера", f"Порт (1–65535):", text=str(default_port))
        if not ok3: return
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, APP_TITLE, "Некорректный порт.")
            return
        
        self.cfg["cameras"].append({"ip": ip, "alias": alias})
        save_config(self.cfg)
        self.refresh_list()

    def edit_camera(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.information(self, APP_TITLE, "Выберите камеру.")
            return
        ip = item.data(Qt.UserRole)
        idx = self._find_camera_index(ip)
        if idx < 0:
            QMessageBox.critical(self, APP_TITLE, "Камера не найдена в конфиге.")
            return

        cur = self.cfg["cameras"][idx]
        # alias
        alias, ok = QInputDialog.getText(self, "Псевдоним камеры", f"{ip}: имя", text=cur.get("alias", ip))
        if not ok: return
        alias = (alias or ip).strip()

        # порт
        port_str, ok2 = QInputDialog.getText(self, "Порт камеры", f"{ip}: порт (1–65535)", text=str(cur.get("port", 80)))
        if not ok2: return
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, APP_TITLE, "Некорректный порт.")
            return

        self.cfg["cameras"][idx]["alias"] = alias
        self.cfg["cameras"][idx]["port"] = port
        save_config(self.cfg)
        self.refresh_list()

    def delete_camera(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.information(self, APP_TITLE, "Выберите камеру.")
            return
        ip = item.data(Qt.UserRole)
        idx = self._find_camera_index(ip)
        if idx < 0:
            QMessageBox.critical(self, APP_TITLE, "Камера не найдена в конфиге.")
            return
        alias = self.cfg["cameras"][idx].get("alias", ip)
        if QMessageBox.question(self, APP_TITLE, f"Удалить камеру {alias} ({ip})?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.cfg["cameras"].pop(idx)
        save_config(self.cfg)
        self.refresh_list()

class PeopleTab(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        splitter = QSplitter(Qt.Horizontal, self)

        # ---- слева: список людей + кнопки ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Сотрудники:"))
        self.people_list = QListWidget()
        left_layout.addWidget(self.people_list)
        btns = QHBoxLayout()
        self.btn_add_person = QPushButton("Добавить")
        self.btn_rename_person = QPushButton("Переименовать")
        self.btn_del_person = QPushButton("Удалить")
        btns.addWidget(self.btn_add_person)
        btns.addWidget(self.btn_rename_person)
        btns.addWidget(self.btn_del_person)
        left_layout.addLayout(btns)
        splitter.addWidget(left)

        # ---- справа: фото + предпросмотр ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("Фотографии:"))
        self.photos_list = QListWidget()
        right_layout.addWidget(self.photos_list)

        photo_btns = QHBoxLayout()
        self.btn_add_photo = QPushButton("Добавить фото")
        self.btn_del_photo = QPushButton("Удалить фото")
        self.btn_open_dir = QPushButton("Открыть папку сотрудника")
        photo_btns.addWidget(self.btn_add_photo)
        photo_btns.addWidget(self.btn_del_photo)
        photo_btns.addWidget(self.btn_open_dir)
        right_layout.addLayout(photo_btns)

        self.preview = QLabel("Предпросмотр")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(220)
        self.preview.setStyleSheet("border:1px solid #999;")
        right_layout.addWidget(self.preview)

        splitter.addWidget(right)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        # connections
        self.people_list.currentItemChanged.connect(self.on_person_selected)
        self.photos_list.currentItemChanged.connect(self.on_photo_selected)
        self.btn_add_person.clicked.connect(self.add_person)
        self.btn_rename_person.clicked.connect(self.rename_person)
        self.btn_del_person.clicked.connect(self.delete_person)
        self.btn_add_photo.clicked.connect(self.add_photos)
        self.btn_del_photo.clicked.connect(self.delete_photo)
        self.btn_open_dir.clicked.connect(self.open_person_dir)

        # ВАЖНО: теперь вызываем refresh_people ПОСЛЕ создания self.photos_list
        self.refresh_people()
    # helpers
    def refresh_people(self):
        self.people_list.clear()
        for person in sorted(self.cfg["people"], key=lambda p: p["name"].lower()):
            item = QListWidgetItem(person["name"])
            # Храним ИМЯ как ключ
            item.setData(Qt.UserRole, person["name"])
            self.people_list.addItem(item)
        self.photos_list.clear()
        self.preview.setText("Предпросмотр")

    def _find_person_index(self, name: str) -> int:
        for i, p in enumerate(self.cfg["people"]):
            if p["name"].lower() == name.lower():
                return i
        return -1

    def current_person_name(self):
        item = self.people_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def open_person_dir(self):
        person = self.current_person()
        if not person:
            QMessageBox.information(self, APP_TITLE, "Выберите сотрудника.")
            return
        dir_path = EMPLOYEES_DIR / person["name"]
        dir_path.mkdir(parents=True, exist_ok=True)

        # Открываем папку в проводнике (Windows/macOS/Linux)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(dir_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(dir_path)])
            else:
                subprocess.Popen(["xdg-open", str(dir_path)])
        except Exception as e:
            QMessageBox.critical(self, APP_TITLE, f"Не удалось открыть папку:\n{e}")
        

    def current_person(self):
        name = self.current_person_name()
        idx = self._find_person_index(name) if name else -1
        return (idx, self.cfg["people"][idx]) if idx >= 0 else (-1, None)

    def refresh_photos(self, person):
        self.photos_list.clear()
        for p in person.get("photos", []):
            self.photos_list.addItem(QListWidgetItem(p))
        self.preview.setText("Предпросмотр")

    # actions
    def add_person(self):
        name, ok = QInputDialog.getText(self, "Новый сотрудник", "Имя (латиница/русский):")
        if not ok: return
        name = name.strip()
        if not name:
            QMessageBox.warning(self, APP_TITLE, "Имя пустое.")
            return
        if any(p["name"].lower() == name.lower() for p in self.cfg["people"]):
            QMessageBox.warning(self, APP_TITLE, "Сотрудник с таким именем уже есть.")
            return
        person = {"name": name, "photos": []}
        self.cfg["people"].append(person)
        save_config(self.cfg)
        # создать папку
        (EMPLOYEES_DIR / name).mkdir(parents=True, exist_ok=True)
        self.refresh_people()

    def rename_person(self):
        idx, person = self.current_person()
        if person is None:
            QMessageBox.information(self, APP_TITLE, "Выберите сотрудника.")
            return
        new_name, ok = QInputDialog.getText(self, "Переименование", "Новое имя:", text=person["name"])
        if not ok:
            return
        new_name = new_name.strip()
        if not new_name:
            return
        if self._find_person_index(new_name) >= 0 and new_name.lower() != person["name"].lower():
            QMessageBox.warning(self, APP_TITLE, "Имя занято.")
            return

        old_dir = EMPLOYEES_DIR / person["name"]
        new_dir = EMPLOYEES_DIR / new_name
        try:
            if old_dir.exists():
                old_dir.rename(new_dir)
        except Exception as e:
            QMessageBox.critical(self, APP_TITLE, f"Не удалось переименовать папку: {e}")
            return

        # Обновляем запись в конфиге по индексу
        self.cfg["people"][idx]["name"] = new_name
        # Переписываем пути к фото
        self.cfg["people"][idx]["photos"] = [str((new_dir / Path(p).name).as_posix()) for p in person.get("photos", [])]
        save_config(self.cfg)
        self.refresh_people()

    def delete_person(self):
        person = self.current_person()
        if not person:
            QMessageBox.information(self, APP_TITLE, "Выберите сотрудника.")
            return
        reply = QMessageBox.question(self, APP_TITLE, f"Удалить сотрудника {person['name']}?\nФото в папке не трогаем.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        # удаляем из конфига, файлы оставляем на диске (безопасно)
        self.cfg["people"] = [p for p in self.cfg["people"] if p is not person]
        save_config(self.cfg)
        self.refresh_people()

    def add_photos(self):
        idx, person = self.current_person()
        if person is None:
            QMessageBox.information(self, APP_TITLE, "Выберите сотрудника.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Добавить фотографии", "",
                                                "Изображения (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        dest_dir = EMPLOYEES_DIR / person["name"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        arr = person.get("photos", [])
        for src in files:
            try:
                ext = Path(src).suffix.lower()
                new_name = f"{Path(src).stem}_{uuid.uuid4().hex[:8]}{ext}"
                dst = dest_dir / new_name
                shutil.copy2(src, dst)
                rel = str(dst.as_posix())
                if rel not in arr:
                    arr.append(rel)
                    added += 1
            except Exception as e:
                QMessageBox.warning(self, APP_TITLE, f"Не удалось добавить {src}:\n{e}")
        # КРИТИЧЕСКОЕ: возвращаем массив обратно в cfg по ИНДЕКСУ и сохраняем
        self.cfg["people"][idx]["photos"] = arr
        if added:
            save_config(self.cfg)
            self.refresh_photos(self.cfg["people"][idx])

    def delete_photo(self):
        idx, person = self.current_person()
        if person is None:
            QMessageBox.information(self, APP_TITLE, "Выберите сотрудника.")
            return
        item = self.photos_list.currentItem()
        if not item:
            QMessageBox.information(self, APP_TITLE, "Выберите фото.")
            return
        path = item.text()
        if QMessageBox.question(self, APP_TITLE, "Удалить запись о фото из конфига? Файл на диске останется.",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        arr = person.get("photos", [])
        if path in arr:
            arr.remove(path)
            self.cfg["people"][idx]["photos"] = arr
            save_config(self.cfg)
            self.refresh_photos(self.cfg["people"][idx])

    def on_person_selected(self, current, _previous):
        name = current.data(Qt.UserRole) if current else None
        idx = self._find_person_index(name) if name else -1
        if idx >= 0:
            self.refresh_photos(self.cfg["people"][idx])
        else:
            self.photos_list.clear()
            self.preview.setText("Предпросмотр")

    def on_photo_selected(self, current, _previous):
        if not current:
            self.preview.clear()
            self.preview.setText("Предпросмотр")
            return
        path = current.text()
        pm = QPixmap(path)
        if pm.isNull():
            self.preview.setText("Не удалось загрузить изображение")
            return
        # масштабируем по размеру области предпросмотра
        self.preview.setPixmap(pm.scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # обновить масштаб предпросмотра
        it = self.photos_list.currentItem()
        if it:
            self.on_photo_selected(it, None)

class SettingsTab(QWidget):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        s = cfg["settings"]

        layout = QVBoxLayout(self)

        form = QFormLayout()

        self.interval = QSpinBox(); self.interval.setRange(1, 3600); self.interval.setValue(s.get("interval_sec", 10))
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0.0, 1.0); self.threshold.setSingleStep(0.01); self.threshold.setValue(s.get("threshold", 0.35))
        self.https = QCheckBox(); self.https.setChecked(s.get("https", False))
        self.channel = QLineEdit(s.get("channel_path", "/ISAPI/Streaming/channels/101/picture"))
        self.t_conn = QDoubleSpinBox(); self.t_conn.setRange(0.1, 60.0); self.t_conn.setSingleStep(0.1); self.t_conn.setValue(s.get("timeout_connect", 3))
        self.t_read = QDoubleSpinBox(); self.t_read.setRange(0.1, 120.0); self.t_read.setSingleStep(0.1); self.t_read.setValue(s.get("timeout_read", 5))
        self.gap = QDoubleSpinBox(); self.gap.setRange(0.0, 10.0); self.gap.setSingleStep(0.1); self.gap.setValue(s.get("gap_between_requests", 0.2))
        self.model = QLineEdit(s.get("model_name", "buffalo_l"))
        self.db_path = QLineEdit(s.get("db_path", "recognition.db"))
        self.save_labeled = QCheckBox(); self.save_labeled.setChecked(s.get("save_labeled", False))

        form.addRow("Интервал опроса (сек)", self.interval)
        form.addRow("Порог распознавания", self.threshold)
        form.addRow("HTTPS (самоподп.)", self.https)
        form.addRow("ISAPI канал", self.channel)
        form.addRow("Таймаут соединения (с)", self.t_conn)
        form.addRow("Таймаут чтения (с)", self.t_read)
        form.addRow("Пауза между камерами (с)", self.gap)
        form.addRow("Модель InsightFace", self.model)
        form.addRow("SQLite база", self.db_path)
        form.addRow("Сохранять размеченные", self.save_labeled)

        layout.addLayout(form)

        # кнопки
        btns = QHBoxLayout()
        self.btn_save = QPushButton("Сохранить")
        self.btn_reload = QPushButton("Перезагрузить из файла")
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_reload)
        layout.addLayout(btns)

        self.btn_save.clicked.connect(self.save)
        self.btn_reload.clicked.connect(self.reload)

        layout.addStretch()

    def save(self):
        s = self.cfg["settings"]
        s["interval_sec"] = int(self.interval.value())
        s["threshold"] = float(self.threshold.value())
        s["https"] = bool(self.https.isChecked())
        s["channel_path"] = self.channel.text().strip()
        s["timeout_connect"] = float(self.t_conn.value())
        s["timeout_read"] = float(self.t_read.value())
        s["gap_between_requests"] = float(self.gap.value())
        s["model_name"] = self.model.text().strip() or "buffalo_l"
        s["db_path"] = self.db_path.text().strip() or "recognition.db"
        s["save_labeled"] = bool(self.save_labeled.isChecked())
        save_config(self.cfg)
        QMessageBox.information(self, APP_TITLE, "Сохранено.")

    def reload(self):
        cfg = load_config()
        # только настройки перезальём
        self.cfg["settings"] = cfg.get("settings", self.cfg["settings"])
        s = self.cfg["settings"]
        self.interval.setValue(s.get("interval_sec", 10))
        self.threshold.setValue(s.get("threshold", 0.35))
        self.https.setChecked(s.get("https", False))
        self.channel.setText(s.get("channel_path", "/ISAPI/Streaming/channels/101/picture"))
        self.t_conn.setValue(s.get("timeout_connect", 3))
        self.t_read.setValue(s.get("timeout_read", 5))
        self.gap.setValue(s.get("gap_between_requests", 0.2))
        self.model.setText(s.get("model_name", "buffalo_l"))
        self.db_path.setText(s.get("db_path", "recognition.db"))
        self.save_labeled.setChecked(s.get("save_labeled", False))
        QMessageBox.information(self, APP_TITLE, "Перезагружено из файла.")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(900, 600)

        self.cfg = load_config()

        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        self.tab_cameras = CamerasTab(self.cfg)
        self.tab_people = PeopleTab(self.cfg)
        self.tab_settings = SettingsTab(self.cfg)
        tabs.addTab(self.tab_cameras, "Камеры")
        tabs.addTab(self.tab_people, "Сотрудники")
        tabs.addTab(self.tab_settings, "Параметры")
        layout.addWidget(tabs)

        # нижние кнопки
        btns = QHBoxLayout()
        btn_save_all = QPushButton("Сохранить всё")
        btn_reload_all = QPushButton("Перезагрузить всё из файла")
        btns.addWidget(btn_save_all)
        btns.addWidget(btn_reload_all)
        layout.addLayout(btns)

        btn_save_all.clicked.connect(lambda: (save_config(self.cfg), QMessageBox.information(self, APP_TITLE, "Сохранено.")))
        btn_reload_all.clicked.connect(self.reload_all)

    def reload_all(self):
        self.cfg.clear()
        self.cfg.update(load_config())
        self.tab_cameras.refresh_list()
        self.tab_people.refresh_people()
        self.tab_settings.reload()

def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()