import sys
import os
import json
import sqlite3
import time
import shutil
import requests
import subprocess
import base64
import stat
import threading
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog,
    QPushButton, QLineEdit, QComboBox, QHBoxLayout, QMessageBox,
    QProgressBar, QScrollArea, QInputDialog, QStackedWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QTextEdit, QFrame, QSizePolicy,
    QSplitter
)
from PyQt6.QtGui import QIcon, QPixmap, QColor, QFont, QTextCursor
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".pandu_settings.json")
DEFAULT_DB_DIR = os.path.join(os.path.expanduser("~"), ".pandu_database")
DEFAULT_AI_DB_DIR = os.path.join(os.path.expanduser("~"), ".pandu_ai_database")
LOGO_PATH = "/media/rinniro/gamessd/ScriptLense/assets/logo.png"
TEMP_ACCESS_DURATION = 300  # 5 minutes in seconds

DEFAULT_SETTINGS = {
    "writing_systems": ["Devanagari", "Cuneiform", "Hieroglyphics", "Latin", "Arabic", "Greek", "Hebrew", "Brahmi", "Phoenician"],
    "sources": ["Stone Tablet", "Clay Tablet", "Copper Plate", "Wall", "Paper"],
    "db_directory": DEFAULT_DB_DIR,
    "active_model": "",
    "gemini_api_key": "",
    "ai_mode": "local",
    "active_gemini_model": "gemini-3.5-flash"
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                return {**DEFAULT_SETTINGS, **json.load(f)}
        except Exception: pass
    return DEFAULT_SETTINGS.copy()

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f: json.dump(settings, f, indent=2)
    except Exception as e: print(f"Could not save settings: {e}")

CURRENT_SETTINGS = load_settings()

class DatabaseManager:
    _current_dir = CURRENT_SETTINGS["db_directory"]
    @classmethod
    def get_dir(cls): return cls._current_dir
    @classmethod
    def get_connection(cls):
        try:
            os.makedirs(cls._current_dir, exist_ok=True)
        except PermissionError:
            cls._current_dir = DEFAULT_DB_DIR
            os.makedirs(cls._current_dir, exist_ok=True)
            CURRENT_SETTINGS["db_directory"] = DEFAULT_DB_DIR
            save_settings(CURRENT_SETTINGS)
        return sqlite3.connect(os.path.join(cls._current_dir, "pandu.db"))
    @classmethod
    def set_dir(cls, new_dir):
        cls._current_dir = new_dir
        CURRENT_SETTINGS["db_directory"] = new_dir
        save_settings(CURRENT_SETTINGS)
        cls.init_db()
    @classmethod
    def init_db(cls):
        with cls.get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT, image_path TEXT, name TEXT,
                writing_system TEXT, time_period TEXT, region TEXT, source TEXT)""")

class AIDatabaseManager:
    _current_dir = CURRENT_SETTINGS.get("ai_db_directory", DEFAULT_AI_DB_DIR)
    @classmethod
    def get_dir(cls): return cls._current_dir
    @classmethod
    def get_connection(cls):
        try:
            os.makedirs(cls._current_dir, exist_ok=True)
        except PermissionError:
            cls._current_dir = DEFAULT_AI_DB_DIR
            os.makedirs(cls._current_dir, exist_ok=True)
            CURRENT_SETTINGS["ai_db_directory"] = DEFAULT_AI_DB_DIR
            save_settings(CURRENT_SETTINGS)
        return sqlite3.connect(os.path.join(cls._current_dir, "pandu_ai.db"))
    @classmethod
    def set_dir(cls, new_dir):
        cls._current_dir = new_dir
        CURRENT_SETTINGS["ai_db_directory"] = new_dir
        save_settings(CURRENT_SETTINGS)
        cls.init_db()
    @classmethod
    def init_db(cls):
        with cls.get_connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS ai_analysis_db (
                id INTEGER PRIMARY KEY AUTOINCREMENT, artifact_name TEXT, model_used TEXT,
                confidence_score TEXT, transcription TEXT, translation TEXT, notes TEXT)""")

DatabaseManager.init_db()
AIDatabaseManager.init_db()

def run_query(query, params=(), fetch=False):
    with DatabaseManager.get_connection() as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchall() if fetch else conn.commit()

def run_ai_query(query, params=(), fetch=False):
    with AIDatabaseManager.get_connection() as conn:
        cursor = conn.execute(query, params)
        return cursor.fetchall() if fetch else conn.commit()

# ── Permission Manager ────────────────────────────────────────────────────────

class PermissionManager:
    """Manages temporary read permissions for image files."""
    _granted_files = {}  # path -> (original_mode, expiry_time)
    _lock = threading.Lock()

    @classmethod
    def check_readable(cls, path):
        return os.access(path, os.R_OK)

    @classmethod
    def grant_temp_access(cls, path, password=None):
        """Try to grant read access. Returns (success, needs_password, error_msg)."""
        if cls.check_readable(path):
            return True, False, ""
        try:
            original_mode = os.stat(path).st_mode
            os.chmod(path, original_mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            if cls.check_readable(path):
                with cls._lock:
                    cls._granted_files[path] = (original_mode, time.time() + TEMP_ACCESS_DURATION)
                cls._schedule_revoke(path)
                return True, False, ""
        except PermissionError:
            pass
        if password is None:
            return False, True, "Password required"
        try:
            cmd = f"echo '{password}' | sudo -S chmod o+r '{path}'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and cls.check_readable(path):
                original_mode = os.stat(path).st_mode
                with cls._lock:
                    cls._granted_files[path] = (original_mode, time.time() + TEMP_ACCESS_DURATION)
                cls._schedule_revoke(path)
                return True, False, ""
            else:
                return False, True, "Incorrect password or permission denied"
        except Exception as e:
            return False, False, str(e)

    @classmethod
    def _schedule_revoke(cls, path):
        def revoke():
            time.sleep(TEMP_ACCESS_DURATION)
            cls._revoke_access(path)
        t = threading.Thread(target=revoke, daemon=True)
        t.start()

    @classmethod
    def _revoke_access(cls, path):
        with cls._lock:
            if path in cls._granted_files:
                original_mode, _ = cls._granted_files.pop(path)
                try:
                    os.chmod(path, original_mode)
                except Exception:
                    try:
                        subprocess.run(f"sudo chmod {oct(original_mode)[-3:]} '{path}'",
                                       shell=True, timeout=5)
                    except Exception:
                        pass

    @classmethod
    def revoke_all(cls):
        with cls._lock:
            paths = list(cls._granted_files.keys())
        for path in paths:
            cls._revoke_access(path)

# ── Background Workers ────────────────────────────────────────────────────────

class MigrationWorker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    migration_finished = pyqtSignal(bool, str)
    def __init__(self, source, target, mode):
        super().__init__()
        self.source, self.target, self.mode = source, target, mode
    def run(self):
        try:
            if not os.path.exists(self.source):
                self.migration_finished.emit(True, "Workspace initialized.")
                return
            files = [os.path.join(r, f) for r, _, fs in os.walk(self.source) for f in fs]
            if not files:
                os.makedirs(self.target, exist_ok=True)
                self.migration_finished.emit(True, "Empty workspace mapped successfully.")
                return
            for idx, file_path in enumerate(files):
                dest = os.path.join(self.target, os.path.relpath(file_path, self.source))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(file_path, dest)
                time.sleep(0.05)
                p = int(((idx + 1) / len(files)) * 100)
                self.progress_updated.emit(p)
                self.status_updated.emit(f"Copying files: {p}% completed...")
            if self.mode == "move": shutil.rmtree(self.source)
            self.migration_finished.emit(True, "Transfer finished successfully.")
        except Exception as e:
            self.migration_finished.emit(False, str(e))

class ImageLoaderThread(QThread):
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    finished_loading = pyqtSignal(list)
    def __init__(self, file_paths):
        super().__init__()
        self.file_paths = file_paths
    def run(self):
        valid = []
        for idx, path in enumerate(self.file_paths):
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.xpm')):
                valid.append(path)
                for p in range(1, 102, 10):
                    time.sleep(0.01)
                    self.progress_changed.emit(int(((idx + (p / 100.0)) / len(self.file_paths)) * 100))
                    self.status_changed.emit(f"Importing: {os.path.basename(path)}")
        self.finished_loading.emit(valid)

class OllamaPullWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
    def run(self):
        try:
            base_url = CURRENT_SETTINGS.get("ollama_url", "http://localhost:11434")
            payload = {"name": self.model_name, "stream": True}
            r = requests.post(f"{base_url}/api/pull", json=payload, stream=True, timeout=300)
            if r.status_code != 200:
                self.finished_signal.emit(False, f"Server error: {r.status_code}")
                return
            for line in r.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        status = data.get("status", "")
                        total = data.get("total", 0)
                        completed = data.get("completed", 0)
                        self.log_signal.emit(f"[Ollama] {status}")
                        if total and completed:
                            pct = int((completed / total) * 100)
                            self.progress_signal.emit(pct)
                        if status == "success":
                            self.finished_signal.emit(True, self.model_name)
                            return
                    except Exception:
                        pass
            self.finished_signal.emit(True, self.model_name)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class LocalChatWorker(QThread):
    chunk_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, base_url, model_name, history):
        super().__init__()
        self.base_url = base_url
        self.model_name = model_name
        self.history = history
    def run(self):
        try:
            payload = {"model": self.model_name, "messages": self.history, "stream": True}
            r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=120)
            if r.status_code != 200:
                self.finished_signal.emit(False, f"Server responded with error code {r.status_code}")
                return
            for line in r.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        content = chunk_data.get("message", {}).get("content", "")
                        if content:
                            self.chunk_ready.emit(content)
                        if chunk_data.get("done", False):
                            break
                    except Exception:
                        pass
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

def resolve_gemini_api_key(api_key):
    return (api_key or "").strip() or os.environ.get("GOOGLE_CLOUD_API_KEY", "")

def make_gemini_client(api_key):
    key = resolve_gemini_api_key(api_key)
    if not key:
        return None
    return genai.Client(vertexai=True, api_key=key)

class GeminiGenerateWorker(QThread):
    output_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, model_name, prompt, api_key):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt
        self.api_key = api_key
    def run(self):
        if genai is None:
            self.finished_signal.emit(False, "Install Gemini support with: pip install google-genai")
            return
        if not resolve_gemini_api_key(self.api_key):
            self.finished_signal.emit(False, "Enter a Google Cloud API key first (or set GOOGLE_CLOUD_API_KEY).")
            return
        try:
            client = make_gemini_client(self.api_key)
            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=self.prompt
            ):
                if chunk.text:
                    self.output_ready.emit(chunk.text)
            self.finished_signal.emit(True, self.model_name)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class GeminiImageAnalysisWorker(QThread):
    output_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, model_name, prompt, api_key, image_paths):
        super().__init__()
        self.model_name = model_name
        self.prompt = prompt
        self.api_key = api_key
        self.image_paths = image_paths
    def run(self):
        if genai is None:
            self.finished_signal.emit(False, "Install Gemini support with: pip install google-genai")
            return
        if not resolve_gemini_api_key(self.api_key):
            self.finished_signal.emit(False, "Enter a Google Cloud API key first (or set GOOGLE_CLOUD_API_KEY).")
            return
        try:
            client = make_gemini_client(self.api_key)
            parts = []
            for img_path in self.image_paths:
                if not os.path.exists(img_path):
                    continue
                ext = os.path.splitext(img_path)[1].lower()
                mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                            '.png': 'image/png', '.bmp': 'image/bmp',
                            '.gif': 'image/gif', '.webp': 'image/webp'}
                mime = mime_map.get(ext, 'image/jpeg')
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                parts.append(types.Part.from_bytes(data=base64.b64decode(img_data), mime_type=mime))
            parts.append(types.Part.from_text(text=self.prompt))
            contents = [types.Content(role="user", parts=parts)]
            for chunk in client.models.generate_content_stream(
                model=self.model_name,
                contents=contents
            ):
                if chunk.text:
                    self.output_ready.emit(chunk.text)
            self.finished_signal.emit(True, self.model_name)
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class LocalImageAnalysisWorker(QThread):
    chunk_ready = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, base_url, model_name, prompt, image_paths):
        super().__init__()
        self.base_url = base_url
        self.model_name = model_name
        self.prompt = prompt
        self.image_paths = image_paths
    def run(self):
        try:
            images_b64 = []
            for img_path in self.image_paths:
                if os.path.exists(img_path):
                    with open(img_path, 'rb') as f:
                        images_b64.append(base64.b64encode(f.read()).decode('utf-8'))
            message = {"role": "user", "content": self.prompt}
            if images_b64:
                message["images"] = images_b64
            payload = {"model": self.model_name, "messages": [message], "stream": True}
            r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=120)
            if r.status_code != 200:
                self.finished_signal.emit(False, f"Server responded with error code {r.status_code}")
                return
            for line in r.iter_lines():
                if line:
                    try:
                        chunk_data = json.loads(line.decode('utf-8'))
                        content = chunk_data.get("message", {}).get("content", "")
                        if content:
                            self.chunk_ready.emit(content)
                        if chunk_data.get("done", False):
                            break
                    except Exception:
                        pass
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))

class TerminalWorker(QThread):
    output_ready = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, command):
        super().__init__()
        self.command = command
        self._process = None
    def run(self):
        if not self.command:
            self.finished_signal.emit()
            return
        try:
            self._process = subprocess.Popen(
                self.command, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in self._process.stdout:
                self.output_ready.emit(line)
            self._process.stdout.close()
            self._process.wait()
        except Exception as e:
            self.output_ready.emit(f"Shell execution error: {str(e)}\n")
        self.finished_signal.emit()
    def kill(self):
        if self._process and self._process.poll() is None:
            self._process.kill()

# ── Shared UI Helpers ─────────────────────────────────────────────────────────

class ChatInput(QTextEdit):
    sendRequested = pyqtSignal()
    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self.sendRequested.emit()
            return
        super().keyPressEvent(event)

class FileTransferProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Data Relocation")
        self.setFixedSize(400, 110)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint)
        self.setModal(True)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Initializing tracking pathways...")
        self.progress_bar = QProgressBar()
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

class MigrationChoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Migration Action")
        self.choice = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("How would you like to handle your existing database files?"))
        btn_layout = QHBoxLayout()
        for action in ["Copy", "Move", "Cancel"]:
            btn = QPushButton(f"{action} Data" if action != "Cancel" else action)
            btn.clicked.connect(lambda checked, a=action.lower(): self.handle_click(a))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)
    def handle_click(self, action):
        if action != "cancel": self.choice = action; self.accept()
        else: self.reject()

class PasswordDialog(QDialog):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Permission Required")
        self.setFixedSize(420, 160)
        self.setModal(True)
        self.password = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Elevated permission needed to read:"))
        path_lbl = QLabel(f"<b>{os.path.basename(file_path)}</b>")
        path_lbl.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(path_lbl)
        layout.addWidget(QLabel("Enter your system password to grant 5-minute temporary read access:"))
        self.pwd_input = QLineEdit()
        self.pwd_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.pwd_input.setPlaceholderText("System password...")
        layout.addWidget(self.pwd_input)
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("Grant Access")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept_password)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)
    def accept_password(self):
        self.password = self.pwd_input.text()
        self.accept()

class SecurityWarningDialog(QDialog):
    def __init__(self, concerns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security Warning")
        self.setFixedSize(550, 400)
        self.setModal(True)
        layout = QVBoxLayout(self)
        warn_lbl = QLabel("⚠  Security Notice")
        warn_lbl.setStyleSheet("color: #ffaa00; font-size: 14px; font-weight: bold;")
        layout.addWidget(warn_lbl)
        layout.addWidget(QLabel("The following security concerns apply to this operation:"))
        concerns_text = QTextEdit()
        concerns_text.setReadOnly(True)
        concerns_text.setStyleSheet("background: #0d0d0d; border: 1px solid #2a2a2a; color: #cccccc;")
        concerns_text.setPlainText(concerns)
        layout.addWidget(concerns_text)
        layout.addWidget(QLabel("Do you want to proceed?"))
        btn_row = QHBoxLayout()
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet("QPushButton { background: #152515; border-color: #254525; color: #aaffaa; }")
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("QPushButton { background: #250505; color: #ff6666; border-color: #4a1515; }")
        proceed_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(proceed_btn)
        layout.addLayout(btn_row)

class EditEntryDialog(QDialog):
    def __init__(self, row_data, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Entry")
        self.settings = settings
        self.entry_id, self.image_path, name, ws, tp, region, src = row_data
        layout = QVBoxLayout(self)
        self.inputs = {
            "name": QLineEdit(name), "ws": QComboBox(), "tp": QLineEdit(tp),
            "region": QLineEdit(region), "src": QComboBox()
        }
        self.inputs["ws"].addItems(settings["writing_systems"])
        self.inputs["ws"].setCurrentText(ws)
        self.inputs["src"].addItems(settings["sources"])
        self.inputs["src"].setCurrentText(src)
        for label, widget in [("Name:", self.inputs["name"]),
                               ("Writing System:", self.inputs["ws"]),
                               ("Time Period:", self.inputs["tp"]),
                               ("Region:", self.inputs["region"]),
                               ("Source:", self.inputs["src"])]:
            layout.addWidget(QLabel(label)); layout.addWidget(widget)
        img_row = QHBoxLayout()
        self.img_lbl = QLabel(os.path.basename(self.image_path) if self.image_path else "No image")
        change_btn = QPushButton("Change Image")
        change_btn.clicked.connect(self.change_image)
        img_row.addWidget(self.img_lbl); img_row.addWidget(change_btn)
        layout.addLayout(img_row)
        btn_row = QHBoxLayout()
        save, cancel = QPushButton("Save Changes"), QPushButton("Cancel")
        save.clicked.connect(self.accept); cancel.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel); btn_row.addWidget(save)
        layout.addLayout(btn_row)
    def change_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if path: self.image_path = path; self.img_lbl.setText(os.path.basename(path))
    def get_data(self):
        return (self.image_path, self.inputs["name"].text(),
                self.inputs["ws"].currentText(), self.inputs["tp"].text(),
                self.inputs["region"].text(), self.inputs["src"].currentText(),
                self.entry_id)

class DragDropItemWidget(QWidget):
    itemDeleted = pyqtSignal(str)
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        lbl = QLabel(f"🖼️ {os.path.basename(file_path)}")
        lbl.setStyleSheet("color: white; font-size: 12px;")
        cross = QPushButton("×")
        cross.setFixedSize(18, 18)
        cross.setStyleSheet("QPushButton { border: none; color: #ff6b6b; font-weight: bold; }"
                            "QPushButton:hover { color: red; }")
        cross.clicked.connect(lambda: self.itemDeleted.emit(self.file_path))
        layout.addWidget(lbl); layout.addStretch(); layout.addWidget(cross)

class PlaceholderPage(QWidget):
    def __init__(self, title):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(QLabel(f"{title} — coming soon"))

# ── Integrated Data Entry Page ────────────────────────────────────────────────

class ImageUploadWidget(QWidget):
    imagesDropped = pyqtSignal(list)
    imageRemoved = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        layout = QVBoxLayout(self)
        self.container = QWidget()
        self.container.setMinimumSize(320, 180)
        self.container.setObjectName("DropContainer")
        self.container.setStyleSheet(
            "QWidget#DropContainer { border: 2px dashed #b0b0b0;"
            " border-radius: 8px; background: #505050; }")
        c_layout = QVBoxLayout(self.container)
        self.hint = QLabel("Drag and Drop multiple images here")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setStyleSheet("color: white;")
        c_layout.addWidget(self.hint)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background: transparent; border: none;")
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        c_layout.addWidget(self.scroll)
        self.scroll.hide()
        layout.addWidget(self.container)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse)
        layout.addWidget(browse, alignment=Qt.AlignmentFlag.AlignRight)
    def update_list(self, paths):
        while self.scroll_layout.count():
            w = self.scroll_layout.takeAt(0).widget()
            if w: w.deleteLater()
        if not paths:
            self.scroll.hide(); self.hint.show(); return
        self.hint.hide(); self.scroll.show()
        for p in paths:
            item = DragDropItemWidget(p)
            item.itemDeleted.connect(self.imageRemoved.emit)
            self.scroll_layout.addWidget(item)
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()
    def dropEvent(self, e):
        self.imagesDropped.emit([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])
    def browse(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if paths: self.imagesDropped.emit(paths)

class DataPage(QWidget):
    goToLibrary = pyqtSignal(dict)
    def __init__(self):
        super().__init__()
        self.staged = []
        self.settings = load_settings()
        self.img_path = ""
        layout = QVBoxLayout(self)
        self.uploader = ImageUploadWidget()
        self.uploader.imagesDropped.connect(self.process_images)
        self.uploader.imageRemoved.connect(self.remove_image)
        layout.addWidget(self.uploader)
        self.p_box = QWidget()
        p_lay = QVBoxLayout(self.p_box)
        self.lbl = QLabel("Status: Awaiting files...")
        self.bar = QProgressBar()
        p_lay.addWidget(self.lbl); p_lay.addWidget(self.bar)
        layout.addWidget(self.p_box)
        self.p_box.hide()
        form_frame = QFrame()
        form_frame.setStyleSheet("QFrame { background: #0d0d0d; border: 1px solid #1f1f1f; border-radius: 6px; }")
        form_lay = QVBoxLayout(form_frame)
        self.name = QLineEdit()
        self.ws = QComboBox(); self.ws.addItems(self.settings["writing_systems"])
        self.start_yr = QLineEdit(); self.end_yr = QLineEdit()
        self.era = QComboBox(); self.era.addItems(["BCE", "CE"])
        self.region = QLineEdit()
        self.src = QComboBox(); self.src.addItems(self.settings["sources"])
        form_lay.addWidget(QLabel("Name:")); form_lay.addWidget(self.name)
        ws_row = QHBoxLayout(); ws_row.addWidget(self.ws)
        add_ws = QPushButton("+")
        add_ws.setFixedWidth(30)
        add_ws.clicked.connect(lambda: self.add_setting("writing_systems", self.ws))
        ws_row.addWidget(add_ws)
        form_lay.addWidget(QLabel("Writing System:")); form_lay.addLayout(ws_row)
        t_row = QHBoxLayout()
        t_row.addWidget(self.start_yr); t_row.addWidget(self.end_yr)
        t_row.addWidget(self.era)
        form_lay.addWidget(QLabel("Time Period:")); form_lay.addLayout(t_row)
        form_lay.addWidget(QLabel("Region:")); form_lay.addWidget(self.region)
        src_row = QHBoxLayout(); src_row.addWidget(self.src)
        add_src = QPushButton("+")
        add_src.setFixedWidth(30)
        add_src.clicked.connect(lambda: self.add_setting("sources", self.src))
        src_row.addWidget(add_src)
        form_lay.addWidget(QLabel("Source:")); form_lay.addLayout(src_row)
        layout.addWidget(form_frame)
        self.save_btn = QPushButton("Save to Database ✓")
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("QPushButton { background: #152515; border-color: #254525; color: #aaffaa; } QPushButton:disabled { background: #111; color: #444; }")
        self.save_btn.clicked.connect(self.save_metadata)
        layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignRight)
    def process_images(self, paths):
        unique = [p for p in paths if p not in self.staged]
        if not unique: return
        self.p_box.show()
        self.thread = ImageLoaderThread(unique)
        self.thread.progress_changed.connect(self.bar.setValue)
        self.thread.status_changed.connect(self.lbl.setText)
        self.thread.finished_loading.connect(self.loading_done)
        self.thread.start()
    def loading_done(self, valid):
        if valid:
            self.staged.extend(valid)
            self.uploader.update_list(self.staged)
            self.img_path = self.staged[0]
            self.save_btn.setEnabled(True)
        else: QMessageBox.warning(self, "Error", "No compatible images added.")
        self.p_box.hide()
    def remove_image(self, path):
        if path in self.staged: self.staged.remove(path)
        self.uploader.update_list(self.staged)
        self.save_btn.setEnabled(bool(self.staged))
        self.img_path = self.staged[0] if self.staged else ""
        if not self.staged: self.p_box.hide()
    def save_metadata(self):
        data = {
            "image_path": self.img_path,
            "name": self.name.text().strip() or "Unnamed Artifact",
            "writing_system": self.ws.currentText(),
            "time_period": f"{self.start_yr.text()} - {self.end_yr.text()} {self.era.currentText()}",
            "region": self.region.text().strip(),
            "source": self.src.currentText()
        }
        run_query("INSERT INTO entries (image_path,name,writing_system,time_period,region,source)"
                  " VALUES (?,?,?,?,?,?)", list(data.values()))
        self.name.clear(); self.start_yr.clear()
        self.end_yr.clear(); self.region.clear()
        self.reset_page()
        self.goToLibrary.emit(data)
    def add_setting(self, key, combo):
        txt, ok = QInputDialog.getText(self, "Add New Entry", "Enter Value:")
        if ok and txt.strip() and txt.strip() not in self.settings[key]:
            self.settings[key].append(txt.strip())
            combo.addItem(txt.strip()); combo.setCurrentText(txt.strip())
            save_settings(self.settings)
    def reset_page(self):
        self.staged.clear(); self.uploader.update_list([])
        self.save_btn.setEnabled(False); self.p_box.hide(); self.img_path = ""

class LibraryPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        ref = QPushButton("⟳ Refresh"); ref.clicked.connect(self.load_data)
        chg = QPushButton("📂 Change Library Dir")
        chg.clicked.connect(lambda: self.migrate(QFileDialog.getExistingDirectory(self, "Select Folder"), "library"))
        df = QPushButton("🔄 Reset Library Default")
        df.clicked.connect(lambda: self.migrate(DEFAULT_DB_DIR, "library"))
        for w in [ref, QFrame(), chg, df]:
            if isinstance(w, QFrame):
                w.setFrameShape(QFrame.Shape.VLine); w.setFrameShadow(QFrame.Shadow.Sunken)
            top.addWidget(w)
        top.addStretch(); layout.addLayout(top)
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["#", "Image", "Name", "Writing System", "Time Period", "Region", "Source", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        self.load_data()
    def load_data(self):
        self.table.setRowCount(0)
        for row in run_query("SELECT * FROM entries ORDER BY id", fetch=True):
            r_idx = self.table.rowCount(); self.table.insertRow(r_idx)
            self.table.setItem(r_idx, 0, QTableWidgetItem(str(row[0])))
            img = QPushButton("🖼")
            img.clicked.connect(lambda checked, p=row[1]: self.view_image(p))
            self.table.setCellWidget(r_idx, 1, img)
            for i in range(2, 7): self.table.setItem(r_idx, i, QTableWidgetItem(row[i] or ""))
            act = QWidget(); a_lay = QHBoxLayout(act); a_lay.setContentsMargins(2, 2, 2, 2)
            e_btn, d_btn = QPushButton("Edit"), QPushButton("Delete")
            e_btn.clicked.connect(lambda checked, r=row: self.edit_row(r))
            d_btn.clicked.connect(lambda checked, i=row[0]: self.delete_row(i))
            a_lay.addWidget(e_btn); a_lay.addWidget(d_btn); self.table.setCellWidget(r_idx, 7, act)
    def view_image(self, path):
        d = QDialog(self); d.setWindowTitle("Image View"); l = QVBoxLayout(d); lbl = QLabel()
        if path and os.path.exists(path):
            lbl.setPixmap(QPixmap(path).scaled(480, 380, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else: lbl.setText("Not found.")
        l.addWidget(lbl); d.exec()
    def edit_row(self, row):
        d = EditEntryDialog(row, load_settings(), self)
        if d.exec() == QDialog.DialogCode.Accepted:
            run_query("UPDATE entries SET image_path=?,name=?,writing_system=?,time_period=?,region=?,source=? WHERE id=?", d.get_data())
            self.load_data()
    def delete_row(self, r_id):
        if QMessageBox.question(self, 'Delete', 'Delete row?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            run_query("DELETE FROM entries WHERE id = ?", (r_id,))
            self.load_data()
    def migrate(self, path, target_type="library"):
        current_dir = DatabaseManager.get_dir() if target_type == "library" else AIDatabaseManager.get_dir()
        if not path or path == current_dir: return
        c = MigrationChoiceDialog(self)
        if c.exec() != QDialog.DialogCode.Accepted: return
        self.pd = FileTransferProgressDialog(self)
        self.worker = MigrationWorker(current_dir, path, c.choice)
        self.worker.progress_updated.connect(self.pd.progress_bar.setValue)
        self.worker.status_updated.connect(self.pd.status_label.setText)
        self.worker.migration_finished.connect(lambda s, m: self.mig_done(s, m, path, target_type))
        self.worker.start(); self.pd.exec()
    def mig_done(self, success, msg, path, target_type):
        self.pd.accept()
        if success:
            if target_type == "library":
                DatabaseManager.set_dir(path); self.load_data()
            else:
                AIDatabaseManager.set_dir(path)
                CURRENT_SETTINGS["ai_db_directory"] = path
                save_settings(CURRENT_SETTINGS)
            QMessageBox.information(self, "Success", f"{target_type.capitalize()} database relocated successfully.")
        else: QMessageBox.critical(self, "Error", f"Failed: {msg}")

# ── Chat Bubble ───────────────────────────────────────────────────────────────

class ChatBubble(QFrame):
    def __init__(self, text: str, is_user: bool, parent=None):
        super().__init__(parent)
        self._text = text
        self.is_user = is_user
        outer = QHBoxLayout(self); outer.setContentsMargins(4, 2, 4, 2); outer.setSpacing(0)
        col = QVBoxLayout(); col.setSpacing(2)
        box = QFrame()
        if is_user:
            box.setStyleSheet("QFrame { background: #0a0a0a; border: 1px solid #bb4400; border-radius: 8px; }")
        else:
            box.setStyleSheet("QFrame { background: #030f0f; border: 1px solid #007799; border-radius: 8px; }")
        box_lay = QVBoxLayout(box); box_lay.setContentsMargins(10, 8, 10, 8)
        self._lbl = QLabel(text); self._lbl.setWordWrap(True)
        self._lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        if is_user:
            self._lbl.setStyleSheet("color: #f0c090; font-size: 12px; background: transparent; border: none;")
        else:
            self._lbl.setStyleSheet("color: #7fffee; font-size: 12px; font-family: 'Consolas', monospace; background: transparent; border: none;")
        box_lay.addWidget(self._lbl); col.addWidget(box)
        copy_row = QHBoxLayout()
        copy_btn = QPushButton("⎘ copy"); copy_btn.setFixedHeight(14)
        copy_btn.setStyleSheet("QPushButton { background: transparent; border: none; color: #2e2e2e; font-size: 9px; padding: 0 2px; } QPushButton:hover { color: #888888; }")
        copy_btn.clicked.connect(self._copy)
        if is_user: copy_row.addStretch(); copy_row.addWidget(copy_btn)
        else: copy_row.addWidget(copy_btn); copy_row.addStretch()
        col.addLayout(copy_row)
        if is_user: outer.addStretch(); outer.addLayout(col)
        else: outer.addLayout(col); outer.addStretch()
    def _copy(self): QApplication.clipboard().setText(self._text)
    def append_text(self, chunk: str):
        self._text += chunk
        self._lbl.setText(self._text)

# ── AI Analysis Page ──────────────────────────────────────────────────────────

class AIAnalysisPage(QWidget):
    progress_updated = pyqtSignal(int)  # percentage of analysis progress
    analysis_completed = pyqtSignal(bool)  # True when analysis done
    analysis_paused_state = pyqtSignal(bool)  # True when paused
    analysis_stopped_state = pyqtSignal(bool)  # True when stopped

    def __init__(self):
        super().__init__()
        self._ai_mode = CURRENT_SETTINGS.get("ai_mode", "local")
        self.gemini_chat_history = []
        self._local_history = []
        self._current_cloud_ai_bubble = None
        self._current_local_ai_bubble = None
        self._local_chat_thread = None
        self._gemini_worker = None
        self._image_analysis_worker = None
        self._analysis_paused = False
        self._analysis_stopped = False
        self._analysis_result_buffer = ""
        self._analysis_total_chunks = 0
        self._analysis_chunks_received = 0
        self._analysis_saved_count = 0
        self.setup_ui()
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_ollama_status)
        self.status_timer.start(4000)
        self.check_ollama_status()

    def _sec(self, text):
        l = QLabel(text)
        l.setStyleSheet("color: #383838; font-size: 9px; font-weight: bold; letter-spacing: 2px; margin-top: 6px;")
        return l

    def _card(self):
        f = QFrame()
        f.setStyleSheet("QFrame { background: #0d0d0d; border: 1px solid #1a1a1a; border-radius: 5px; }")
        lay = QVBoxLayout(f); lay.setContentsMargins(10, 8, 10, 8); lay.setSpacing(5)
        return f, lay

    def setup_ui(self):
        self.setStyleSheet("""
            QWidget { background: #070707; color: #b0b0b0; font-family: 'Segoe UI', Arial; font-size: 12px; }
            QLineEdit { background: #0f0f0f; border: 1px solid #252525; border-radius: 4px; padding: 5px 8px; color: #dddddd; }
            QLineEdit:focus { border-color: #bb4400; }
            QPushButton { background: #111111; border: 1px solid #282828; border-radius: 4px; padding: 4px 12px; color: #aaaaaa; font-weight: bold; }
            QPushButton:hover { background: #181818; border-color: #404040; }
            QPushButton:disabled { color: #282828; border-color: #151515; background: #0a0a0a; }
            QComboBox { background: #0f0f0f; border: 1px solid #252525; border-radius: 4px; padding: 4px 8px; color: #cccccc; }
            QComboBox::drop-down { border: none; }
            QTextEdit { background: #050505; border: 1px solid #121212; border-radius: 4px; color: #00ff00; font-family: 'Consolas', monospace; }
        """)
        root = QVBoxLayout(self); root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        top = QHBoxLayout()
        self.banner = QFrame(); self.banner.setFixedHeight(28)
        b_lay = QHBoxLayout(self.banner); b_lay.setContentsMargins(10, 0, 10, 0)
        self.banner_text = QLabel("Checking models...")
        self.deactivate_btn = QPushButton("Turn Off"); self.deactivate_btn.setFixedSize(64, 18)
        self.deactivate_btn.setStyleSheet("QPushButton { background: #200808; border: 1px solid #3a1010; color: #aa3333; font-size: 9px; border-radius: 3px; padding: 1px 8px; }")
        self.deactivate_btn.clicked.connect(self.deactivate_model)
        b_lay.addWidget(self.banner_text); b_lay.addStretch(); b_lay.addWidget(self.deactivate_btn)
        top.addWidget(self.banner)
        self.toggle_mode_btn = QPushButton("Switch Mode")
        self.toggle_mode_btn.setFixedSize(110, 28)
        self.toggle_mode_btn.clicked.connect(self.toggle_ai_mode)
        top.addWidget(self.toggle_mode_btn)
        root.addLayout(top)

        self.pages_stack = QStackedWidget()
        self.local_panel = self._build_local_panel()
        self.cloud_panel = self._build_cloud_panel()
        self.pages_stack.addWidget(self.local_panel)
        self.pages_stack.addWidget(self.cloud_panel)
        root.addWidget(self.pages_stack, 1)
        self.update_banner_style()

    def toggle_ai_mode(self):
        self._ai_mode = "cloud" if self._ai_mode == "local" else "local"
        CURRENT_SETTINGS["ai_mode"] = self._ai_mode
        save_settings(CURRENT_SETTINGS)
        self.update_banner_style()

    def update_banner_style(self):
        if self._ai_mode == "local":
            self.pages_stack.setCurrentIndex(0)
            act = CURRENT_SETTINGS.get("active_model", "")
            if act and not act.startswith("gemini:"):
                self.banner.setStyleSheet("QFrame { background: #121008; border: 1px solid #3a3010; border-radius: 4px; }")
                self.banner_text.setText(f"Active Local Model Path: {act}")
                self.deactivate_btn.show()
            else:
                self.banner.setStyleSheet("QFrame { background: #120808; border: 1px solid #3a1010; border-radius: 4px; }")
                self.banner_text.setText("No local execution model targeted.")
                self.deactivate_btn.hide()
        else:
            self.pages_stack.setCurrentIndex(1)
            act = CURRENT_SETTINGS.get("active_model", "")
            if act and act.startswith("gemini:"):
                self.banner.setStyleSheet("QFrame { background: #081212; border: 1px solid #103a3a; border-radius: 4px; }")
                self.banner_text.setText(f"Active Cloud Provider Instance: {act.split(':', 1)[1]}")
                self.deactivate_btn.show()
            else:
                self.banner.setStyleSheet("QFrame { background: #120808; border: 1px solid #3a1010; border-radius: 4px; }")
                self.banner_text.setText("Cloud architecture initialized but offline.")
                self.deactivate_btn.hide()

    def activate_model(self, model_identifier):
        CURRENT_SETTINGS["active_model"] = model_identifier
        save_settings(CURRENT_SETTINGS)
        self.update_banner_style()

    def deactivate_model(self):
        CURRENT_SETTINGS["active_model"] = ""
        save_settings(CURRENT_SETTINGS)
        self.update_banner_style()

    def _build_image_analysis_panel(self):
        """Shared image analysis panel embedded in both local and cloud panels."""
        frame, lay = self._card()
        lay.addWidget(self._sec("IMAGE ANALYSIS FROM LIBRARY DATABASE"))

        # Library image selector
        sel_row = QHBoxLayout()
        self.img_selector_combo = QComboBox()
        self.img_selector_combo.setMinimumWidth(200)
        refresh_img_btn = QPushButton("⟳")
        refresh_img_btn.setFixedWidth(30)
        refresh_img_btn.clicked.connect(self.refresh_library_images)
        sel_row.addWidget(QLabel("Select Image:"))
        sel_row.addWidget(self.img_selector_combo, 1)
        sel_row.addWidget(refresh_img_btn)
        lay.addLayout(sel_row)

        # Access status
        self.access_status_lbl = QLabel("Access: Not checked")
        self.access_status_lbl.setStyleSheet("color: #555555; font-size: 10px;")
        lay.addWidget(self.access_status_lbl)

        # Timer label
        self.access_timer_lbl = QLabel("")
        self.access_timer_lbl.setStyleSheet("color: #557755; font-size: 10px;")
        lay.addWidget(self.access_timer_lbl)
        self._access_timer = QTimer(self)
        self._access_timer.timeout.connect(self._update_access_timer)
        self._access_expiry = None

        # Analysis prompt area
        lay.addWidget(self._sec("ANALYSIS PROMPT"))
        self.analysis_prompt_input = QTextEdit()
        self.analysis_prompt_input.setFixedHeight(70)
        self.analysis_prompt_input.setPlaceholderText(
            "Describe what you want the AI to analyse in the selected image(s)...\n"
            "e.g. 'Transcribe all visible text', 'Identify the writing system', 'Describe the artifact'")
        self.analysis_prompt_input.setStyleSheet(
            "QTextEdit { background: #0a0a0a; border: 1px solid #1e1e1e; color: #cccccc; font-family: 'Segoe UI'; }")
        lay.addWidget(self.analysis_prompt_input)

        # Analyse, Pause, Stop buttons in a row
        btn_row = QHBoxLayout()
        analyse_btn = QPushButton("▶  Analyse Image with AI")
        analyse_btn.setStyleSheet(
            "QPushButton { background: #0a1020; border: 1px solid #1a3060; color: #4488ff; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background: #0d1830; border-color: #2255aa; }"
            "QPushButton:disabled { color: #222; border-color: #111; background: #080808; }")
        analyse_btn.clicked.connect(self.run_image_analysis)
        btn_row.addWidget(analyse_btn, 3)
        self.analyse_btn = analyse_btn

        pause_btn = QPushButton("⏸  Pause")
        pause_btn.setStyleSheet(
            "QPushButton { background: #201808; border: 1px solid #604010; color: #ffaa33; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background: #302010; }"
            "QPushButton:disabled { color: #222; border-color: #111; background: #080808; }")
        pause_btn.clicked.connect(self.pause_analysis)
        btn_row.addWidget(pause_btn, 1)
        self.pause_btn = pause_btn

        stop_btn = QPushButton("⏹  Stop")
        stop_btn.setStyleSheet(
            "QPushButton { background: #200808; border: 1px solid #501010; color: #ff4444; font-weight: bold; padding: 6px; }"
            "QPushButton:hover { background: #301010; }"
            "QPushButton:disabled { color: #222; border-color: #111; background: #080808; }")
        stop_btn.clicked.connect(self.stop_analysis)
        btn_row.addWidget(stop_btn, 1)
        self.stop_btn = stop_btn

        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        lay.addLayout(btn_row)

        self.refresh_library_images()
        return frame

    def refresh_library_images(self):
        self.img_selector_combo.clear()
        rows = run_query("SELECT id, name, image_path FROM entries ORDER BY id", fetch=True)
        for row in rows:
            entry_id, name, img_path = row
            label = f"[{entry_id}] {name or 'Unnamed'} — {os.path.basename(img_path or '')}"
            self.img_selector_combo.addItem(label, userData=img_path)
        if self.img_selector_combo.count() == 0:
            self.img_selector_combo.addItem("No images in library", userData=None)
        self._check_current_image_access()

    def _check_current_image_access(self):
        img_path = self.img_selector_combo.currentData()
        if not img_path:
            self.access_status_lbl.setText("Access: No image selected")
            self.access_status_lbl.setStyleSheet("color: #555555; font-size: 10px;")
            return
        if not os.path.exists(img_path):
            self.access_status_lbl.setText("Access: File not found on disk")
            self.access_status_lbl.setStyleSheet("color: #aa3333; font-size: 10px;")
        elif PermissionManager.check_readable(img_path):
            self.access_status_lbl.setText("Access: Readable ✓")
            self.access_status_lbl.setStyleSheet("color: #33aa33; font-size: 10px;")
        else:
            self.access_status_lbl.setText("Access: Permission required")
            self.access_status_lbl.setStyleSheet("color: #aa6600; font-size: 10px;")

    def _update_access_timer(self):
        if self._access_expiry is None:
            self._access_timer.stop()
            self.access_timer_lbl.setText("")
            return
        remaining = int(self._access_expiry - time.time())
        if remaining <= 0:
            self._access_timer.stop()
            self._access_expiry = None
            self.access_timer_lbl.setText("")
            self._check_current_image_access()
        else:
            mins, secs = divmod(remaining, 60)
            self.access_timer_lbl.setText(f"Temporary access expires in: {mins:02d}:{secs:02d}")

    def _build_local_panel(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        body = QHBoxLayout(); body.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(6)

        # Add scroll area for the left menu
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        left_scroll_content = QWidget()
        left_scroll_layout = QVBoxLayout(left_scroll_content)
        left_scroll_layout.setSpacing(6)

        card1, l1 = self._card()
        l1.addWidget(self._sec("TARGET SYSTEM PATH"))
        self.ollama_url_input = QLineEdit("http://localhost:11434")
        l1.addWidget(self.ollama_url_input)
        left_scroll_layout.addWidget(card1)

        card2, l2 = self._card()
        l2.addWidget(self._sec("AVAILABLE ENDPOINTS"))
        # Make endpoints smaller - side by side with download
        endpoints_row = QHBoxLayout()
        self.local_models_combo = QComboBox()
        self.local_models_combo.setMinimumWidth(120)
        endpoints_row.addWidget(self.local_models_combo)
        self.use_local_btn = QPushButton("Target")
        self.use_local_btn.setFixedWidth(60)
        self.use_local_btn.clicked.connect(self.target_local_model)
        self.refresh_local_btn = QPushButton("⟳")
        self.refresh_local_btn.setFixedWidth(28)
        self.refresh_local_btn.clicked.connect(self.check_ollama_status)
        endpoints_row.addWidget(self.use_local_btn)
        endpoints_row.addWidget(self.refresh_local_btn)
        l2.addLayout(endpoints_row)
        left_scroll_layout.addWidget(card2)

        card3, l3 = self._card()
        l3.addWidget(self._sec("DOWNLOAD CORE REMOTE WEIGHTS"))
        # Download side by side with endpoints area
        download_row = QHBoxLayout()
        self.pull_model_input = QLineEdit()
        self.pull_model_input.setPlaceholderText("e.g., llama3, mistral...")
        download_row.addWidget(self.pull_model_input)
        self.pull_btn = QPushButton("Pull")
        self.pull_btn.setFixedWidth(50)
        self.pull_btn.clicked.connect(self.start_ollama_pull)
        download_row.addWidget(self.pull_btn)
        l3.addLayout(download_row)
        self.pull_progress = QProgressBar()
        self.pull_progress.setFixedHeight(10)
        self.pull_progress.hide()
        l3.addWidget(self.pull_progress)
        left_scroll_layout.addWidget(card3)

        # Image analysis panel (no longer in separate analysis section of menu)
        left_scroll_layout.addWidget(self._build_image_analysis_panel())
        left_scroll_layout.addStretch()

        left_scroll.setWidget(left_scroll_content)
        left.addWidget(left_scroll, 4)

        body.addLayout(left, 4)

        right = QVBoxLayout(); right.setSpacing(6)
        right.addWidget(self._sec("INTEGRATED LOCAL CHAT INTERFACE"))
        self.local_chat_scroll = QScrollArea()
        self.local_chat_scroll.setWidgetResizable(True)
        self.local_chat_scroll.setStyleSheet("background: #050505; border: 1px solid #111;")
        self.local_chat_widget = QWidget()
        self.local_chat_layout = QVBoxLayout(self.local_chat_widget)
        self.local_chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.local_chat_scroll.setWidget(self.local_chat_widget)
        right.addWidget(self.local_chat_scroll, 1)
        chat_inp_row = QHBoxLayout(); chat_inp_row.setSpacing(6)
        self.local_msg_input = ChatInput()
        self.local_msg_input.setFixedHeight(40)
        self.local_msg_input.setPlaceholderText("Type local engine message & press Enter...")
        self.local_msg_input.sendRequested.connect(self.submit_embedded_local_chat)
        self.local_send_btn = QPushButton("Send")
        self.local_send_btn.setFixedHeight(40)
        self.local_send_btn.clicked.connect(self.submit_embedded_local_chat)
        chat_inp_row.addWidget(self.local_msg_input, 1); chat_inp_row.addWidget(self.local_send_btn)
        right.addLayout(chat_inp_row)
        body.addLayout(right, 6)
        lay.addLayout(body)

        # Terminal at the bottom (replaces the old system diagnostic log)
        lay.addWidget(self._build_terminal_panel())
        return w

    def _build_terminal_panel(self):
        """Build a VS Code-like terminal panel."""
        frame, lay = self._card()
        lay.addWidget(self._sec("TERMINAL"))
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFixedHeight(120)
        self.terminal_output.setStyleSheet(
            "QTextEdit { background: #0a0a0a; border: 1px solid #252525; color: #00ff00; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }")
        self.terminal_output.setPlainText("Pandu Terminal v1.0\nType a command below and press Enter.\n")
        lay.addWidget(self.terminal_output)

        cmd_row = QHBoxLayout()
        # Terminal prompt label
        prompt_lbl = QLabel("$")
        prompt_lbl.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 12px;")
        cmd_row.addWidget(prompt_lbl)

        self.terminal_cmd_input = QLineEdit()
        self.terminal_cmd_input.setPlaceholderText("Enter command...")
        self.terminal_cmd_input.setStyleSheet(
            "QLineEdit { background: #0a0a0a; border: 1px solid #252525; color: #00ff00; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; }")
        self.terminal_cmd_input.returnPressed.connect(self.execute_terminal_command)
        cmd_row.addWidget(self.terminal_cmd_input, 1)

        self.term_clear_btn = QPushButton("Clear")
        self.term_clear_btn.setFixedWidth(50)
        self.term_clear_btn.clicked.connect(lambda: self.terminal_output.setPlainText(""))
        cmd_row.addWidget(self.term_clear_btn)

        lay.addLayout(cmd_row)
        self._current_term_worker = None
        return frame

    def execute_terminal_command(self):
        cmd = self.terminal_cmd_input.text().strip()
        if not cmd:
            return
        self.terminal_output.append(f"$ {cmd}")
        self.terminal_cmd_input.clear()
        if self._current_term_worker and self._current_term_worker.isRunning():
            self.terminal_output.append("A command is already running. Wait or restart.")
            return
        self._current_term_worker = TerminalWorker(cmd)
        self._current_term_worker.output_ready.connect(lambda txt: self.terminal_output.insertPlainText(txt))
        self._current_term_worker.finished_signal.connect(self._on_terminal_finished)
        self._current_term_worker.start()

    def _on_terminal_finished(self):
        self._current_term_worker = None

    def _build_cloud_panel(self):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        body = QHBoxLayout(); body.setSpacing(10)
        left = QVBoxLayout(); left.setSpacing(6)

        # Add scroll area for the left menu (same as local panel)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        left_scroll_content = QWidget()
        left_scroll_layout = QVBoxLayout(left_scroll_content)
        left_scroll_layout.setSpacing(6)

        # Provider and model selection - change models based on provider
        prov_card, prov_lay = self._card()
        prov_lay.addWidget(self._sec("PROVIDER"))
        self.api_provider_combo = QComboBox()
        self.api_provider_combo.addItems(["Gemini", "OpenAI", "Anthropic", "Mistral"])
        self.api_provider_combo.currentTextChanged.connect(self._update_cloud_models)
        prov_lay.addWidget(self.api_provider_combo)
        left_scroll_layout.addWidget(prov_card)

        api_card, api_lay = self._card()
        api_lay.addWidget(self._sec("MODEL"))
        model_row = QHBoxLayout(); model_row.setSpacing(6)
        self.cloud_model_combo = QComboBox()
        self._update_cloud_models("Gemini")
        saved_model = CURRENT_SETTINGS.get("active_gemini_model", "gemini-3.5-flash")
        idx = self.cloud_model_combo.findText(saved_model)
        if idx >= 0: self.cloud_model_combo.setCurrentIndex(idx)
        self.activate_cloud_btn = QPushButton("Use"); self.activate_cloud_btn.setFixedWidth(44)
        self.activate_cloud_btn.setStyleSheet("QPushButton { background: #081508; border: 1px solid #174517; color: #39ff14; font-size: 11px; border-radius: 4px; }")
        self.activate_cloud_btn.clicked.connect(self.activate_cloud_model)
        model_row.addWidget(self.cloud_model_combo, 1); model_row.addWidget(self.activate_cloud_btn)
        api_lay.addLayout(model_row)
        api_lay.addWidget(self._sec("API KEY"))
        key_row = QHBoxLayout(); key_row.setSpacing(6)
        self.cloud_api_key_input = QLineEdit()
        self.cloud_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.cloud_api_key_input.setPlaceholderText("API key for selected provider...")
        self.cloud_api_key_input.setText(CURRENT_SETTINGS.get("gemini_api_key", ""))
        self.cloud_api_key_input.editingFinished.connect(self.save_cloud_api_key)
        save_key_btn = QPushButton("Save"); save_key_btn.setFixedWidth(46)
        save_key_btn.clicked.connect(self.save_cloud_api_key)
        key_row.addWidget(self.cloud_api_key_input, 1); key_row.addWidget(save_key_btn)
        api_lay.addLayout(key_row)
        self.cloud_status_label = QLabel("Initializing status checking...")
        api_lay.addWidget(self.cloud_status_label)
        left_scroll_layout.addWidget(api_card)

        # Image analysis panel for cloud mode
        left_scroll_layout.addWidget(self._build_image_analysis_panel())
        left_scroll_layout.addStretch()

        left_scroll.setWidget(left_scroll_content)
        left.addWidget(left_scroll, 4)

        # Terminal in cloud panel too
        left.addWidget(self._build_terminal_panel())
        left.addStretch()

        body.addLayout(left, 4)

        right = QVBoxLayout(); right.setSpacing(6)
        right.addWidget(self._sec("INTEGRATED CLOUD CHAT INTERFACE"))
        self.cloud_chat_scroll = QScrollArea()
        self.cloud_chat_scroll.setWidgetResizable(True)
        self.cloud_chat_scroll.setStyleSheet("background: #050505; border: 1px solid #111;")
        self.cloud_chat_widget = QWidget()
        self.cloud_chat_layout = QVBoxLayout(self.cloud_chat_widget)
        self.cloud_chat_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.cloud_chat_scroll.setWidget(self.cloud_chat_widget)
        right.addWidget(self.cloud_chat_scroll, 1)
        chat_inp_row = QHBoxLayout(); chat_inp_row.setSpacing(6)
        self.cloud_msg_input = ChatInput()
        self.cloud_msg_input.setFixedHeight(40)
        self.cloud_msg_input.setPlaceholderText("Type cloud engine message & press Enter...")
        self.cloud_msg_input.sendRequested.connect(self.submit_embedded_cloud_chat)
        self.cloud_send_btn = QPushButton("Send")
        self.cloud_send_btn.setFixedHeight(40)
        self.cloud_send_btn.clicked.connect(self.submit_embedded_cloud_chat)
        chat_inp_row.addWidget(self.cloud_msg_input, 1); chat_inp_row.addWidget(self.cloud_send_btn)
        right.addLayout(chat_inp_row)
        body.addLayout(right, 6)
        lay.addLayout(body)
        self.update_cloud_status()
        return w

    def _update_cloud_models(self, provider):
        """Update the cloud model combo based on selected provider."""
        self.cloud_model_combo.clear()
        model_map = {
            "Gemini": ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"],
            "OpenAI": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "Mistral": ["mistral-large", "mistral-medium", "mistral-small"]
        }
        models = model_map.get(provider, ["gemini-3.5-flash", "gemini-2.5-flash"])
        self.cloud_model_combo.addItems(models)

    def activate_cloud_model(self):
        m = self.cloud_model_combo.currentText().strip()
        provider = self.api_provider_combo.currentText().lower()
        CURRENT_SETTINGS["active_cloud_provider"] = provider
        CURRENT_SETTINGS["active_gemini_model"] = m
        CURRENT_SETTINGS["active_model"] = f"cloud:{provider}:{m}"
        save_settings(CURRENT_SETTINGS)
        self.activate_model(f"cloud:{provider}:{m}")
        self.append_log(f"[Cloud] Architecture target set to: {provider}/{m}")

    def save_cloud_api_key(self):
        k = self.cloud_api_key_input.text().strip()
        CURRENT_SETTINGS["gemini_api_key"] = k
        save_settings(CURRENT_SETTINGS)
        self.update_cloud_status()
        self.append_log("[Cloud] Connection credentials saved.")

    def update_cloud_status(self):
        k = self.cloud_api_key_input.text().strip()
        if not k:
            self.cloud_status_label.setText("Status: No API key configured")
            self.cloud_status_label.setStyleSheet("color: #aa3333;")
        else:
            self.cloud_status_label.setText("Status: API key stored ✓")
            self.cloud_status_label.setStyleSheet("color: #33aa33;")

    def append_log(self, text):
        self.terminal_output.append(text)
        self.terminal_output.moveCursor(QTextCursor.MoveOperation.End)

    def check_ollama_status(self):
        base_url = self.ollama_url_input.text().strip()
        CURRENT_SETTINGS["ollama_url"] = base_url
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=2)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                cur = self.local_models_combo.currentText()
                self.local_models_combo.clear()
                if models:
                    self.local_models_combo.addItems(models)
                    if cur in models: self.local_models_combo.setCurrentText(cur)
                else: self.local_models_combo.addItem("No local models found")
            else:
                self.local_models_combo.clear(); self.local_models_combo.addItem("Offline status error")
        except requests.exceptions.RequestException:
            self.local_models_combo.clear(); self.local_models_combo.addItem("Engine unreachable")
        self.update_cloud_status()

    def target_local_model(self):
        m = self.local_models_combo.currentText()
        if m in ["No local models found", "Offline status error", "Engine unreachable", ""]: return
        self.activate_model(m)
        self.append_log(f"[Target] Selected model shifted to locally hosted: {m}")

    def start_ollama_pull(self):
        m = self.pull_model_input.text().strip()
        if not m: return
        self.pull_btn.setEnabled(False); self.pull_progress.setValue(0); self.pull_progress.show()
        self.pull_worker = OllamaPullWorker(m)
        self.pull_worker.log_signal.connect(self.append_log)
        self.pull_worker.progress_signal.connect(self.pull_progress.setValue)
        self.pull_worker.finished_signal.connect(self._finished_ollama_pull)
        self.pull_worker.start()

    def _finished_ollama_pull(self, success, name):
        self.pull_btn.setEnabled(True); self.pull_progress.hide()
        if success:
            self.append_log(f"[Ollama] Model {name} pulled successfully!")
            self.check_ollama_status()
        else: self.append_log(f"[Ollama Error] Deployment failed: {name}")

    def get_gemini_api_key(self):
        return (self.cloud_api_key_input.text().strip() or
                CURRENT_SETTINGS.get("gemini_api_key", "") or
                os.environ.get("GOOGLE_CLOUD_API_KEY", ""))

    # ── PAUSE / STOP ANALYSIS ──
    def pause_analysis(self):
        if self._analysis_paused:
            self._analysis_paused = False
            self.pause_btn.setText("⏸  Pause")
            self.pause_btn.setStyleSheet(
                "QPushButton { background: #201808; border: 1px solid #604010; color: #ffaa33; font-weight: bold; padding: 6px; }"
                "QPushButton:hover { background: #302010; }"
                "QPushButton:disabled { color: #222; border-color: #111; background: #080808; }")
            self.analysis_paused_state.emit(False)
        else:
            self._analysis_paused = True
            self.pause_btn.setText("▶  Resume")
            self.pause_btn.setStyleSheet(
                "QPushButton { background: #082010; border: 1px solid #106020; color: #33ff66; font-weight: bold; padding: 6px; }"
                "QPushButton:hover { background: #103020; }"
                "QPushButton:disabled { color: #222; border-color: #111; background: #080808; }")
            self.analysis_paused_state.emit(True)

    def stop_analysis(self):
        self._analysis_stopped = True
        if self._image_analysis_worker and self._image_analysis_worker.isRunning():
            self._image_analysis_worker.terminate()
            self._image_analysis_worker = None
        self.analyse_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self._analysis_paused = False
        self._analysis_stopped = False
        self._analysis_total_chunks = 0
        self._analysis_chunks_received = 0
        self._analysis_saved_count = 0
        self.progress_updated.emit(0)
        self.analysis_stopped_state.emit(True)
        self.append_log("[Analysis] Stopped by user.")

    # ── IMAGE ANALYSIS ──
    def run_image_analysis(self):
        img_path = self.img_selector_combo.currentData()
        prompt = self.analysis_prompt_input.toPlainText().strip()
        if not img_path or not os.path.exists(img_path):
            QMessageBox.warning(self, "No Image", "Please select a valid image from the library.")
            return
        if not prompt:
            QMessageBox.warning(self, "No Prompt", "Please enter an analysis prompt.")
            return
        if self._image_analysis_worker is not None and self._image_analysis_worker.isRunning():
            return

        self._analysis_paused = False
        self._analysis_stopped = False
        self._analysis_result_buffer = ""

        # Check access and collect security concerns
        concerns = []
        needs_permission = False
        if not PermissionManager.check_readable(img_path):
            needs_permission = True
            concerns.append("• This file requires elevated permissions to read.")

        is_cloud = self._ai_mode == "cloud"
        if is_cloud:
            concerns.append("• The image will be uploaded and sent to a third-party API server.")
            concerns.append("• This means image data leaves your local machine temporarily.")
            concerns.append("• Ensure you have rights to share this image with a third-party service.")
            concerns.append("• The selected cloud provider may log or retain your data per their privacy policy.")
            concerns.append("• Do NOT upload sensitive, personal, or confidential images.")

        if needs_permission:
            concerns.append("• Your system password will be used via sudo to grant temporary read access.")
            concerns.append("• This access automatically expires after 5 minutes.")
            concerns.append("• If the application crashes, file permissions may not be restored automatically.")
            concerns.append("• Temporary file access is granted to the current user only.")

        if concerns:
            warning_text = "\n".join(concerns)
            dlg = SecurityWarningDialog(warning_text, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

        # Handle permissions
        if needs_permission:
            success, needs_pwd, err = PermissionManager.grant_temp_access(img_path)
            if not success and needs_pwd:
                pwd_dlg = PasswordDialog(img_path, self)
                if pwd_dlg.exec() != QDialog.DialogCode.Accepted:
                    return
                success, _, err = PermissionManager.grant_temp_access(img_path, pwd_dlg.password)
            if not success:
                QMessageBox.critical(self, "Permission Error", f"Could not gain access to file:\n{err}")
                return
            self._access_expiry = time.time() + TEMP_ACCESS_DURATION
            self._access_timer.start(1000)
            self.access_status_lbl.setText("Access: Temporary access granted ✓")
            self.access_status_lbl.setStyleSheet("color: #33aa33; font-size: 10px;")

        self.analyse_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        if is_cloud:
            api_key = self.get_gemini_api_key()
            if not api_key:
                self.analyse_btn.setEnabled(True)
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                return
            model_name = self.cloud_model_combo.currentText().strip()
            self._image_analysis_worker = GeminiImageAnalysisWorker(model_name, prompt, api_key, [img_path])
            self._image_analysis_worker.output_ready.connect(self._append_analysis_chunk)
            self._image_analysis_worker.finished_signal.connect(self._finished_image_analysis)
            self._image_analysis_worker.start()
        else:
            active_model = CURRENT_SETTINGS.get("active_model", "")
            if not active_model or active_model.startswith("gemini:"):
                self.analyse_btn.setEnabled(True)
                self.pause_btn.setEnabled(False)
                self.stop_btn.setEnabled(False)
                return
            base_url = self.ollama_url_input.text().strip()
            self._image_analysis_worker = LocalImageAnalysisWorker(base_url, active_model, prompt, [img_path])
            self._image_analysis_worker.chunk_ready.connect(self._append_analysis_chunk)
            self._image_analysis_worker.finished_signal.connect(self._finished_image_analysis)
            self._image_analysis_worker.start()

        self.append_log(f"[Analysis] Started analysis of: {os.path.basename(img_path)}")

    def _append_analysis_chunk(self, chunk):
        if self._analysis_stopped:
            return
        if self._analysis_paused:
            # Buffer chunks while paused
            self._analysis_result_buffer += chunk
            return
        self._analysis_chunks_received += 1
        pct = min(95, int((self._analysis_chunks_received / (self._analysis_chunks_received + 5)) * 100))
        self.progress_updated.emit(pct)
        # Gradually save to AI database as data comes in
        if self._analysis_chunks_received % 5 == 0:
            self._analysis_saved_count += 1
            partial_data = self._analysis_result_buffer + chunk if self._analysis_result_buffer else chunk
            img_path = self.img_selector_combo.currentData() or ""
            try:
                run_ai_query(
                    "INSERT INTO ai_analysis_db (artifact_name, model_used, confidence_score, transcription, translation, notes) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"{os.path.basename(img_path)} (partial)", CURRENT_SETTINGS.get("active_model", ""), "N/A", partial_data[:200], "N/A", f"Partial save #{self._analysis_saved_count}"))
            except Exception:
                pass

    def _finished_image_analysis(self, success, err):
        self.analyse_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸  Pause")
        self._analysis_paused = False
        worker = self._image_analysis_worker
        self._image_analysis_worker = None
        if success and not self._analysis_stopped:
            result = self._analysis_result_buffer or "Analysis completed"
            img_path = self.img_selector_combo.currentData() or ""
            run_ai_query(
                "INSERT INTO ai_analysis_db (artifact_name, model_used, confidence_score, transcription, translation, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (os.path.basename(img_path), CURRENT_SETTINGS.get("active_model", ""), "N/A", result, "N/A", "Image Analysis"))
            self.progress_updated.emit(100)
            self.analysis_completed.emit(True)
            self.append_log("[Analysis] Completed and saved to AI database.")
        elif not self._analysis_stopped:
            self.append_log(f"[Analysis Error] {err}")
            self.progress_updated.emit(0)
        self._analysis_total_chunks = 0
        self._analysis_chunks_received = 0
        self._analysis_saved_count = 0

    # ── CLOUD CHAT HANDLERS ──
    def submit_embedded_cloud_chat(self):
        text = self.cloud_msg_input.toPlainText().strip()
        if not text: return
        if self._gemini_worker is not None and self._gemini_worker.isRunning():
            return
        self.cloud_msg_input.clear()
        user_bubble = ChatBubble(text, is_user=True)
        self.cloud_chat_layout.addWidget(user_bubble)
        self._current_cloud_ai_bubble = ChatBubble("", is_user=False)
        self.cloud_chat_layout.addWidget(self._current_cloud_ai_bubble)
        history_text = "\n".join(self.gemini_chat_history)
        full_prompt = (f"{history_text}\nUser: {text}\nAI:" if history_text else text)
        self.gemini_chat_history.append(f"User: {text}")
        model_name = self.cloud_model_combo.currentText().strip()
        api_key = self.get_gemini_api_key()
        if not api_key:
            self._current_cloud_ai_bubble.append_text("[Error: No API key configured. Save your API key first.]")
            return
        self.cloud_msg_input.setEnabled(False); self.cloud_send_btn.setEnabled(False)
        self._gemini_parts = []
        self._gemini_worker = GeminiGenerateWorker(model_name, full_prompt, api_key)
        self._gemini_worker.output_ready.connect(self._append_embedded_cloud_chunk)
        self._gemini_worker.finished_signal.connect(self._finished_embedded_cloud_chat)
        self._gemini_worker.start()
        sb = self.cloud_chat_scroll.verticalScrollBar()
        QTimer.singleShot(50, lambda: sb.setValue(sb.maximum()))

    def _append_embedded_cloud_chunk(self, chunk):
        if self._current_cloud_ai_bubble is not None:
            self._current_cloud_ai_bubble.append_text(chunk)
            self._gemini_parts.append(chunk)
            sb = self.cloud_chat_scroll.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _finished_embedded_cloud_chat(self, success, engine_msg):
        self.cloud_msg_input.setEnabled(True); self.cloud_send_btn.setEnabled(True)
        self.cloud_msg_input.setFocus()
        self._gemini_worker = None
        if success:
            full_response = "".join(self._gemini_parts)
            self.gemini_chat_history.append(f"AI: {full_response}")
            run_ai_query(
                "INSERT INTO ai_analysis_db (artifact_name, model_used, confidence_score, transcription, translation, notes) VALUES (?, ?, ?, ?, ?, ?)",
                ("Cloud Conversation Fragment", CURRENT_SETTINGS.get("active_model", ""), "N/A", "N/A", "N/A", full_response))
        else:
            self.append_log(f"[Cloud Chat Error] {engine_msg}")
            if self._current_cloud_ai_bubble is not None:
                self._current_cloud_ai_bubble.append_text(f"\n[Error: {engine_msg}]")

    # ── LOCAL CHAT HANDLERS ──
    def submit_embedded_local_chat(self):
        text = self.local_msg_input.toPlainText().strip()
        if not text: return
        if self._local_chat_thread is not None and self._local_chat_thread.isRunning():
            return
        self.local_msg_input.clear()
        user_bubble = ChatBubble(text, is_user=True)
        self.local_chat_layout.addWidget(user_bubble)
        self._current_local_ai_bubble = ChatBubble("", is_user=False)
        self.local_chat_layout.addWidget(self._current_local_ai_bubble)
        base_url = self.ollama_url_input.text().strip()
        active_model = CURRENT_SETTINGS.get("active_model", "")
        if not active_model or active_model.startswith("gemini:"):
            self._current_local_ai_bubble.append_text(
                "[System error: No local model targeted. Select a model from the dropdown and click 'Target Model'.]")
            return
        self._local_history.append({"role": "user", "content": text})
        self.local_msg_input.setEnabled(False); self.local_send_btn.setEnabled(False)
        self._local_response_chunks = []
        self._local_chat_thread = LocalChatWorker(base_url, active_model, self._local_history)
        self._local_chat_thread.chunk_ready.connect(self._append_embedded_local_chunk)
        self._local_chat_thread.finished_signal.connect(self._finished_embedded_local_chat)
        self._local_chat_thread.start()
        sb = self.local_chat_scroll.verticalScrollBar()
        QTimer.singleShot(50, lambda: sb.setValue(sb.maximum()))

    def _append_embedded_local_chunk(self, chunk):
        if self._current_local_ai_bubble is not None:
            self._current_local_ai_bubble.append_text(chunk)
            self._local_response_chunks.append(chunk)
            sb = self.local_chat_scroll.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _finished_embedded_local_chat(self, success, err):
        self.local_msg_input.setEnabled(True); self.local_send_btn.setEnabled(True)
        self.local_msg_input.setFocus()
        self._local_chat_thread = None
        if success:
            full_response = "".join(self._local_response_chunks)
            self._local_history.append({"role": "assistant", "content": full_response})
            run_ai_query(
                "INSERT INTO ai_analysis_db (artifact_name, model_used, confidence_score, transcription, translation, notes) VALUES (?, ?, ?, ?, ?, ?)",
                ("Local Conversation Fragment", CURRENT_SETTINGS.get("active_model", ""), "N/A", "N/A", "N/A", full_response))
        else:
            self.append_log(f"[Local Chat Error] {err}")
            if self._current_local_ai_bubble is not None:
                self._current_local_ai_bubble.append_text(f"\n[Connection Error: {err}]")

# ── AI Database Page ──────────────────────────────────────────────────────────

class AIDatabasePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        ref = QPushButton("⟳ Refresh"); ref.clicked.connect(self.load_data)
        chg_ai = QPushButton("📂 Change AI Dir")
        chg_ai.clicked.connect(lambda: self.migrate(QFileDialog.getExistingDirectory(self, "Select Folder")))
        df_ai = QPushButton("🔄 Reset AI Default")
        df_ai.clicked.connect(lambda: self.migrate(DEFAULT_AI_DB_DIR))
        clear_all = QPushButton("🗑️ Purge All Records")
        clear_all.setStyleSheet("QPushButton { background: #250505; color: #ff6666; border-color: #4a1515; }")
        clear_all.clicked.connect(self.purge_all_records)
        top.addWidget(ref); top.addWidget(chg_ai); top.addWidget(df_ai)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine); sep.setFrameShadow(QFrame.Shadow.Sunken)
        top.addWidget(sep); top.addWidget(clear_all); top.addStretch()
        layout.addLayout(top)
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["ID", "Artifact Name", "Model Used", "Confidence Score", "Transcription", "Translation", "Notes", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table)
        self.load_data()
    def load_data(self):
        self.table.setRowCount(0)
        for row in run_ai_query("SELECT * FROM ai_analysis_db ORDER BY id DESC", fetch=True):
            r_idx = self.table.rowCount(); self.table.insertRow(r_idx)
            for i in range(7):
                self.table.setItem(r_idx, i, QTableWidgetItem(str(row[i]) if row[i] is not None else ""))
            # Edit and Delete buttons
            act = QWidget(); a_lay = QHBoxLayout(act); a_lay.setContentsMargins(2, 2, 2, 2)
            e_btn = QPushButton("Edit")
            d_btn = QPushButton("Delete")
            e_btn.clicked.connect(lambda checked, r=row: self.edit_row(r))
            d_btn.clicked.connect(lambda checked, i=row[0]: self.delete_row(i))
            a_lay.addWidget(e_btn); a_lay.addWidget(d_btn)
            self.table.setCellWidget(r_idx, 7, act)
    def edit_row(self, row):
        d = EditAIDialog(row, self)
        if d.exec() == QDialog.DialogCode.Accepted:
            data = d.get_data()
            run_ai_query(
                "UPDATE ai_analysis_db SET artifact_name=?,model_used=?,confidence_score=?,transcription=?,translation=?,notes=? WHERE id=?",
                data)
            self.load_data()
    def delete_row(self, r_id):
        if QMessageBox.question(self, 'Delete', 'Delete this record?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            run_ai_query("DELETE FROM ai_analysis_db WHERE id = ?", (r_id,))
            self.load_data()
    def migrate(self, path):
        current_dir = AIDatabaseManager.get_dir()
        if not path or path == current_dir: return
        c = MigrationChoiceDialog(self)
        if c.exec() != QDialog.DialogCode.Accepted: return
        self.pd = FileTransferProgressDialog(self)
        self.worker = MigrationWorker(current_dir, path, c.choice)
        self.worker.progress_updated.connect(self.pd.progress_bar.setValue)
        self.worker.status_updated.connect(self.pd.status_label.setText)
        self.worker.migration_finished.connect(lambda s, m: self.mig_done(s, m, path))
        self.worker.start(); self.pd.exec()
    def mig_done(self, success, msg, path):
        self.pd.accept()
        if success:
            AIDatabaseManager.set_dir(path)
            CURRENT_SETTINGS["ai_db_directory"] = path
            save_settings(CURRENT_SETTINGS)
            self.load_data()
            QMessageBox.information(self, "Success", "AI database relocated successfully.")
        else: QMessageBox.critical(self, "Error", f"Failed: {msg}")
    def purge_all_records(self):
        if QMessageBox.question(self, 'Purge Data', 'Delete all AI analysis records permanently?',
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            run_ai_query("DELETE FROM ai_analysis_db")
            self.load_data()

class EditAIDialog(QDialog):
    def __init__(self, row_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit AI Analysis Record")
        self.entry_id = row_data[0]
        layout = QVBoxLayout(self)
        labels = ["Artifact Name", "Model Used", "Confidence Score", "Transcription", "Translation", "Notes"]
        self.inputs = {}
        for i, label in enumerate(labels):
            val = row_data[i + 1] if i + 1 < len(row_data) else ""
            inp = QTextEdit(val)
            inp.setFixedHeight(60)
            self.inputs[label] = inp
            layout.addWidget(QLabel(label))
            layout.addWidget(inp)
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
    def get_data(self):
        return (
            self.inputs["Artifact Name"].toPlainText(),
            self.inputs["Model Used"].toPlainText(),
            self.inputs["Confidence Score"].toPlainText(),
            self.inputs["Transcription"].toPlainText(),
            self.inputs["Translation"].toPlainText(),
            self.inputs["Notes"].toPlainText(),
            self.entry_id
        )

# ── Main Application Framework Window ─────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pandu v1.7")
        self.setMinimumSize(1000, 650)
        if os.path.exists(LOGO_PATH):
            app_icon = QIcon(LOGO_PATH)
            self.setWindowIcon(app_icon)
            QApplication.instance().setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow { background: #070707; }")

        w = QWidget(); self.setCentralWidget(w)
        layout = QVBoxLayout(w); layout.setContentsMargins(10, 10, 10, 10); layout.setSpacing(10)

        # Header
        header = QHBoxLayout(); header.setSpacing(10)
        self.logo_lbl = QLabel()
        self.logo_lbl.setFixedSize(48, 48)
        if os.path.exists(LOGO_PATH):
            self.logo_lbl.setPixmap(QPixmap(LOGO_PATH).scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.logo_lbl.setStyleSheet("background: #111; border: 1px dashed #333;")
        header.addWidget(self.logo_lbl)
        title_col = QVBoxLayout(); title_col.setSpacing(2)
        t_lbl = QLabel("PANDU"); t_lbl.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; letter-spacing: 3px;")
        sub_lbl = QLabel("INTELLIGENT COMPUTATIONAL ARCHAEOLOGICAL WORKSPACE"); sub_lbl.setStyleSheet("color: #444444; font-size: 9px; font-weight: bold; letter-spacing: 1px;")
        title_col.addWidget(t_lbl); title_col.addWidget(sub_lbl)
        header.addLayout(title_col); header.addStretch(); layout.addLayout(header)

        # Horizontal toolbar at top
        self.toolbar_frame = QFrame()
        self.toolbar_frame.setStyleSheet("QFrame { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 4px; padding: 2px; }")
        toolbar_layout = QHBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(6, 3, 6, 3)
        toolbar_layout.setSpacing(4)
        items = ["Data Entry", "Library", "AI Analysis", "AI Database", "Analytics", "Train", "Script Analysis"]
        self.btns = {}
        for item in items:
            btn = QPushButton(item)
            btn.setFixedSize(110, 28)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton { text-align: center; background: #0d0d0d; border: 1px solid #141414; color: #777; font-size: 11px; border-radius: 3px; }
                QPushButton:hover { background: #121212; color: #bbb; }
                QPushButton:checked { background: #161616; border-color: #bb4400; color: #bb4400; font-weight: bold; }
            """)
            btn.clicked.connect(lambda checked, name=item: self.navigate(name))
            toolbar_layout.addWidget(btn); self.btns[item] = btn
        self.btns["Data Entry"].setChecked(True)
        layout.addWidget(self.toolbar_frame)

        # Stacked pages
        self.stack = QStackedWidget()
        self.data_page = DataPage()
        self.library_page = LibraryPage()
        self.ai_analysis_page = AIAnalysisPage()
        self.ai_database_page = AIDatabasePage()
        self.data_page.goToLibrary.connect(self.saved_to_library)
        pages = [self.data_page, self.library_page, self.ai_analysis_page, self.ai_database_page]
        placeholder_names = ["Analytics", "Train", "Script Analysis"]
        for pn in placeholder_names:
            pages.append(PlaceholderPage(pn))
        self.pages = {}
        for idx, name in enumerate(items):
            self.pages[name] = idx
        for page in pages:
            self.stack.addWidget(page)
        layout.addWidget(self.stack, 1)

        # Progress bar at the very bottom
        progress_row = QHBoxLayout()
        self.analysis_progress_bar = QProgressBar()
        self.analysis_progress_bar.setFixedHeight(16)
        self.analysis_progress_bar.setValue(0)
        self.analysis_progress_bar.setStyleSheet("""
            QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
            QProgressBar::chunk { background: #2255aa; border-radius: 2px; }
        """)
        self.analysis_complete_label = QLabel("")
        self.analysis_complete_label.setFixedWidth(20)
        self.analysis_complete_label.setStyleSheet("color: #33ff33; font-size: 14px; font-weight: bold;")
        progress_row.addWidget(self.analysis_progress_bar, 1)
        progress_row.addWidget(self.analysis_complete_label)
        layout.addLayout(progress_row)

        self.status_footer = QLabel(f"Project Library DB: {DatabaseManager.get_dir()} | AI Database: {AIDatabaseManager.get_dir()}")
        self.status_footer.setStyleSheet("color: #333; font-size: 10px; padding-top: 4px;")
        layout.addWidget(self.status_footer)

        # Connect AI analysis page signals to progress bar
        self.ai_analysis_page.progress_updated.connect(self._update_analysis_progress)
        self.ai_analysis_page.analysis_completed.connect(self._on_analysis_completed)
        self.ai_analysis_page.analysis_paused_state.connect(self._on_analysis_paused)
        self.ai_analysis_page.analysis_stopped_state.connect(self._on_analysis_stopped)

    def _update_analysis_progress(self, value):
        self.analysis_progress_bar.setValue(value)
        if value == 100:
            self.analysis_progress_bar.setStyleSheet("""
                QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
                QProgressBar::chunk { background: #22aa55; border-radius: 2px; }
            """)
            self.analysis_complete_label.setText("✓")
        elif value > 0:
            self.analysis_complete_label.setText("")

    def _on_analysis_completed(self, completed):
        if completed:
            self.analysis_progress_bar.setValue(100)
            self.analysis_progress_bar.setStyleSheet("""
                QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
                QProgressBar::chunk { background: #22aa55; border-radius: 2px; }
            """)
            self.analysis_complete_label.setText("✓")

    def _on_analysis_paused(self, paused):
        if paused:
            self.analysis_progress_bar.setStyleSheet("""
                QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
                QProgressBar::chunk { background: #aaaa22; border-radius: 2px; }
            """)
        else:
            self.analysis_progress_bar.setStyleSheet("""
                QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
                QProgressBar::chunk { background: #2255aa; border-radius: 2px; }
            """)

    def _on_analysis_stopped(self, stopped):
        if stopped:
            self.analysis_progress_bar.setValue(0)
            self.analysis_progress_bar.setStyleSheet("""
                QProgressBar { background: #0a0a0a; border: 1px solid #1a1a1a; border-radius: 3px; text-align: center; color: #aaaaaa; font-size: 9px; }
                QProgressBar::chunk { background: #2255aa; border-radius: 2px; }
            """)
            self.analysis_complete_label.setText("")

    def closeEvent(self, event):
        # Check for any leftover temporary permissions and revoke them on startup recovery
        PermissionManager.revoke_all()
        super().closeEvent(event)

    def saved_to_library(self, data):
        self.library_page.load_data()
        self.navigate("Library")

    def navigate(self, name):
        if name in self.pages:
            for b in self.btns.values(): b.setChecked(False)
            self.btns[name].setChecked(True)
            self.stack.setCurrentIndex(self.pages[name])
            if name == "AI Database":
                self.ai_database_page.load_data()
            elif name == "AI Analysis":
                self.ai_analysis_page.update_banner_style()
                self.ai_analysis_page.refresh_library_images()
            self.status_footer.setText(
                f"Project Library DB: {DatabaseManager.get_dir()} | AI Database: {AIDatabaseManager.get_dir()}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QIcon(LOGO_PATH))
    # On startup, check for any leftover temporary permissions and revoke
    PermissionManager.revoke_all()
    window = MainWindow()
    window.show()
    sys.exit(app.exec())