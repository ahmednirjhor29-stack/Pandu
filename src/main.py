import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QFileDialog, QPushButton, QTextEdit, QLineEdit, QComboBox, QHBoxLayout, QListWidget, QListWidgetItem, QCheckBox, QMessageBox, QProgressBar, QScrollArea
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import os

class ImageUploadWidget(QWidget):
    imageUploaded = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.image_label = QLabel("Drag and drop an image here")
        self.image_label.setFixedSize(300, 200)
        self.main_layout.addWidget(self.image_label)

        self.upload_button = QPushButton("Browse")
        self.upload_button.clicked.connect(self.browse_image)
        self.main_layout.addWidget(self.upload_button)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(None, "Select Image", "", "Images (*.png *.xpm *.jpg);;All Files (*)")
        if file_path:
            pixmap = QPixmap(file_path).scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.imageUploaded.emit(pixmap)

class LetterInfoForm(QWidget):
    def __init__(self):
        super().__init__()
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.name_input = QLineEdit("Letter Name or Symbol")
        self.main_layout.addWidget(QLabel("Name:"))
        self.main_layout.addWidget(self.name_input)

        self.writing_system_combo = QComboBox()
        self.writing_system_combo.addItems(["Devanagari", "Cuneiform", "Hieroglyphics", "Latin", "Arabic", "Greek", "Hebrew", "Brahmi", "Phoenician", "Add New"])
        self.main_layout.addWidget(QLabel("Writing System:"))
        self.main_layout.addWidget(self.writing_system_combo)

        self.time_period_start = QLineEdit()
        self.time_period_end = QLineEdit()
        self.bce_ce_toggle = QComboBox()
        self.bce_ce_toggle.addItems(["BCE", "CE"])
        self.main_layout.addWidget(QLabel("Time Period (Start/End):"))
        self.main_layout.addWidget(self.time_period_start)
        self.main_layout.addWidget(self.time_period_end)
        self.main_layout.addWidget(self.bce_ce_toggle)
        

        self.region_input = QLineEdit("Region/Origin")
        self.main_layout.addWidget(QLabel("Region/Origin:"))
        self.main_layout.addWidget(self.region_input)

        self.material_combo = QComboBox()
        self.material_combo.addItems(["Stone", "Clay Tablet", "Papyrus", "Parchment", "Metal", "Ink on Paper", "Other"])
        self.main_layout.addWidget(QLabel("Material Found On:"))
        self.main_layout.addWidget(self.material_combo)

        self.tags_input = QLineEdit("Comma-separated Tags")
        self.main_layout.addWidget(QLabel("Tags:"))
        self.main_layout.addWidget(self.tags_input)

        self.notes_textedit = QTextEdit()
        self.main_layout.addWidget(QLabel("Notes:"))
        self.main_layout.addWidget(self.notes_textedit)

class LetterPartNavigator(QWidget):
    def __init__(self, image_pixmap):
        super().__init__()
        self.image_pixmap = image_pixmap
        self.image_label = QLabel()
        self.image_label.setPixmap(image_pixmap)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.setLayout(self.main_layout)

class SaveButton(QWidget):
    def __init__(self, database):
        super().__init__()
        self.database = database
        self.save_button = QPushButton("Save to Library")
        self.save_button.clicked.connect(self.save_to_database)
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.save_button)
        self.setLayout(self.main_layout)

    def save_to_database(self):
        # Placeholder for saving data to the SQLite database
        print("Saving data to database...")

class MainWindow(QMainWindow):
    def __init__(self, database_path):
        super().__init__()

        self.setWindowTitle("ScriptLens Vibe Coding Prompt")
        self.setGeometry(100, 100, 800, 600)

        # Set the application icon
        self.setWindowIcon(QIcon("assets/adobe_icon.png"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout()
        body_layout = QHBoxLayout()

        # Left Sidebar Navigation
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("Navigation"))
        sidebar_layout.addWidget(QLabel("Data Entry"))
        sidebar_layout.addWidget(QLabel("Library"))
        sidebar_layout.addWidget(QLabel("AI Analysis"))
        sidebar_layout.addWidget(QLabel("Review & Corrections"))
        sidebar_layout.addWidget(QLabel("Results & Reports"))
        sidebar_layout.addWidget(QLabel("Unknown Script Analyser"))

        body_layout.addLayout(sidebar_layout)

        # Main Content Area
        content_widget = QWidget()
        content_layout = QVBoxLayout()

        self.image_upload_widget = ImageUploadWidget()
        self.image_upload_widget.imageUploaded.connect(self.on_image_uploaded)
        content_layout.addWidget(self.image_upload_widget)

        self.letter_info_form = LetterInfoForm()
        content_layout.addWidget(self.letter_info_form)

        self.letter_part_navigator = None

        self.save_button = SaveButton(database_path)
        content_layout.addWidget(self.save_button)

        content_widget.setLayout(content_layout)

        # Bottom Status Bar
        status_bar = QLabel("Current Project: None - Storage Usage: 0GB")
        body_layout.addWidget(content_widget)
        main_layout.addLayout(body_layout)
        main_layout.addWidget(status_bar)
        main_widget.setLayout(main_layout)

    def on_image_uploaded(self, pixmap):
        self.letter_part_navigator = LetterPartNavigator(pixmap)
        content_layout = self.centralWidget().layout()
        content_layout.addWidget(self.letter_part_navigator)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    database_path = os.path.join(os.getcwd(), "data/scriptlens.db")
    window = MainWindow(database_path)
    window.show()
    sys.exit(app.exec())
