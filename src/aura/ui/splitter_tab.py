import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aura.audio.splitter import SmartSplitterThread


class SplitterTab(QWidget):
    def __init__(self):
        super().__init__()
        self.file_path = None
        self.output_dir = None
        self.thread = None
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        header = QLabel("✂️ Intelligent Track Splitter")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #00bcd4;")
        layout.addWidget(header, alignment=Qt.AlignmentFlag.AlignCenter)

        desc = QLabel("Automatically find speaker pauses or breaths for cutting to avoid abrupt interruptions.")
        desc.setStyleSheet("font-size: 14px; color: #aaaaaa;")
        layout.addWidget(desc, alignment=Qt.AlignmentFlag.AlignCenter)

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("Target Segment Length (minutes):"))
        self.spin_target = QSpinBox()
        self.spin_target.setRange(5, 120)
        self.spin_target.setValue(40)
        settings_layout.addWidget(self.spin_target)

        settings_layout.addWidget(QLabel(" Tolerance (minutes):"))
        self.spin_tol = QSpinBox()
        self.spin_tol.setRange(1, 15)
        self.spin_tol.setValue(5)
        settings_layout.addWidget(self.spin_tol)
        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("1. Select Source Audio")
        self.btn_select.setFixedHeight(50)
        self.btn_select.clicked.connect(self.select_file)

        self.btn_outdir = QPushButton("2. Select Output Folder")
        self.btn_outdir.setFixedHeight(50)
        self.btn_outdir.clicked.connect(self.select_outdir)

        self.btn_start = QPushButton("3. Start Intelligent Splitting")
        self.btn_start.setFixedHeight(50)
        self.btn_start.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_split)
        self.btn_start.setEnabled(False)

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_outdir)
        btn_layout.addWidget(self.btn_start)
        layout.addLayout(btn_layout)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setStyleSheet("color: #ff9800;")
        layout.addWidget(self.lbl_file)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        layout.addWidget(self.log_area)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select audio to split",
            "",
            "Audio/Video Files (*.mp3 *.wav *.m4a *.mp4 *.flac *.ogg *.aac *.mkv *.mov *.wma *.aiff *.opus)",
        )
        if path:
            self.file_path = path
            self.output_dir = os.path.dirname(path)
            self.update_status()

    def select_outdir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if dir_path:
            self.output_dir = dir_path
            self.update_status()

    def update_status(self):
        if self.file_path and self.output_dir:
            file_name = os.path.basename(self.file_path)
            self.lbl_file.setText(f"Source: {file_name} | Output to: {self.output_dir}")
            self.btn_start.setEnabled(True)

    def start_split(self):
        if not self.file_path or not self.output_dir:
            return
        self.btn_select.setEnabled(False)
        self.btn_outdir.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_area.clear()

        self.thread = SmartSplitterThread(
            self.file_path,
            self.output_dir,
            self.spin_target.value(),
            self.spin_tol.value(),
        )
        self.thread.log_signal.connect(self.append_log)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.error_signal.connect(self.handle_error)
        self.thread.finished_signal.connect(self.process_finished)
        self.thread.start()

    def append_log(self, text):
        self.log_area.append(text)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def handle_error(self, err_msg):
        QMessageBox.critical(self, "Error", f"An error occurred during processing:\n{err_msg}")
        self.reset_ui()

    def process_finished(self):
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Completed", "Intelligent splitting completed!")
        self.reset_ui()

    def reset_ui(self):
        self.btn_select.setEnabled(True)
        self.btn_outdir.setEnabled(True)
        self.btn_start.setEnabled(True)
